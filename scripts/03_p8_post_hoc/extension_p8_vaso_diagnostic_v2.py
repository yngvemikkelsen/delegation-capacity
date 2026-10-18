# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
# ]
# ///
"""
extension_p8_vaso_diagnostic_v2.py -- POST HOC checks of the P8 vasopressor
result. Not part of the prespecified analysis; report as post hoc.

Two questions raised in review:
  (1) Vocabulary. The delegate's question names five generic agents and
      "intravenous"; the documentation regex also matches brand names
      (levophed, neosynephrine) and the umbrella terms pressor/vasopressor.
      Are the misses (and the gradient) concentrated in notes that document
      vasopressors only by brand or umbrella term?
  (2) Documentation. A regex hit shows that a term occurs, not that the note
      documents administration during this admission. How often does the
      flag hold up on blinded reading, and does the gradient survive when
      "documented" is defined by the reader?

v2 writes to a NEW folder (default ~/extension_results/p8_adjudication).
The v1 folder and its review sheet are not touched.

MODE 1 (default): term check + blinded review sheet
  * re-scans the vasopressor primary-set notes with the P8 regex and
    negation window (agreement with P8's text_mentioned is checked)
  * classifies every mention by context as in v1 (exclusion contexts first,
    then specific term, then administration context)
  * TERM CHECK. Administration-relevant mentions (classes specific_term and
    ambiguous_admin_context) are grouped as
        generic   norepinephrine, epinephrine, phenylephrine, vasopressin,
                  dopamine (the agents named in the delegate's question)
        brand     levophed, neosynephrine / neo-synephrine
        umbrella  pressor(s), vasopressor(s)
    Note category, hierarchical: generic > brand_only > umbrella_only;
    a note with no administration-relevant mention is loose_only.
    Reported: notes, misses and miss rate by category and quartile; category
    shares by quartile; logistic models with cluster-robust SE by subject:
        M1  miss ~ q_num                          (reproduces P8 S5)
        M2  miss ~ q_num + C(category)            (gradient net of vocabulary)
        M3  miss ~ q_num, generic notes only      (question names the agent)
  * BLINDED REVIEW SHEET (never overwritten): per quartile a random sample
    of misses and of hits (default 50 + 10), shuffled; merged excerpts around
    every mention. No quartile, LLM answer, class or category is shown.

MODE 2: --labels <labelled sheet>
  (a) label distribution by quartile among reviewed misses and hits
  (b) documentation flag on blinded reading: share labelled A among reviewed
      misses (t) and hits (h) by quartile, Wilson 95% CIs; trend of t over
      quartiles (logistic, A vs O/N, U excluded); weighted share A in the
      whole primary set by quartile
  (c) rule-based context classes vs labels: agreement, Cohen's kappa
  (d) adjudicated gradient: miss rate with documentation as judged by the
      reader, per quartile  M_k t_k / (M_k t_k + H_k h_k), trend OR with a
      two-stage bootstrap CI: patients resampled with replacement (counts
      M_k, H_k), then t_k and h_k drawn from Beta(A + 0.5, not-A + 0.5) for
      each quartile x answer cell, so cells where every reviewed item is A
      still carry uncertainty; U excluded (main), U as A, U as not A
  (e) share labelled A among reviewed misses by term category
  Scope: the review measures how often the documentation flag holds up
  (precision). Notes that document vasopressors only in forms the regex
  does not match (e.g. "norepi", "levo", "neo") never enter the primary set,
  so the review cannot assess what the flag misses.

LABELS (column "label" in review_sheet.csv; one letter)
  A  administered: the excerpts document intravenous norepinephrine,
     epinephrine, phenylephrine, vasopressin or dopamine given during THIS
     admission, named generically, by brand, or as pressors/vasopressors.
     If administration before and during this admission is documented, A.
  O  other admission: administration is documented, but only before this
     admission (history, earlier admission, another hospital before transfer)
  N  not administration: the mentions do not describe vasopressor
     administration (allergy, local anaesthetic, eye drops, dopamine agonist,
     a plan or consideration only, negated)
  U  unclear from the excerpts
  Enter one letter (A, O, N, U; either case) or the words administered,
  other, not, unclear. Anything else is reported and excluded. Optional free
  text in "comment". Save as CSV (comma or semicolon; UTF-8 or Windows
  encoding). Do not open the key/ subfolder before labelling: it holds the
  quartile, LLM answer and term category of every item.

USAGE
  uv run --offline extension_p8_vaso_diagnostic_v2.py
  uv run --offline extension_p8_vaso_diagnostic_v2.py --labels ~/extension_results/p8_adjudication/review_sheet.csv

OUTPUTS (in --out)
  diag2_summary.txt         mode 1 results (aggregates only)
  review_sheet.csv          blinded sheet to label (never overwritten)
  key/review_key.csv        hidden key (never overwritten)
  key/diag2_notes.csv       one row per vasopressor primary-set note
  key/diag2_mentions.csv    one row per mention: term, group, class, context
  adjudicated_summary.txt   mode 2 results

  Category 'brand_only' means a brand name and no generic name (an umbrella
  term may also occur); 'umbrella_only' means pressor/vasopressor only.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HOME = Path.home()
DEF_P8 = HOME / "extension_results" / "p8"
DEF_OUT = HOME / "extension_results" / "p8_adjudication"
DEF_MIMIC_NOTE = HOME / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"

# identical to extension_p8_error_demand_v2.py
VASO_TEXT_RE = re.compile(
    r"(?:norepinephrine|levophed|epinephrine|phenylephrine|"
    r"neo-?synephrine|vasopressin|dopamine|pressors?|vasopressors?)",
    re.IGNORECASE)
NEGATION_RE = re.compile(
    r"\b(?:no|not|non|without|denies|denied|never|negative for|free of|"
    r"ruled out|r/o|declined|deferred)\b[^.;:!?]{0,30}$",
    re.IGNORECASE)

# identical to extension_p8_vaso_diagnostic.py (v1): context classes, checked
# in this order on a window of 100 characters before and 60 after the mention
EXCLUDE_CTX = [
    ("local_anaesthetic", re.compile(
        r"lidocaine|\blido\b|xylocaine|bupivacaine|marcaine|ropivacaine|"
        r"mepivacaine|local an(?:a)?esth")),
    ("allergy", re.compile(r"allerg|epi-?pen|auto-?injector")),
    ("ophthalmic_topical", re.compile(
        r"ophth|eye ?drop|\bgtts? o[uds]\b|nasal|spray|topical|mydria")),
    ("neuro_agonist", re.compile(
        r"agonist|carbidopa|sinemet|levodopa|ropinirole|pramipexole|parkinson")),
]
SPECIFIC = re.compile(
    r"^(?:norepinephrine|levophed|vasopressin|neo-?synephrine|"
    r"pressors?|vasopressors?)$")
ADMIN_CTX = re.compile(
    r"drip|infusion|\bgtt\b|started on|requir|wean|titrat|mcg/kg|mcg/min|"
    r"units?/min|pressor support|\bshock\b|hypotens|on pressors|off pressors")
STRICT_CLASSES = {"specific_term", "ambiguous_admin_context"}

# v2: term groups
GENERIC = {"norepinephrine", "epinephrine", "phenylephrine", "vasopressin", "dopamine"}
BRAND = {"levophed", "neosynephrine", "neo-synephrine"}
UMBRELLA = {"pressor", "pressors", "vasopressor", "vasopressors"}
CATEGORIES = ["generic", "brand_only", "umbrella_only", "loose_only"]
LABELS = {"a": "A", "administered": "A", "o": "O", "other": "O",
          "n": "N", "not": "N", "u": "U", "unclear": "U"}


def log(m: str = "") -> None:
    print(m, flush=True)


def find_table(root: Path, name: str) -> Path:
    hits: list[Path] = []
    for ext in (".csv.gz", ".csv"):
        hits.extend(root.rglob(f"{name}{ext}"))
    if not hits:
        sys.exit(f"[fatal] table '{name}' not found under {root}")
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def classify(term: str, ctx: str) -> str:
    for name, pat in EXCLUDE_CTX:
        if pat.search(ctx):
            return name
    if SPECIFIC.match(term):
        return "specific_term"
    if ADMIN_CTX.search(ctx):
        return "ambiguous_admin_context"
    return "ambiguous_no_context"


def term_group(term: str) -> str:
    if term in GENERIC:
        return "generic"
    if term in BRAND:
        return "brand"
    if term in UMBRELLA:
        return "umbrella"
    raise ValueError(f"unexpected term {term!r}")


def scan(text: str) -> list[dict]:
    """text must be lower-cased."""
    out = []
    for m in VASO_TEXT_RE.finditer(text):
        if NEGATION_RE.search(text[max(0, m.start() - 40):m.start()]):
            continue
        term = m.group(0).lower()
        ctx = text[max(0, m.start() - 100):m.end() + 60]
        out.append({"term": term, "group": term_group(term),
                    "cls": classify(term, ctx), "start": m.start(), "end": m.end()})
    return out


def note_category(ms: list[dict]) -> str:
    groups = {m["group"] for m in ms if m["cls"] in STRICT_CLASSES}
    if "generic" in groups:
        return "generic"
    if "brand" in groups:
        return "brand_only"
    if "umbrella" in groups:
        return "umbrella_only"
    return "loose_only"


def excerpt(text: str, mentions: list[dict], width: int = 250) -> str:
    """Merged excerpts around all mentions, separated by ' ||| '."""
    spans = []
    for m in sorted(mentions, key=lambda d: d["start"]):
        a, b = max(0, m["start"] - width), min(len(text), m["end"] + width)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    parts = []
    for a, b in spans:
        s = re.sub(r"\s+", " ", text[a:b]).strip()
        parts.append(("..." if a > 0 else "") + s + ("..." if b < len(text) else ""))
    return " ||| ".join(parts)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, min(p, c - h)), min(1.0, max(p, c + h))


def fit_or(df: pd.DataFrame, y: str, target: str, covars: list[str],
           cat: str | None, label: str) -> str:
    """Logistic GLM, cluster-robust SE by subject_id; OR for `target`."""
    if len(df) < 30 or df[y].nunique() < 2:
        return f"{label:<52} skipped (n={len(df)})"
    X = df[[target] + covars].astype(float)
    if cat and df[cat].nunique() > 1:
        levels = [c for c in CATEGORIES if c in set(df[cat])]
        d = pd.get_dummies(pd.Categorical(df[cat], categories=levels),
                           prefix=cat, drop_first=True, dtype=float)
        d.index = df.index
        X = pd.concat([X, d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": df["subject_id"].values})
    except Exception as exc:  # noqa: BLE001
        return f"{label:<52} failed: {exc}"
    lo, hi = res.conf_int().loc[target]
    return (f"{label:<52} OR {math.exp(res.params[target]):.3f} "
            f"({math.exp(lo):.3f}-{math.exp(hi):.3f}) p={res.pvalues[target]:.4g} "
            f"n={len(df)} events={int(df[y].sum())}")


def load_per(p8_dir: Path) -> pd.DataFrame:
    path = p8_dir / "p8_per_note_fact.csv"
    if not path.exists():
        sys.exit(f"[fatal] {path} not found")
    per = pd.read_csv(path, dtype={"note_id": str}, low_memory=False)
    for c in ("structured_present", "text_mentioned", "possibly_truncated"):
        per[c] = per[c].astype(str).str.lower() == "true"
    per["llm_answer"] = per["llm_answer"].astype(str).str.lower()
    per = per[per["llm_answer"].isin(["yes", "no"]) & ~per["possibly_truncated"]].copy()
    per["miss"] = (per["llm_answer"] == "no").astype(int)
    log(f"P8 per-note table <- {path}")
    return per


# ---------------------------------------------------------------------------
# mode 1
# ---------------------------------------------------------------------------

def mode_diagnose(args) -> None:
    out = args.out
    per = load_per(args.p8_dir)
    vaso_all = per[per["fact"] == "vasopressor"].copy()
    vaso = vaso_all[vaso_all["structured_present"] & vaso_all["text_mentioned"]].copy()
    log(f"vasopressor primary-set notes: {len(vaso)}  misses: {int(vaso.miss.sum())}")

    need = set(vaso_all["note_id"])
    path = find_table(args.mimic_note, "discharge")
    log(f"discharge <- {path}")
    raw: dict[str, str] = {}
    for ch in pd.read_csv(path, usecols=["note_id", "text"], dtype={"note_id": str},
                          chunksize=20_000, low_memory=False):
        h = ch[ch["note_id"].isin(need)]
        raw.update(zip(h["note_id"], h["text"].fillna("").astype(str)))
    texts = {n: t.lower() for n, t in raw.items()}
    missing = need - set(texts)
    if missing:
        sys.exit(f"[fatal] {len(missing)} note_ids not in discharge table")

    rescan = {nid: scan(texts[nid]) for nid in need}
    agree = (vaso_all["note_id"].map(lambda n: len(rescan[n]) > 0)
             == vaso_all["text_mentioned"]).mean()
    log(f"agreement with P8 text_mentioned (vasopressor, all notes): {agree:.4%}")

    mrows, nrows = [], []
    for r in vaso.itertuples():
        ms = rescan[r.note_id]
        for m in ms:
            mrows.append({"note_id": r.note_id, "q_num": r.q_num, "miss": r.miss,
                          "term": m["term"], "group": m["group"], "cls": m["cls"],
                          "context": re.sub(r"\s+", " ", texts[r.note_id][
                              max(0, m["start"] - 100):m["end"] + 60])})
        classes = {m["cls"] for m in ms}
        nrows.append({"note_id": r.note_id, "subject_id": r.subject_id,
                      "hadm_id": r.hadm_id, "q_num": int(r.q_num), "miss": int(r.miss),
                      "n_mentions": len(ms),
                      "strict": bool(classes & STRICT_CLASSES),
                      "category": note_category(ms),
                      "classes": ";".join(sorted(classes))})
    ment = pd.DataFrame(mrows)
    notes = pd.DataFrame(nrows)
    keydir = out / "key"
    keydir.mkdir(exist_ok=True)
    ment.to_csv(keydir / "diag2_mentions.csv", index=False)
    notes.to_csv(keydir / "diag2_notes.csv", index=False)

    L = ["POST HOC DIAGNOSTIC v2 -- not part of the prespecified P8 analysis", "",
         f"Vasopressor primary-set notes {len(notes)}, misses {int(notes.miss.sum())}",
         f"Agreement of re-scan with P8 text_mentioned: {agree:.4%}"
         + ("" if agree >= 0.9999 else "   !! differs from the P8 run -- check"), ""]

    L += ["1. Mentions by term group and context class (counts)", ""]
    L.append(pd.crosstab(ment["group"], ment["cls"], margins=True).to_string())

    L += ["", "2. Notes by term category and quartile", ""]
    rows = []
    for cat in CATEGORIES + ["ALL"]:
        for q in list(range(4)) + ["all"]:
            s = notes if cat == "ALL" else notes[notes["category"] == cat]
            s = s if q == "all" else s[s["q_num"] == q]
            p, lo, hi = wilson(int(s["miss"].sum()), len(s))
            rows.append({"category": cat, "quartile": q if q == "all" else q + 1,
                         "notes": len(s), "misses": int(s["miss"].sum()),
                         "miss_rate": p, "lo95": lo, "hi95": hi})
    tab = pd.DataFrame(rows)
    L.append(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    L += ["", "3. Share of notes in each category, by quartile", ""]
    sh = pd.crosstab(notes["category"], notes["q_num"] + 1, normalize="columns")
    L.append(sh.reindex([c for c in CATEGORIES if c in sh.index]).to_string(
        float_format=lambda v: f"{v:.3f}"))

    L += ["", "4. Models (logistic, OR per quartile step, cluster-robust by subject)", ""]
    L.append(fit_or(notes, "miss", "q_num", [], None, "M1 miss ~ q_num (reproduces P8 S5)"))
    L.append(fit_or(notes, "miss", "q_num", [], "category",
                    "M2 miss ~ q_num + category"))
    L.append(fit_or(notes[notes["category"] == "generic"], "miss", "q_num", [], None,
                    "M3 miss ~ q_num, generic notes only"))
    L += ["", "Reading: if M2 and M3 keep the M1 gradient, question vocabulary does",
          "not explain it; if the gradient vanishes in M2/M3 and brand_only or",
          "umbrella_only notes have high miss rates and grow with quartile, it does."]

    sheet_path, key_path = out / "review_sheet.csv", keydir / "review_key.csv"
    if sheet_path.exists() or key_path.exists():
        L += ["", "review sheet or key already exists -- neither is overwritten "
              f"({sheet_path}, {key_path})"]
    else:
        rng = np.random.default_rng(args.seed)
        pick = []
        for q in range(4):
            for miss_flag, k in ((1, args.review_misses), (0, args.review_hits)):
                pool = notes[(notes["q_num"] == q) & (notes["miss"] == miss_flag)]
                pick.append(pool.sample(n=min(k, len(pool)),
                                        random_state=int(rng.integers(1 << 31))))
        sample = (pd.concat(pick).sample(frac=1.0, random_state=args.seed)
                  .reset_index(drop=True))
        sample["review_id"] = [f"R{i + 1:03d}" for i in range(len(sample))]
        # positions come from the lower-cased text; show original case when
        # lower-casing did not change the length
        sample["excerpts"] = [excerpt(raw[n] if len(raw[n]) == len(texts[n]) else texts[n],
                                      rescan[n]) for n in sample["note_id"]]
        sheet = sample[["review_id", "excerpts"]].copy()
        sheet["label"] = ""
        sheet["comment"] = ""
        sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
        sample[["review_id", "note_id", "hadm_id", "subject_id", "q_num", "miss",
                "strict", "category", "classes"]].to_csv(key_path, index=False)
        cells = (sample.assign(answer=np.where(sample["miss"] == 1, "miss", "hit"),
                               quartile=sample["q_num"] + 1)
                 .groupby(["answer", "quartile"]).size().unstack("quartile"))
        L += ["", f"Review sheet: {len(sheet)} items (requested {args.review_misses} "
              f"misses + {args.review_hits} hits per quartile), shuffled; no quartile, "
              f"LLM answer, class or category shown -> {sheet_path}",
              "Items sampled per cell (fewer than requested where the cell is smaller):",
              cells.to_string(),
              "Label each row A / O / N / U (see the script header). Do not open the "
              "key/ subfolder before labelling."]

    (out / "diag2_summary.txt").write_text("\n".join(L) + "\n")
    for l in L:
        log(l)


# ---------------------------------------------------------------------------
# mode 2
# ---------------------------------------------------------------------------

def read_labels(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if path.suffix.lower() in (".xlsx", ".xls"):
        sys.exit("[fatal] save the labelled sheet as CSV")
    lab = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            lab = pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    if lab is None:
        sys.exit(f"[fatal] cannot decode {path}")
    lab.columns = [c.strip().lower().lstrip("\ufeff") for c in lab.columns]
    if not {"review_id", "label"} <= set(lab.columns):
        sys.exit(f"[fatal] {path} needs columns review_id and label; has {list(lab.columns)}")
    lab["review_id"] = lab["review_id"].fillna("").str.strip()
    raw = lab["label"].fillna("").str.strip()
    lab["label"] = raw.str.lower().map(LABELS)
    bad = sorted({f"{r}: '{v}'" for r, v, m in zip(lab["review_id"], raw, lab["label"])
                  if v and pd.isna(m)})
    return lab[["review_id", "label"]], bad


def corrected(key: pd.DataFrame, counts: pd.DataFrame, u_as: str | None) -> pd.DataFrame:
    """Miss rate with documentation as judged by the reader."""
    k = key.copy()
    if u_as:
        k.loc[k["label"] == "U", "label"] = u_as
    k = k[k["label"].isin(["A", "O", "N"])]
    rows = []
    for q in range(4):
        c = counts.loc[q]
        s_m = k[(k["q_num"] == q) & (k["miss"] == 1)]
        s_h = k[(k["q_num"] == q) & (k["miss"] == 0)]
        t = (s_m["label"] == "A").mean() if len(s_m) else float("nan")
        h = (s_h["label"] == "A").mean() if len(s_h) else float("nan")
        ev = c["misses"] * t
        den = c["hits"] * h + ev
        rows.append({"q_num": q, "misses": int(c["misses"]), "hits": int(c["hits"]),
                     "rev_misses": len(s_m), "t_A_misses": t,
                     "rev_hits": len(s_h), "h_A_hits": h,
                     "share_A_all": (ev + c["hits"] * h) / (c["misses"] + c["hits"]),
                     "corr_events": ev, "corr_denom": den,
                     "corr_rate": ev / den if den > 0 else float("nan")})
    return pd.DataFrame(rows)


def trend_or(tab: pd.DataFrame) -> float:
    t = tab.dropna(subset=["corr_rate"])
    t = t[t["corr_denom"] > 0]
    if len(t) < 2 or t["corr_rate"].nunique() < 2:
        return float("nan")
    X = sm.add_constant(t["q_num"].astype(float))
    try:
        res = sm.GLM(t["corr_rate"], X, family=sm.families.Binomial(),
                     var_weights=t["corr_denom"]).fit()
    except Exception:  # noqa: BLE001
        return float("nan")
    return math.exp(res.params["q_num"])


def kappa(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def mode_adjudicate(args) -> None:
    out = args.out
    key_path = out / "key" / "review_key.csv"
    if not key_path.exists():
        sys.exit(f"[fatal] {key_path} not found; run mode 1 first (v2 folder)")
    key = pd.read_csv(key_path, dtype={"note_id": str, "review_id": str})
    if "category" not in key.columns:
        sys.exit(f"[fatal] {key_path} is not a v2 key (no 'category' column)")
    labels, bad = read_labels(args.labels)
    key = key.merge(labels, on="review_id", how="left")
    n_blank = int(key["label"].isna().sum())
    notes = pd.read_csv(out / "key" / "diag2_notes.csv", dtype={"note_id": str})
    counts = (notes.groupby("q_num")
              .agg(misses=("miss", "sum"), hits=("miss", lambda s: int((s == 0).sum())))
              .reindex(range(4)).fillna(0))

    L = ["POST HOC ADJUDICATION v2 -- vasopressor documentation judged blind", ""]
    if n_blank:
        L += [f"!! {n_blank} items unlabelled or with unrecognised labels -- excluded", ""]
    if bad:
        L += ["!! unrecognised label values: " + "; ".join(bad), ""]
    lab = key.dropna(subset=["label"]).copy()
    lab["strict"] = lab["strict"].astype(str).str.lower() == "true"
    lab["answer"] = np.where(lab["miss"] == 1, "miss", "hit")

    L += ["(a) Labels by quartile and LLM answer (counts)", ""]
    L.append(pd.crosstab([lab["answer"], lab["q_num"] + 1], lab["label"],
                         margins=True).to_string())

    L += ["", "(b) Share labelled A (documentation of administration this admission)", ""]
    rows = []
    for ans in ("miss", "hit"):
        for q in list(range(4)) + ["all"]:
            s = lab[lab["answer"] == ans]
            s = s if q == "all" else s[s["q_num"] == q]
            s = s[s["label"] != "U"]
            p, lo, hi = wilson(int((s["label"] == "A").sum()), len(s))
            rows.append({"answer": ans, "quartile": q if q == "all" else q + 1,
                         "n_rated": len(s), "A": int((s["label"] == "A").sum()),
                         "O": int((s["label"] == "O").sum()),
                         "N": int((s["label"] == "N").sum()),
                         "share_A": p, "lo95": lo, "hi95": hi})
    L.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    m = lab[(lab["answer"] == "miss") & (lab["label"] != "U")].copy()
    m["notA"] = (m["label"] != "A").astype(int)
    L += ["", fit_or(m, "notA", "q_num", [], None,
                     "trend in NOT-A among reviewed misses (OR/quartile)"),
          "  (an OR above 1 means flag failures concentrate in deranged quartiles)"]

    L += ["", "(c) Rule-based context classes vs reader (U excluded; A vs O/N)", ""]
    v = lab[lab["label"] != "U"]
    if len(v):
        isA = (v["label"] == "A").astype(int)
        st = v["strict"].astype(int)
        L.append(pd.crosstab(v["label"], v["strict"].map({True: "strict",
                                                           False: "loose-only"})).to_string())
        po, k = kappa(isA, st)
        sens = float(st[isA == 1].mean()) if (isA == 1).any() else float("nan")
        spec = float((1 - st[isA == 0]).mean()) if (isA == 0).any() else float("nan")
        ktxt = (f"kappa {k:.3f}" if st.nunique() > 1 and isA.nunique() > 1
                else "kappa not estimable (one rating constant)")
        L += ["", f"agreement {po:.3f}  {ktxt}  strict among A {sens:.3f}  "
              f"loose-only among O/N {spec:.3f}"]

    L += ["", "(d) Gradient with documentation judged by the reader", ""]
    rng = np.random.default_rng(args.seed)
    subj = notes["subject_id"].to_numpy()
    s_codes, s_idx = np.unique(subj, return_inverse=True)
    q_arr = notes["q_num"].to_numpy().astype(int)
    m_arr = notes["miss"].to_numpy().astype(int)
    Mmat = np.zeros((len(s_codes), 4))
    Hmat = np.zeros((len(s_codes), 4))
    np.add.at(Mmat, (s_idx, q_arr), m_arr)
    np.add.at(Hmat, (s_idx, q_arr), 1 - m_arr)
    raw = (counts["misses"] / (counts["misses"] + counts["hits"])).round(3).tolist()
    for u_as, name in ((None, "U excluded (main)"), ("A", "U counted as A"),
                       ("N", "U counted as not A")):
        k = lab.copy()
        if u_as:
            k.loc[k["label"] == "U", "label"] = u_as
        k = k[k["label"].isin(["A", "O", "N"])]
        cell = {}
        for q in range(4):
            for mf in (0, 1):
                c = k[(k["q_num"] == q) & (k["miss"] == mf)]
                cell[(q, mf)] = (int((c["label"] == "A").sum()), int((c["label"] != "A").sum()))
        tab = corrected(lab, counts, u_as)
        point = trend_or(tab)
        boots = []
        for _ in range(args.boot):
            w = np.bincount(rng.integers(0, len(s_codes), len(s_codes)),
                            minlength=len(s_codes))
            mq, hq = w @ Mmat, w @ Hmat
            rows = []
            for q in range(4):
                a1, b1 = cell[(q, 1)]
                a0, b0 = cell[(q, 0)]
                t = rng.beta(a1 + 0.5, b1 + 0.5) if a1 + b1 else float("nan")
                h = rng.beta(a0 + 0.5, b0 + 0.5) if a0 + b0 else float("nan")
                ev = mq[q] * t
                den = hq[q] * h + ev
                rows.append({"q_num": q, "corr_denom": den,
                             "corr_rate": ev / den if den > 0 else float("nan")})
            boots.append(trend_or(pd.DataFrame(rows)))
        b = np.array([x for x in boots if not math.isnan(x)])
        lo, hi = ((np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b)
                  else (float("nan"), float("nan")))
        L += [f"--- {name} ---",
              tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
              f"raw miss rate by quartile (regex-documented): {raw}",
              f"adjudicated miss-rate trend: OR per quartile {point:.3f} "
              f"(bootstrap 95% CI {lo:.3f}-{hi:.3f}, {len(b)} resamples)"]
        gone = (tab.loc[tab["corr_rate"].isna(), "q_num"] + 1).tolist()
        if gone:
            L.append(f"!! quartile(s) {gone} have no rated misses or hits and drop out")
        L.append("")

    L += ["(e) Reviewed misses: share A by term category (U excluded)", ""]
    e = lab[(lab["answer"] == "miss") & (lab["label"] != "U")]
    if len(e):
        g = e.groupby("category")["label"].agg(n="size", A=lambda s: int((s == "A").sum()))
        g["share_A"] = g["A"] / g["n"]
        L.append(g.reindex([c for c in CATEGORIES if c in g.index]).to_string(
            float_format=lambda x: f"{x:.3f}"))

    L += ["", "Reading: share_A among misses is the share of the delegate's misses whose",
          "excerpts document administration during this admission on blinded reading.",
          "Label A includes notes that document vasopressors only as pressors or",
          "vasopressors, where a literal reading of the question might justify 'no';",
          "(e) separates them. A share that falls with quartile would mean the raw",
          "gradient was inflated by the documentation flag; the adjudicated trend in (d)",
          "is the corrected estimate. The review measures the flag's precision, not",
          "what the flag misses."]
    (out / "adjudicated_summary.txt").write_text("\n".join(L) + "\n")
    for l in L:
        log(l)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p8-dir", type=Path, default=DEF_P8)
    ap.add_argument("--mimic-note", type=Path, default=DEF_MIMIC_NOTE)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--review-misses", type=int, default=50,
                    help="misses sampled per quartile for review")
    ap.add_argument("--review-hits", type=int, default=10,
                    help="hits sampled per quartile for review")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labelled review sheet (CSV) -> adjudication mode")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    args.p8_dir = args.p8_dir.expanduser()
    args.mimic_note = args.mimic_note.expanduser()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.labels:
        args.labels = args.labels.expanduser()
        mode_adjudicate(args)
    else:
        mode_diagnose(args)


if __name__ == "__main__":
    main()
