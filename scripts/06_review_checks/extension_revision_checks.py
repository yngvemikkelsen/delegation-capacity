# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
# ]
# ///
"""
extension_revision_checks.py -- checks requested in review of Paper 24
(Frontiers, Reviewer 2, comments 3 and 4). No LLM calls; no note text is
read or written. All outputs are aggregates.

PART A  Review counts in the P4 experiment (Reviewer 2, comment 4)
    Replays the random draws of extension_p4_label_provenance.py (same seed,
    same order of draws, same data order) without retraining, and counts, per
    fact, review scenario and quartile, the delegate errors in the training
    labels, how many the simulated review corrected and how many remained.
    The replay is checked against p4_contamination.csv from the P4 run.
    Also reports the deterministic counts in the whole cohort: delegate errors
    by fact and quartile, and the number expected to be corrected (errors x q).

PART B  Reference records (Reviewer 2, comment 3)
  B1  Was the recorded treatment given? Reference events are re-scanned with
      their status and amount. An event counts as given if its status is not
      "Rewritten" and its amount (inputevents) or value (procedureevents) is
      above 0. The P8 primary model, S1 and S5 are refitted with this stricter
      reference.
  B2  Treatment outside the ICU stay. For admissions where the delegate said
      "yes" but the ICU record has no event (disagreements that treatment
      outside the ICU could explain), the hospital-wide tables are searched:
        hosp/emar               vasopressor (and any red-cell product)
                                administrations charted outside the ICU stay
        hosp/procedures_icd     dialysis, red-cell transfusion and vasopressor
                                procedure codes dated outside the ICU stay
                                (dates only; codes dated within the ICU stay
                                dates cannot be placed and are counted apart)
      Delegate error against the reference is recomputed with these "yes"
      answers counted as correct.
      Limits: eMAR does not cover every admission (coverage is reported);
      emergency department and operating room records are not in these
      tables, so the counts are lower bounds.

INPUTS (read only)
    --p8-dir      ~/extension_results/p8      p8_per_note_fact.csv
    --p4-dir      ~/extension_results/p4      p4_contamination.csv (check only)
    --mimic-root  MIMIC-IV 3.1 root           icu/ and hosp/ tables

OUTPUTS (in --out, default ~/extension_results/revision_checks)
    checks_summary.txt          all results
    a_review_counts.csv         Part A, mean over repetitions
    a_cohort_errors.csv         Part A, whole cohort
    b1_status_counts.csv        statusdescription of reference events
    b1_models.csv               P8 models with the 'given' reference
    b2_disagreements.csv        Part B2 by fact
    b2_codes_matched.csv        procedure codes and eMAR medication strings used

USAGE
    uv run --offline extension_revision_checks.py
    Run time: about 15-30 min (most of it reading hosp/emar and icu/inputevents).
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
DEF_P4 = HOME / "extension_results" / "p4"
DEF_MIMIC_ROOT = HOME / "physionet.org" / "files" / "mimiciv" / "3.1"
DEF_OUT = HOME / "extension_results" / "revision_checks"
FACT_ORDER = ["vasopressor", "renal_replacement", "rbc_transfusion"]

# reference definitions, identical to extension_p8_error_demand_v2.py
FACTS = {
    "vasopressor": {"table": "inputevents",
                    "label_re": r"\b(?:norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine)\b",
                    "label_exclude": r"nebuliz|racemic|inhal"},
    "renal_replacement": {"table": "procedureevents",
                          "label_re": r"(?:dialysis|crrt|cvvh|scuf)",
                          "label_exclude": r"catheter"},
    "rbc_transfusion": {"table": "inputevents",
                        "label_re": r"packed red blood", "label_exclude": r""},
}
AMOUNT_COL = {"inputevents": "amount", "procedureevents": "value"}

# P4 settings (defaults of extension_p4_label_provenance.py)
P4_SEED, P4_REPS, P4_TEST_FRAC = 42, 50, 0.3
P4_SCEN = {"constant": np.full(4, 0.5), "load": np.array([0.8, 0.6, 0.4, 0.2])}

# hospital-wide searches
EMAR_MED = {
    "vasopressor": re.compile(r"norepinephrine|levophed|epinephrine|phenylephrine|"
                              r"neo-?synephrine|vasopressin|dopamine", re.I),
    "rbc_transfusion": re.compile(r"red blood|packed (?:red )?cell|\bprbc|\brbc\b", re.I),
}
EMAR_MED_EXCLUDE = re.compile(
    r"lidocaine|bupivacaine|ropivacaine|articaine|prilocaine|ophth|eye|nasal|spray|"
    r"topical|epipen|auto-?inject|racepinephrine|inhal|neb|carbidopa|levodopa|"
    r"ropinirole|pramipexole|agonist|dobutamine", re.I)
EMAR_ADMIN = re.compile(r"administ|start|rate change|bolus|new bag", re.I)
EMAR_NOT_GIVEN = re.compile(r"not given|hold|held|delay|refus|cancel|missed|"
                            r"\bnot\b|stop", re.I)
ICD_TITLE = {
    "renal_replacement": (re.compile(r"dialysis|hemofiltration|hemodiafiltration|"
                                     r"ultrafiltration|urinary filtration", re.I),
                          re.compile(r"shunt|fistula|graft|catheter|cannula|insertion|"
                                     r"removal|revision|creation|anastomosis|bypass|"
                                     r"access|declot|thrombectomy|examination", re.I)),
    "rbc_transfusion": (re.compile(r"transfusion.*(?:red blood|packed cell|"
                                   r"erythrocyte|red cell)", re.I),
                        re.compile(r"$^")),
    "vasopressor": (re.compile(r"vasopressor", re.I), re.compile(r"$^")),
}


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


def header(root: Path, name: str, need: list[str]) -> Path:
    path = find_table(root, name)
    cols = list(pd.read_csv(path, nrows=0).columns)
    miss = [c for c in need if c not in cols]
    if miss:
        sys.exit(f"[fatal] {name} lacks {miss}; present: {cols}")
    log(f"    {name:<18} <- {path}")
    return path


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, min(p, c - h)), min(1.0, max(p, c + h))


def fit_logit(df: pd.DataFrame, y: str, fact_fe: bool, label: str) -> dict:
    """As extension_p8_error_demand_v2.py: logistic GLM, cluster-robust by subject."""
    if len(df) < 30 or df[y].nunique() < 2:
        return {"model": label, "n": len(df), "status": "skipped"}
    X = df[["q_num"]].astype(float).copy()
    if fact_fe and df["fact"].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(df["fact"], prefix="fact", drop_first=True,
                                         dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["subject_id"].values})
    b = float(res.params["q_num"])
    lo, hi = (float(v) for v in res.conf_int().loc["q_num"].values)
    return {"model": label, "n": len(df), "events": int(df[y].sum()),
            "OR": math.exp(b), "OR_lo": math.exp(lo), "OR_hi": math.exp(hi),
            "p": float(res.pvalues["q_num"]), "status": "ok"}


def fmt_model(r: dict) -> str:
    if r.get("status") != "ok":
        return f"{r['model']:<52} {r.get('status')}"
    return (f"{r['model']:<52} OR {r['OR']:.3f} ({r['OR_lo']:.3f}-{r['OR_hi']:.3f}) "
            f"p={r['p']:.3g} n={r['n']} events={r['events']}")


def load_per(p8_dir: Path) -> pd.DataFrame:
    path = p8_dir / "p8_per_note_fact.csv"
    if not path.exists():
        sys.exit(f"[fatal] {path} not found")
    per = pd.read_csv(path, dtype={"note_id": str}, low_memory=False)
    need = {"note_id", "subject_id", "hadm_id", "stay_id", "q_num", "fact",
            "structured_present", "text_mentioned", "llm_answer", "possibly_truncated"}
    miss = need - set(per.columns)
    if miss:
        sys.exit(f"[fatal] {path} lacks columns {sorted(miss)}")
    for c in ("structured_present", "text_mentioned", "possibly_truncated"):
        per[c] = per[c].astype(str).str.lower() == "true"
    per["llm_answer"] = per["llm_answer"].astype(str).str.lower()
    log(f"    p8_per_note_fact   <- {path}  ({per.note_id.nunique()} notes)")
    return per


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------

def p4_frame(per_raw: pd.DataFrame) -> pd.DataFrame:
    """Replicates load_p8() of extension_p4_label_provenance.py, row order included."""
    per = per_raw.copy()
    per["valid"] = per["llm_answer"].isin(["yes", "no"]) & ~per["possibly_truncated"]
    ok = per.groupby("note_id")["valid"].all()
    per = per[per["note_id"].isin(set(ok[ok].index))]
    wide = per.pivot_table(index=["note_id", "subject_id", "hadm_id", "q_num"],
                           columns="fact", values=["structured_present", "llm_answer"],
                           aggfunc="first")
    out = pd.DataFrame(index=wide.index).reset_index()
    for f in FACT_ORDER:
        out[f"ref_{f}"] = wide[("structured_present", f)].astype(bool).values.astype(int)
        out[f"llm_{f}"] = (wide[("llm_answer", f)].values == "yes").astype(int)
    out["q_num"] = out["q_num"].astype(int)
    return out


def part_a(per: pd.DataFrame, p4_dir: Path, out: Path, L: list[str]) -> None:
    L += ["=" * 78, "PART A  Simulated review in the P4 experiment (Reviewer 2, comment 4)",
          "=" * 78]
    df = p4_frame(per)
    qn = df["q_num"].to_numpy()

    # whole cohort, deterministic
    rows = []
    for f in FACT_ORDER:
        err = (df[f"llm_{f}"] != df[f"ref_{f}"]).to_numpy()
        for q in range(4):
            m = qn == q
            e = int(err[m].sum())
            r = {"fact": f, "quartile": q + 1, "notes": int(m.sum()), "errors": e}
            for s, qv in P4_SCEN.items():
                r[f"expected_corrected_{s}"] = e * qv[q]
                r[f"expected_remaining_{s}"] = e * (1 - qv[q])
            rows.append(r)
    coh = pd.DataFrame(rows)
    coh.to_csv(out / "a_cohort_errors.csv", index=False)
    L.append("\nA1  Whole cohort: delegate errors against the ICU reference, and the "
             "number expected to be corrected (errors x q) and to remain")
    for f in FACT_ORDER:
        g = coh[coh["fact"] == f]
        L.append(f"    {f:<18} errors by quartile {g['errors'].tolist()}  total "
                 f"{int(g['errors'].sum())}")
        for s in P4_SCEN:
            L.append(f"        {s:<8} corrected {g[f'expected_corrected_{s}'].sum():7.1f}  "
                     f"remaining {g[f'expected_remaining_{s}'].sum():7.1f}  | by quartile "
                     f"corrected {[round(v, 1) for v in g[f'expected_corrected_{s}']]} "
                     f"remaining {[round(v, 1) for v in g[f'expected_remaining_{s}']]}")

    # replay of the draws
    rng = np.random.default_rng(P4_SEED)
    subjects = df["subject_id"].unique()
    recs = []
    for rep in range(P4_REPS):
        test_subj = set(rng.choice(subjects, size=int(round(len(subjects) * P4_TEST_FRAC)),
                                   replace=False))
        te = df["subject_id"].isin(test_subj).to_numpy()
        tr = ~te
        u_over = rng.random(tr.sum())
        _u_prov = rng.random(tr.sum())  # drawn to keep the sequence; omega = 1 here
        for f in FACT_ORDER:
            ref_tr = df[f"ref_{f}"].to_numpy()[tr]
            llm_tr = df[f"llm_{f}"].to_numpy()[tr]
            ref_te = df[f"ref_{f}"].to_numpy()[te]
            if ref_tr.sum() < 10 or ref_te.sum() < 5:
                continue
            err = llm_tr != ref_tr
            for s, qv in P4_SCEN.items():
                corr = err & (u_over < qv[qn[tr]])
                rem = err & ~corr
                for q in range(4):
                    m = qn[tr] == q
                    recs.append({"rep": rep, "fact": f, "scenario": s, "quartile": q + 1,
                                 "n_train": int(m.sum()), "errors": int(err[m].sum()),
                                 "corrected": int(corr[m].sum()),
                                 "remaining": int(rem[m].sum()),
                                 "contamination": float(rem[m].mean()) if m.any()
                                 else float("nan")})
    rp = pd.DataFrame(recs)

    # check against the P4 run
    cpath = p4_dir / "p4_contamination.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        c = c[np.isclose(c["omega"], 1.0)]
        mrg = rp.merge(c, on=["rep", "fact", "scenario", "quartile"],
                       suffixes=("", "_p4"), how="inner")
        dmax = float((mrg["contamination"] - mrg["contamination_p4"]).abs().max())
        L.append(f"\nA2  Replay check against {cpath.name}: {len(mrg)} of {len(rp)} cells "
                 f"matched; largest difference in contamination {dmax:.2e} "
                 f"({'REPRODUCED' if len(mrg) == len(rp) and dmax < 1e-9 else 'NOT REPRODUCED -- do not use Part A'})")
    else:
        L.append(f"\nA2  [warning] {cpath} not found; replay not checked")

    mean = (rp.groupby(["fact", "scenario", "quartile"])
            [["n_train", "errors", "corrected", "remaining", "contamination"]]
            .mean().reset_index())
    tot = (rp.groupby(["rep", "fact", "scenario"])[["n_train", "errors", "corrected",
                                                    "remaining"]].sum()
           .groupby(["fact", "scenario"]).mean().reset_index())
    mean.to_csv(out / "a_review_counts.csv", index=False)
    L.append("\nA3  Training labels at omega = 1, mean over 50 repetitions "
             "(errors in the delegate's answers; corrected by simulated review; remaining)")
    for f in FACT_ORDER:
        for s in P4_SCEN:
            t = tot[(tot["fact"] == f) & (tot["scenario"] == s)]
            g = mean[(mean["fact"] == f) & (mean["scenario"] == s)]
            if t.empty:
                continue
            t = t.iloc[0]
            L.append(f"    {f:<18} {s:<8} n_train {t['n_train']:.0f}  errors "
                     f"{t['errors']:.1f}  corrected {t['corrected']:.1f}  remaining "
                     f"{t['remaining']:.1f} ({100 * t['remaining'] / t['n_train']:.1f}% "
                     f"of labels)")
            L.append("        by quartile  errors " + ", ".join(f"{v:.1f}" for v in g["errors"])
                     + " | corrected " + ", ".join(f"{v:.1f}" for v in g["corrected"])
                     + " | remaining " + ", ".join(f"{v:.1f}" for v in g["remaining"])
                     + " | contamination % " + ", ".join(f"{100 * v:.1f}"
                                                        for v in g["contamination"]))
    L.append("\nA4  The 0.004 reported for vasopressor is a difference in differences: "
             "[Brier loss Q4 - Q1 under load-dependent review] minus [Brier loss Q4 - Q1 "
             "under constant review], at omega = 1, mean over repetitions "
             "(extension_p4_label_provenance.py, summarise()).")


# ---------------------------------------------------------------------------
# PART B
# ---------------------------------------------------------------------------

def resolve_itemids(root: Path) -> dict[str, list[int]]:
    path = header(root, "d_items", ["itemid", "label", "linksto"])
    d = pd.read_csv(path, usecols=["itemid", "label", "linksto"])
    d["label"] = d["label"].fillna("").astype(str)
    d["linksto"] = d["linksto"].astype(str).str.lower()
    res = {}
    for f, s in FACTS.items():
        m = d["label"].str.contains(s["label_re"], case=False, regex=True, na=False)
        if s["label_exclude"]:
            m &= ~d["label"].str.contains(s["label_exclude"], case=False, regex=True,
                                          na=False)
        m &= d["linksto"] == s["table"]
        res[f] = sorted(d.loc[m, "itemid"].astype(int).tolist())
        log(f"    {f:<18} {len(res[f])} itemids {res[f]}")
    return res


def part_b1(root: Path, per: pd.DataFrame, out: Path, L: list[str]) -> pd.DataFrame:
    L += ["", "=" * 78, "PART B1  Was the recorded treatment given? (Reviewer 2, comment 3)",
          "=" * 78]
    itemids = resolve_itemids(root)
    hadms = set(per["hadm_id"].astype("int64"))
    any_ev = {f: set() for f in FACT_ORDER}
    given = {f: set() for f in FACT_ORDER}
    status_rows = []
    for table in ("inputevents", "procedureevents"):
        amt = AMOUNT_COL[table]
        path = header(root, table, ["hadm_id", "itemid", amt, "statusdescription"])
        ids_for = {f: set(itemids[f]) for f, s in FACTS.items() if s["table"] == table}
        all_ids = set().union(*ids_for.values())
        for ch in pd.read_csv(path, usecols=["hadm_id", "itemid", amt, "statusdescription"],
                              chunksize=2_000_000, low_memory=False):
            ch = ch.dropna(subset=["hadm_id"])
            ch = ch[ch["itemid"].isin(all_ids)]
            if ch.empty:
                continue
            ch = ch.assign(hadm_id=ch["hadm_id"].astype("int64"))
            ch = ch[ch["hadm_id"].isin(hadms)]
            st = ch["statusdescription"].fillna("(missing)").astype(str)
            ok = (~st.str.lower().eq("rewritten")) & (pd.to_numeric(ch[amt], errors="coerce") > 0)
            for f, ids in ids_for.items():
                m = ch["itemid"].isin(ids)
                any_ev[f].update(ch.loc[m, "hadm_id"].unique())
                given[f].update(ch.loc[m & ok, "hadm_id"].unique())
                vc = st[m].value_counts()
                status_rows += [{"fact": f, "statusdescription": k, "events": int(v)}
                                for k, v in vc.items()]
                status_rows += [{"fact": f, "statusdescription": "(amount <= 0 or missing)",
                                 "events": int((m & ~(pd.to_numeric(ch[amt], errors='coerce') > 0)).sum())}]
    sc = (pd.DataFrame(status_rows).groupby(["fact", "statusdescription"])["events"]
          .sum().reset_index())
    sc.to_csv(out / "b1_status_counts.csv", index=False)
    L.append("\nB1a Reference events by status (cohort admissions)")
    for f in FACT_ORDER:
        g = sc[sc["fact"] == f].sort_values("events", ascending=False)
        L.append(f"    {f:<18} " + "; ".join(f"{r.statusdescription}: {r.events}"
                                             for r in g.itertuples()))

    per = per.copy()
    per["given"] = [h in given[f] for h, f in zip(per["hadm_id"].astype("int64"), per["fact"])]
    per["any_event_rescan"] = [h in any_ev[f] for h, f in
                               zip(per["hadm_id"].astype("int64"), per["fact"])]
    agree = (per["any_event_rescan"] == per["structured_present"]).mean()
    L.append(f"\nB1b Rescan agrees with P8 structured_present in {100 * agree:.3f}% of "
             f"note-fact pairs")
    for f in FACT_ORDER:
        g = per[per["fact"] == f]
        lost = g[g["structured_present"] & ~g["given"]]
        L.append(f"    {f:<18} admissions in reference {int(g['structured_present'].sum())}; "
                 f"with no event meeting 'given' {len(lost)}")

    per["valid"] = per["llm_answer"].isin(["yes", "no"]) & ~per["possibly_truncated"]
    per["miss"] = (per["llm_answer"] == "no").astype(int)
    ok = per[per["valid"]]
    p_old = ok[ok["structured_present"] & ok["text_mentioned"]]
    p_new = ok[ok["given"] & ok["text_mentioned"]]
    dropped = p_old[~p_old["given"]]
    L.append(f"\nB1c Primary set: {len(p_old)} pairs ({int(p_old['miss'].sum())} misses) -> "
             f"{len(p_new)} pairs ({int(p_new['miss'].sum())} misses) with the 'given' "
             f"reference; pairs removed {len(dropped)}, of which misses "
             f"{int(dropped['miss'].sum())}")
    for q in range(4):
        a = p_new[p_new["q_num"] == q]
        r, lo, hi = wilson(int(a["miss"].sum()), len(a))
        L.append(f"    quartile {q + 1}: {int(a['miss'].sum())}/{len(a)} "
                 f"{100 * r:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})")
    models = [fit_logit(p_old, "miss", True, "PRIMARY, P8 reference (check 1.136)"),
              fit_logit(p_new, "miss", True, "PRIMARY, 'given' reference"),
              fit_logit(ok[ok["given"]], "miss", True, "S1, 'given' reference")]
    for f in FACT_ORDER:
        models.append(fit_logit(p_new[p_new["fact"] == f], "miss", False,
                                f"S5 {f}, 'given' reference"))
    pd.DataFrame(models).to_csv(out / "b1_models.csv", index=False)
    L += ["", "B1d Models (OR per quartile step, cluster-robust by subject)"]
    L += ["    " + fmt_model(r) for r in models]
    return per


def part_b2(root: Path, per: pd.DataFrame, out: Path, L: list[str]) -> None:
    L += ["", "=" * 78,
          "PART B2  Treatment outside the ICU stay (Reviewer 2, comment 3)", "=" * 78]
    hadms = set(per["hadm_id"].astype("int64"))
    stays = set(per["stay_id"].astype("int64"))
    path = header(root, "icustays", ["hadm_id", "stay_id", "intime", "outtime"])
    icu = pd.read_csv(path, usecols=["hadm_id", "stay_id", "intime", "outtime"])
    icu = icu[icu["stay_id"].isin(stays)].copy()
    icu["hadm_id"] = icu["hadm_id"].astype("int64")
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    if icu["hadm_id"].duplicated().any():
        sys.exit("[fatal] more than one ICU stay per cohort admission")
    win = icu.set_index("hadm_id")[["intime", "outtime"]]

    evid = {f: {"emar_out": set(), "icd_out": set(), "icd_icu_dates": set()}
            for f in FACT_ORDER}
    code_rows = []

    # eMAR
    epath = header(root, "emar", ["hadm_id", "charttime", "medication", "event_txt"])
    emar_cov: set[int] = set()
    med_counts: dict[tuple[str, str], int] = {}
    evt_counts: dict[tuple[str, str], int] = {}
    for ch in pd.read_csv(epath, usecols=["hadm_id", "charttime", "medication", "event_txt"],
                          chunksize=2_000_000, low_memory=False):
        ch = ch.dropna(subset=["hadm_id"])
        ch = ch.assign(hadm_id=ch["hadm_id"].astype("int64"))
        ch = ch[ch["hadm_id"].isin(hadms)]
        if ch.empty:
            continue
        emar_cov.update(ch["hadm_id"].unique())
        med = ch["medication"].fillna("").astype(str)
        for f, rx in EMAR_MED.items():
            m = med.str.contains(rx) & ~med.str.contains(EMAR_MED_EXCLUDE)
            if not m.any():
                continue
            s = ch[m].copy()
            for k, v in s["medication"].value_counts().items():
                med_counts[(f, k)] = med_counts.get((f, k), 0) + int(v)
            evt = s["event_txt"].fillna("(missing)").astype(str)
            for k, v in evt.value_counts().items():
                evt_counts[(f, k)] = evt_counts.get((f, k), 0) + int(v)
            adm = evt.str.contains(EMAR_ADMIN) & ~evt.str.contains(EMAR_NOT_GIVEN)
            s = s[adm.values]
            if s.empty:
                continue
            s["charttime"] = pd.to_datetime(s["charttime"])
            s = s.join(win, on="hadm_id")
            outside = (s["charttime"] < s["intime"]) | (s["charttime"] > s["outtime"])
            evid[f]["emar_out"].update(s.loc[outside, "hadm_id"].unique())
    for (f, k), v in sorted(med_counts.items(), key=lambda x: -x[1]):
        code_rows.append({"source": "emar medication", "fact": f, "code_or_text": k,
                          "rows_in_cohort": v})
    for (f, k), v in sorted(evt_counts.items(), key=lambda x: -x[1]):
        code_rows.append({"source": "emar event_txt", "fact": f, "code_or_text": k,
                          "rows_in_cohort": v,
                          "counted_as_given": bool(EMAR_ADMIN.search(k)
                                                   and not EMAR_NOT_GIVEN.search(k))})

    # procedures_icd
    dpath = header(root, "d_icd_procedures", ["icd_code", "icd_version", "long_title"])
    d = pd.read_csv(dpath, dtype={"icd_code": str})
    ppath = header(root, "procedures_icd", ["hadm_id", "chartdate", "icd_code", "icd_version"])
    pr = pd.read_csv(ppath, usecols=["hadm_id", "chartdate", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    pr = pr[pr["hadm_id"].isin(hadms)].copy()
    pr["hadm_id"] = pr["hadm_id"].astype("int64")
    pr = pr.merge(d, on=["icd_code", "icd_version"], how="left")
    pr["long_title"] = pr["long_title"].fillna("").astype(str)
    pr["chartdate"] = pd.to_datetime(pr["chartdate"]).dt.normalize()
    pr = pr.join(win, on="hadm_id")
    pr["out"] = ((pr["chartdate"] < pr["intime"].dt.normalize())
                 | (pr["chartdate"] > pr["outtime"].dt.normalize()))
    for f, (inc, exc) in ICD_TITLE.items():
        m = pr["long_title"].str.contains(inc) & ~pr["long_title"].str.contains(exc)
        s = pr[m]
        for (code, ver, title), v in s.groupby(["icd_code", "icd_version", "long_title"]).size().items():
            code_rows.append({"source": f"procedures_icd v{ver}", "fact": f,
                              "code_or_text": f"{code} {title}", "rows_in_cohort": int(v)})
        evid[f]["icd_out"].update(s.loc[s["out"], "hadm_id"].unique())
        evid[f]["icd_icu_dates"].update(s.loc[~s["out"], "hadm_id"].unique())
    pd.DataFrame(code_rows).to_csv(out / "b2_codes_matched.csv", index=False)

    L.append("\nB2a Search terms matched in the cohort (full list in b2_codes_matched.csv)")
    cr = pd.DataFrame(code_rows)
    for f in FACT_ORDER:
        g = cr[cr["fact"] == f].sort_values("rows_in_cohort", ascending=False)
        for src in g["source"].unique():
            h = g[g["source"] == src].head(8)
            L.append(f"    {f:<18} {src}: " + "; ".join(
                f"{r.code_or_text} ({r.rows_in_cohort})" for r in h.itertuples()))
    L.append(f"    eMAR rows exist for {len(emar_cov)} of {len(hadms)} cohort admissions")

    per = per.copy()
    per["valid"] = per["llm_answer"].isin(["yes", "no"]) & ~per["possibly_truncated"]
    ok = per[per["valid"]].copy()
    ok["yes"] = (ok["llm_answer"] == "yes").astype(int)
    ok["ref"] = ok["structured_present"].astype(int)
    h = ok["hadm_id"].astype("int64")
    ok["emar_cov"] = h.isin(emar_cov)
    ok["ev_out"] = [hh in evid[f]["emar_out"] or hh in evid[f]["icd_out"]
                    for hh, f in zip(h, ok["fact"])]
    ok["ev_icu_dates_only"] = [(hh in evid[f]["icd_icu_dates"]) and not ev
                               for hh, f, ev in zip(h, ok["fact"], ok["ev_out"])]
    rows = []
    L.append("\nB2b Disagreements: delegate 'yes', no ICU event")
    for f in FACT_ORDER:
        g = ok[ok["fact"] == f]
        fp = g[(g["yes"] == 1) & (g["ref"] == 0)]
        neg = g[g["ref"] == 0]
        rec = {"fact": f, "notes": len(g), "icu_negative": len(neg),
               "icu_negative_with_outside_evidence": int(neg["ev_out"].sum()),
               "false_yes": len(fp),
               "false_yes_documented": int(fp["text_mentioned"].sum()),
               "false_yes_outside_evidence": int(fp["ev_out"].sum()),
               "false_yes_outside_evidence_documented":
                   int((fp["ev_out"] & fp["text_mentioned"]).sum()),
               "false_yes_code_on_icu_dates_only": int(fp["ev_icu_dates_only"].sum()),
               "false_yes_with_emar_coverage": int(fp["emar_cov"].sum()),
               "misses_icu_positive": int(((g["yes"] == 0) & (g["ref"] == 1)).sum())}
        g = g.assign(err=(g["yes"] != g["ref"]).astype(int))
        g = g.assign(err_adj=((g["err"] == 1) & ~((g["yes"] == 1) & (g["ref"] == 0)
                                                  & g["ev_out"])).astype(int))
        rec["error_rate"] = g["err"].mean()
        rec["error_rate_adjusted"] = g["err_adj"].mean()
        for q in range(4):
            rec[f"error_q{q + 1}"] = g.loc[g["q_num"] == q, "err"].mean()
            rec[f"error_adj_q{q + 1}"] = g.loc[g["q_num"] == q, "err_adj"].mean()
        rows.append(rec)
        L.append(f"    {f:<18} ICU-negative admissions {rec['icu_negative']}, of which with "
                 f"treatment recorded outside the ICU stay {rec['icu_negative_with_outside_evidence']}")
        L.append(f"        delegate 'yes' without ICU event: {rec['false_yes']} "
                 f"(documented in note {rec['false_yes_documented']}); with treatment "
                 f"recorded outside the ICU stay {rec['false_yes_outside_evidence']} "
                 f"({100 * rec['false_yes_outside_evidence'] / max(1, rec['false_yes']):.1f}%); "
                 f"procedure code on ICU dates only {rec['false_yes_code_on_icu_dates_only']}; "
                 f"admissions with any eMAR rows {rec['false_yes_with_emar_coverage']}")
        L.append(f"        delegate 'no' with ICU event (cannot be explained by treatment "
                 f"outside the ICU): {rec['misses_icu_positive']}")
        L.append(f"        error vs ICU reference {100 * rec['error_rate']:.1f}% (by quartile "
                 + ", ".join(f"{100 * rec[f'error_q{q}']:.1f}" for q in range(1, 5))
                 + f"); counting supported 'yes' as correct {100 * rec['error_rate_adjusted']:.1f}% "
                 "(" + ", ".join(f"{100 * rec[f'error_adj_q{q}']:.1f}" for q in range(1, 5)) + ")")
    pd.DataFrame(rows).to_csv(out / "b2_disagreements.csv", index=False)
    L.append("    Lower bounds: eMAR does not cover every admission; emergency department "
             "and operating room records are not searched; procedure codes carry dates only.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p8-dir", type=Path, default=DEF_P8)
    ap.add_argument("--p4-dir", type=Path, default=DEF_P4)
    ap.add_argument("--mimic-root", type=Path, default=DEF_MIMIC_ROOT)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--skip-b", action="store_true", help="Part A only")
    args = ap.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    root = args.mimic_root.expanduser()
    log("loading inputs")
    per = load_per(args.p8_dir.expanduser())
    L: list[str] = ["extension_revision_checks.py -- aggregates only", ""]
    part_a(per, args.p4_dir.expanduser(), out, L)
    if not args.skip_b:
        log("\nPart B: reading MIMIC-IV tables (about 15-30 min)")
        part_b1(root, per, out, L)
        part_b2(root, per, out, L)
    (out / "checks_summary.txt").write_text("\n".join(L) + "\n")
    log("")
    for line in L:
        log(line)
    log(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
