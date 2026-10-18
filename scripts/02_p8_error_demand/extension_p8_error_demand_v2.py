# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
#   "matplotlib>=3.8",
# ]
# ///
"""
extension_p8_error_demand_v2.py -- test of P8 (error-demand coupling).

v2 (after the pilot): the RRT reference excludes d_items labels containing
"catheter" (224270 Dialysis Catheter is line insertion, not therapy), and the
RRT question now includes peritoneal dialysis (225805 is in the reference).
Answers cached under v1 used a different question and must not be reused.

PROPOSITION (specification P8)
    Within a specified task, the delegate's initial error probability p is not
    independent of case mix x; it rises with case complexity, a component of
    D0. A flat or falling p across complexity strata, at matched task and
    reference standard, would contradict it.

DELEGATE AND TASK
    A local LLM (default qwen3:30b through Ollama, think=False, temperature 0,
    fixed seed, JSON-schema output) reads each discharge summary and answers
    three yes/no questions about the admission: intravenous vasopressor, renal
    replacement therapy, red-cell transfusion. Nothing leaves the machine.

REFERENCE STANDARD
    ICU event tables (icu/inputevents, icu/procedureevents). itemids are
    resolved by d_items label regex and echoed for inspection, with the same
    definitions as paper20_note_structured_discordance.py. None of the three
    facts shares an input with acuity_proxy_24h (HR, MAP, RR, SpO2, GCS, age,
    mechanical ventilation); invasive ventilation is left out for that reason.
    The script re-checks the acuity column names at run time.

STRATA (case complexity)
    Primary: acuity_quartile in paper18_phase1_acuity.csv (acute physiological
    derangement composite, first 24 ICU hours).
    Sensitivity: lab-only SOFA (coagulation, liver, renal; worst value in the
    first 24 ICU hours), computed here. RRT is excluded from that sensitivity
    because the renal component uses creatinine.

COHORT
    Same linkage rule as paper18x_acuity_build.py and the discordance script:
    single-ICU-stay admissions joined to the acuity file by stay_id; one
    discharge summary per admission (the longest if several).

ENDPOINTS
    Fixed in ANALYSIS_PLAN below and written to analysis_plan.txt before the
    first LLM call. Register the plan (e.g. on OSF) before the full run.

RUN ORDER
    1. Pilot -- timing, truncation and parsing on a stratified sample of notes:
           uv run extension_p8_error_demand_v2.py --pilot 40
    2. Full run (resumable; interrupted runs continue from llm_cache.jsonl):
           uv run extension_p8_error_demand_v2.py
    3. Re-analyse from the cache without calling the LLM:
           uv run extension_p8_error_demand_v2.py --analyse-only

    Ollama must be running with the model pulled (ollama pull qwen3:30b).

OUTPUTS (in --out)
    analysis_plan.txt          prespecified endpoints and decision rule
    manifest.json              data paths, Ollama version, model digest/quant
    itemids.txt                itemid resolution echo
    llm_cache.jsonl            one line per note: answers + token counts
    lab_sofa_24h.csv           lab-only SOFA per stay (cached)
    p8_per_note_fact.csv       one row per note x fact (input to the P4 script)
    p8_rates_by_stratum.csv    absolute rates with Wilson 95% CIs
    p8_models.txt              all fitted models
    p8_summary.txt             primary result against the decision rule
    p8_error_by_stratum.png    figure, dpi=140
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

HOME = Path.home()
DEF_MIMIC_ROOT = HOME / "physionet.org" / "files" / "mimiciv" / "3.1"
DEF_MIMIC_NOTE = HOME / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"
DEF_ACUITY = HOME / "projects" / "BCST" / "paper18_results" / "paper18_phase1_acuity.csv"
DEF_OUT = HOME / "extension_results" / "p8"
DEF_OLLAMA = "http://localhost:11434"
DEF_MODEL = "qwen3:30b"


def log(m: str = "") -> None:
    print(m, flush=True)


# ---------------------------------------------------------------------------
# table location / loading
# ---------------------------------------------------------------------------

def find_table(root: Path, name: str) -> Path:
    hits: list[Path] = []
    for ext in (".csv.gz", ".csv"):
        hits.extend(root.rglob(f"{name}{ext}"))
    if not hits:
        sys.exit(f"[fatal] table '{name}' not found under {root}")
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def read_table(root: Path, name: str, usecols=None, chunksize=None):
    path = find_table(root, name)
    log(f"    {name:<16} <- {path}")
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                       low_memory=False)


# ---------------------------------------------------------------------------
# facts: reference definitions (as in the discordance script) + LLM questions
# ---------------------------------------------------------------------------

FACTS: dict[str, dict] = {
    "vasopressor": {
        "table": "inputevents",
        "label_re": r"\b(?:norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine)\b",
        "label_exclude": r"nebuliz|racemic|inhal",
        "text_re": r"(?:norepinephrine|levophed|epinephrine|phenylephrine|"
                   r"neo-?synephrine|vasopressin|dopamine|pressors?|vasopressors?)",
        "question": "Did the patient receive any intravenous norepinephrine, "
                    "epinephrine, phenylephrine, vasopressin or dopamine?",
    },
    "renal_replacement": {
        "table": "procedureevents",
        "label_re": r"(?:dialysis|crrt|cvvh|scuf)",
        "label_exclude": r"catheter",
        "text_re": r"(?:dialysis|dialyz|h[ae]modialysis|crrt|cvvh|"
                   r"renal replacement|ultrafiltrat)",
        "question": "Did the patient receive renal replacement therapy "
                    "(haemodialysis, CRRT, CVVH, CVVHD, CVVHDF, SCUF or "
                    "peritoneal dialysis)?",
    },
    "rbc_transfusion": {
        "table": "inputevents",
        "label_re": r"packed red blood",
        "label_exclude": r"",
        "text_re": r"(?:transfus|prbcs?|packed red (?:blood )?cells?|units? of blood)",
        "question": "Did the patient receive a transfusion of red blood cells "
                    "(packed red blood cells)?",
    },
}
FACT_ORDER = list(FACTS)

NEGATION_RE = re.compile(
    r"\b(?:no|not|non|without|denies|denied|never|negative for|free of|"
    r"ruled out|r/o|declined|deferred)\b[^.;:!?]{0,30}$",
    re.IGNORECASE,
)

CIRCULARITY_WATCH = {
    "vasopressor": ("vaso", "pressor", "norepi", "dopamine"),
    "renal_replacement": ("dialysis", "crrt", "rrt", "renal"),
    "rbc_transfusion": ("transfus", "rbc", "hemoglobin", "haemoglobin"),
}

# lab-only SOFA: d_labitems label regex (fluid Blood), aggregation, thresholds
LAB_SOFA = {
    "platelets": {"label_re": r"^platelet count$", "agg": "min", "units": ("k/ul",)},
    "bilirubin": {"label_re": r"^bilirubin, total$", "agg": "max", "units": ("mg/dl",)},
    "creatinine": {"label_re": r"^creatinine$", "agg": "max", "units": ("mg/dl",)},
}

SYSTEM_PROMPT = ("You are a careful clinical chart abstractor. You answer "
                 "strictly from the document provided.")

ANALYSIS_PLAN = """\
P8 ERROR-DEMAND COUPLING -- PRESPECIFIED ANALYSIS PLAN
(written before the first LLM call; do not edit after the full run starts)

Unit            note x fact (three facts: vasopressor, renal_replacement,
                rbc_transfusion), one discharge summary per single-ICU-stay
                admission.
Delegate output LLM answer yes/no. Answers that fail to parse, and notes
                flagged as possibly truncated, are excluded from the primary
                analysis and counted by stratum.
Reference       fact present in the ICU event tables (structured_present).
Documented      fact mentioned in the summary (regex with a 40-character
                negation window; text_mentioned).
Strata          acuity_quartile coded 0-3 (q_num).

PRIMARY
  Set      structured_present AND text_mentioned (documented positives).
  Outcome  miss = LLM answered "no".
  Model    logistic GLM: miss ~ q_num + C(fact), cluster-robust SE by
           subject_id. Estimate: odds ratio per quartile step, 95% CI.
  Decision supported      OR > 1 and lower 95% bound > 1
           contradicted   OR <= 1 (flat or falling)
           inconclusive   OR > 1 and lower 95% bound <= 1

SECONDARY
  S1  all structured positives; outcome miss (includes documentation
      omission: the error as the clinician meets it). Same model.
  S2  structured-negative AND not mentioned; outcome LLM "yes"
      (unsupported positive). Same model.
  S3  primary set; model adds log(note words) -- whether the gradient runs
      through note length (a mediator, not a confounder).
  S4  primary set, vasopressor and rbc_transfusion only; lab-only SOFA
      (0-12, per point) in place of q_num.
  S5  primary set, one model per fact (q_num only).
  S6  primary set with excluded notes (parse failure, truncation) counted
      as misses.
  Absolute rates by stratum with Wilson 95% CIs for primary, S1 and S2.

Reference-window note: the reference covers the ICU stay; the question
covers the admission. The primary set (ICU-confirmed positives) is robust to
this; S2 is not (ward-only events are scored as unsupported positives).
"""


# ---------------------------------------------------------------------------
# cohort, reference standard, notes
# ---------------------------------------------------------------------------

def build_cohort(mimic_root: Path, acuity_path: Path) -> pd.DataFrame:
    log("\nbuilding cohort")
    if not acuity_path.exists():
        sys.exit(f"[fatal] acuity file not found: {acuity_path}")
    ac = pd.read_csv(acuity_path, low_memory=False)
    log(f"    acuity          <- {acuity_path}")
    log(f"    acuity columns: {list(ac.columns)}")
    for col in ("stay_id", "acuity_quartile"):
        if col not in ac.columns:
            sys.exit(f"[fatal] acuity file lacks '{col}'. "
                     f"Present: {list(ac.columns)}")

    lower = " ".join(c.lower() for c in ac.columns)
    log("\n--- circularity check against acuity columns ---")
    for fact, toks in CIRCULARITY_WATCH.items():
        bad = [t for t in toks if t in lower]
        if bad:
            sys.exit(f"[fatal] {fact}: acuity columns match {bad}. The fact may "
                     f"share an input with the exposure; resolve before running.")
        log(f"    ok {fact}: no overlapping acuity column")
    log("--- end circularity check ---")

    icu = read_table(mimic_root, "icustays",
                     usecols=["subject_id", "hadm_id", "stay_id", "intime"])
    n_stays = len(icu)
    per_hadm = icu.groupby("hadm_id")["stay_id"].nunique()
    single = set(per_hadm[per_hadm == 1].index)
    icu = icu[icu["hadm_id"].isin(single)].copy()
    log(f"    icustays: {n_stays} stays -> {len(icu)} in single-stay admissions")

    keep = ["stay_id", "acuity_quartile"] + (
        ["acuity_proxy_24h"] if "acuity_proxy_24h" in ac.columns else [])
    coh = icu.merge(ac[keep], on="stay_id", how="inner")
    coh = coh.dropna(subset=["acuity_quartile"])
    coh["hadm_id"] = coh["hadm_id"].astype("int64")
    coh["subject_id"] = coh["subject_id"].astype("int64")
    coh["stay_id"] = coh["stay_id"].astype("int64")
    coh["intime"] = pd.to_datetime(coh["intime"])
    qlev = sorted(coh["acuity_quartile"].unique())
    if len(qlev) != 4:
        sys.exit(f"[fatal] expected 4 acuity_quartile levels, found {qlev}")
    coh["q_num"] = coh["acuity_quartile"].map({q: i for i, q in enumerate(qlev)})
    log(f"    joined to acuity: {len(coh)} admissions, "
        f"{coh.subject_id.nunique()} patients; quartile coding "
        f"{ {str(q): i for i, q in enumerate(qlev)} }")
    return coh


def resolve_itemids(mimic_root: Path, out: Path) -> dict[str, list[int]]:
    log("\n--- itemid resolution (VERIFY THESE) ---")
    d_items = read_table(mimic_root, "d_items",
                         usecols=["itemid", "label", "linksto"])
    d_items["label"] = d_items["label"].fillna("").astype(str)
    d_items["linksto"] = d_items["linksto"].astype(str).str.lower()
    res: dict[str, list[int]] = {}
    lines = []
    for fact, spec in FACTS.items():
        m = d_items["label"].str.contains(spec["label_re"], case=False,
                                          regex=True, na=False)
        if spec["label_exclude"]:
            m &= ~d_items["label"].str.contains(spec["label_exclude"],
                                                case=False, regex=True, na=False)
        m &= d_items["linksto"] == spec["table"]
        hits = d_items.loc[m, ["itemid", "label"]].sort_values("label")
        res[fact] = hits["itemid"].astype(int).tolist()
        lines.append(f"{fact} via {spec['table']}: {len(hits)} item(s)")
        lines.extend(f"    {int(r.itemid):>7}  {r.label}"
                     for r in hits.itertuples())
        if hits.empty:
            sys.exit(f"[fatal] no itemids matched for {fact}")
    (out / "itemids.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        log("  " + l)
    log("--- end itemid resolution ---")
    return res


def structured_presence(mimic_root: Path, itemids: dict[str, list[int]],
                        hadm_ids: set[int]) -> pd.DataFrame:
    log("\nscanning ICU event tables (reference standard)")
    seen = {f: set() for f in FACTS}
    for table in sorted({s["table"] for s in FACTS.values()}):
        ids_for = {f: set(itemids[f]) for f, s in FACTS.items()
                   if s["table"] == table}
        all_ids = set().union(*ids_for.values())
        for chunk in read_table(mimic_root, table,
                                usecols=["hadm_id", "itemid"],
                                chunksize=2_000_000):
            chunk = chunk.dropna(subset=["hadm_id"])
            chunk = chunk[chunk["itemid"].isin(all_ids)]
            if chunk.empty:
                continue
            chunk = chunk.assign(hadm_id=chunk["hadm_id"].astype("int64"))
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
            for fact, ids in ids_for.items():
                seen[fact].update(
                    chunk.loc[chunk["itemid"].isin(ids), "hadm_id"].unique())
    rows = []
    for fact in FACT_ORDER:
        log(f"    {fact:<20} present in {len(seen[fact])} admissions")
        rows.extend({"hadm_id": int(h), "fact": fact} for h in seen[fact])
    df = pd.DataFrame(rows, columns=["hadm_id", "fact"])
    df["structured_present"] = True
    return df


def load_notes(note_dir: Path, hadm_ids: set[int]) -> pd.DataFrame:
    log("\nreading discharge summaries")
    path = find_table(note_dir, "discharge")
    log(f"    discharge       <- {path}")
    pieces = []
    for chunk in pd.read_csv(path, usecols=["note_id", "hadm_id", "text"],
                             chunksize=20_000, low_memory=False):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk = chunk.assign(hadm_id=chunk["hadm_id"].astype("int64"))
        hit = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not hit.empty:
            pieces.append(hit)
    if not pieces:
        sys.exit("[fatal] no cohort admissions matched the discharge table")
    notes = pd.concat(pieces, ignore_index=True)
    notes["text"] = notes["text"].fillna("").astype(str)
    notes["n_chars"] = notes["text"].str.len()
    notes = (notes.sort_values(["n_chars", "note_id"], ascending=[False, True])
             .drop_duplicates(subset="hadm_id", keep="first")
             .reset_index(drop=True))
    notes["n_words"] = notes["text"].str.split().str.len()
    log(f"    {len(notes)} summaries (one per admission)")
    return notes


def scan_text(text: str, pat: re.Pattern) -> bool:
    for m in pat.finditer(text):
        if not NEGATION_RE.search(text[max(0, m.start() - 40):m.start()]):
            return True
    return False


def mention_flags(notes: pd.DataFrame) -> pd.DataFrame:
    compiled = {f: re.compile(s["text_re"], re.IGNORECASE)
                for f, s in FACTS.items()}
    rows = []
    for note_id, hadm, text in zip(notes["note_id"], notes["hadm_id"],
                                   notes["text"].str.lower()):
        for fact, pat in compiled.items():
            rows.append({"note_id": note_id, "hadm_id": int(hadm),
                         "fact": fact, "text_mentioned": scan_text(text, pat)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# lab-only SOFA (sensitivity strata)
# ---------------------------------------------------------------------------

def sofa_points(name: str, v: float) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    if name == "platelets":
        return 4 if v < 20 else 3 if v < 50 else 2 if v < 100 else 1 if v < 150 else 0
    if name == "bilirubin":
        return 4 if v >= 12 else 3 if v >= 6 else 2 if v >= 2 else 1 if v >= 1.2 else 0
    if name == "creatinine":
        return 4 if v >= 5 else 3 if v >= 3.5 else 2 if v >= 2 else 1 if v >= 1.2 else 0
    raise ValueError(name)


def lab_sofa(mimic_root: Path, cohort: pd.DataFrame, out: Path) -> pd.DataFrame:
    cache = out / "lab_sofa_24h.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if set(cohort["stay_id"]) <= set(df["stay_id"]):
            log(f"\nlab-only SOFA from cache {cache}")
            return df
    log("\ncomputing lab-only SOFA, first 24 ICU hours")
    dl = read_table(mimic_root, "d_labitems",
                    usecols=["itemid", "label", "fluid"])
    dl["label"] = dl["label"].fillna("").astype(str).str.strip()
    dl["fluid"] = dl["fluid"].fillna("").astype(str)
    item2comp: dict[int, str] = {}
    for comp, spec in LAB_SOFA.items():
        m = (dl["label"].str.contains(spec["label_re"], case=False, regex=True)
             & (dl["fluid"].str.lower() == "blood"))
        ids = dl.loc[m, "itemid"].astype(int).tolist()
        log(f"    {comp:<11} itemids {ids}  labels "
            f"{sorted(set(dl.loc[m, 'label']))}")
        if not ids:
            log(f"    !! no itemid for {comp}; lab-only SOFA skipped")
            return pd.DataFrame()
        item2comp.update({i: comp for i in ids})
    stays = cohort[["subject_id", "stay_id", "intime"]].copy()
    subj = set(stays["subject_id"])
    parts = []
    units: dict[str, set] = {c: set() for c in LAB_SOFA}
    for chunk in read_table(mimic_root, "labevents",
                            usecols=["subject_id", "itemid", "charttime",
                                     "valuenum", "valueuom"],
                            chunksize=5_000_000):
        chunk = chunk[chunk["itemid"].isin(item2comp.keys())
                      & chunk["subject_id"].isin(subj)].dropna(subset=["valuenum"])
        if chunk.empty:
            continue
        for comp in LAB_SOFA:
            u = chunk.loc[chunk["itemid"].map(item2comp) == comp, "valueuom"]
            units[comp].update(str(v).strip().lower() for v in u.dropna().unique())
        chunk = chunk.assign(charttime=pd.to_datetime(chunk["charttime"]))
        mm = chunk.merge(stays, on="subject_id", how="inner")
        hrs = (mm["charttime"] - mm["intime"]).dt.total_seconds() / 3600.0
        mm = mm[(hrs >= 0) & (hrs < 24)]
        if not mm.empty:
            parts.append(mm[["stay_id", "itemid", "valuenum"]])
    for comp, spec in LAB_SOFA.items():
        log(f"    {comp:<11} units found {sorted(units[comp])}")
        bad = units[comp] - set(spec["units"])
        if bad:
            sys.exit(f"[fatal] {comp}: unexpected units {sorted(bad)}; the SOFA "
                     f"thresholds assume {spec['units']}. Resolve or use "
                     f"--skip-lab-sofa.")
    lab = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["stay_id", "itemid", "valuenum"]))
    lab["comp"] = lab["itemid"].map(item2comp)
    df = stays[["stay_id"]].drop_duplicates().copy()
    df["lab_sofa"] = 0
    df["n_components_observed"] = 0
    for comp, spec in LAB_SOFA.items():
        worst = (lab[lab["comp"] == comp].groupby("stay_id")["valuenum"]
                 .agg(spec["agg"]).rename(comp))
        df = df.merge(worst, left_on="stay_id", right_index=True, how="left")
        df[f"{comp}_pts"] = [sofa_points(comp, float(v)) for v in df[comp]]
        df["lab_sofa"] += df[f"{comp}_pts"]
        df["n_components_observed"] += df[comp].notna().astype(int)
    df.to_csv(cache, index=False)
    log(f"    lab-only SOFA for {len(df)} stays; components observed "
        f"(mean) {df['n_components_observed'].mean():.2f} of 3")
    return df


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def http_json(url: str, payload: dict | None = None, timeout: float = 30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_manifest(base: str, model: str) -> dict:
    info: dict = {"ollama_url": base, "model": model}
    try:
        info["ollama_version"] = http_json(f"{base}/api/version").get("version")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[fatal] Ollama not reachable at {base}: {exc}")
    tags = http_json(f"{base}/api/tags").get("models", [])
    hit = [m for m in tags if m.get("name") == model or m.get("model") == model]
    if not hit:
        sys.exit(f"[fatal] model '{model}' not found in Ollama; available: "
                 f"{[m.get('name') for m in tags]}")
    info["digest"] = hit[0].get("digest")
    info["details"] = hit[0].get("details", {})
    return info


def build_user_prompt(text: str) -> str:
    qs = "\n".join(f"{i}. {f}: {FACTS[f]['question']}"
                   for i, f in enumerate(FACT_ORDER, 1))
    return ("DISCHARGE SUMMARY\n<<<\n" + text + "\n>>>\n\n"
            "Using only the discharge summary above, answer each question about "
            "THIS hospital admission. Answer \"yes\" only if the summary states or "
            "clearly implies that it happened during this admission; otherwise "
            "answer \"no\".\n\n" + qs + "\n\nReturn only a JSON object with the "
            "keys " + ", ".join(FACT_ORDER) + ", each \"yes\" or \"no\".")


JSON_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string", "enum": ["yes", "no"]} for f in FACT_ORDER},
    "required": FACT_ORDER,
}

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def ask_llm(base: str, model: str, text: str, num_ctx: int, seed: int,
            timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_user_prompt(text)}],
        "stream": False,
        "think": False,
        "format": JSON_SCHEMA,
        "options": {"temperature": 0, "seed": seed, "num_ctx": num_ctx,
                    "num_predict": 200},
    }
    last_err = ""
    for attempt in range(3):
        try:
            t0 = time.time()
            r = http_json(f"{base}/api/chat", payload, timeout=timeout)
            content = THINK_RE.sub("", r.get("message", {}).get("content", "")).strip()
            rec = {"raw": content[:500],
                   "prompt_eval_count": r.get("prompt_eval_count"),
                   "eval_count": r.get("eval_count"),
                   "seconds": round(time.time() - t0, 2),
                   "error": ""}
            try:
                obj = json.loads(content)
                rec["answers"] = {f: (str(obj.get(f, "")).strip().lower()
                                      if str(obj.get(f, "")).strip().lower()
                                      in ("yes", "no") else None)
                                  for f in FACT_ORDER}
            except json.JSONDecodeError:
                rec["answers"] = {f: None for f in FACT_ORDER}
                rec["error"] = "json_parse"
            return rec
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(5 * (attempt + 1))
        except urllib.error.HTTPError as exc:  # pragma: no cover
            body = exc.read().decode("utf-8", "replace")[:300]
            sys.exit(f"[fatal] Ollama HTTP {exc.code}: {body}")
    return {"answers": {f: None for f in FACT_ORDER}, "raw": "",
            "prompt_eval_count": None, "eval_count": None, "seconds": None,
            "error": last_err}


def prompt_sha() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + build_user_prompt("")).encode()
                          ).hexdigest()[:16]


def check_prompt_stamp(out: Path) -> None:
    stamp = out / "prompt_sha.txt"
    cur = prompt_sha()
    if (out / "llm_cache.jsonl").exists():
        if not stamp.exists() or stamp.read_text().strip() != cur:
            sys.exit(f"[fatal] {out}/llm_cache.jsonl was written with a different "
                     f"prompt (or by v1). Move that folder aside or use a new --out.")
    stamp.write_text(cur + "\n")


def read_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if not rec.get("error", "").startswith(("URLError", "Timeout",
                                                         "Connection")):
                    out[str(rec["note_id"])] = rec
    return out


def run_llm(notes: pd.DataFrame, args, cache_path: Path,
            order: list[str]) -> None:
    done = read_cache(cache_path)
    todo = [n for n in order if n not in done]
    log(f"\nLLM: {len(done)} cached, {len(todo)} to run "
        f"(model {args.model}, num_ctx {args.num_ctx})")
    if not todo:
        return
    text_of = dict(zip(notes["note_id"].astype(str), notes["text"]))
    words_of = dict(zip(notes["note_id"].astype(str), notes["n_words"]))
    t_start = time.time()
    with cache_path.open("a", encoding="utf-8") as fh:
        try:
            for i, nid in enumerate(todo, 1):
                rec = ask_llm(args.ollama_url, args.model, text_of[nid],
                              args.num_ctx, args.seed, args.timeout)
                pe = rec.get("prompt_eval_count") or 0
                rec["note_id"] = nid
                rec["n_words"] = int(words_of[nid])
                rec["possibly_truncated"] = bool(
                    pe + 300 >= args.num_ctx
                    or words_of[nid] * 1.6 > args.num_ctx - 600)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i in (5, 20) or i % 100 == 0 or i == len(todo):
                    el = time.time() - t_start
                    eta = el / i * (len(todo) - i)
                    log(f"    {i}/{len(todo)}  {el/i:.1f} s/note  "
                        f"ETA {eta/3600:.1f} h")
        except KeyboardInterrupt:
            log("\n[interrupted] progress is cached; rerun to continue")
            sys.exit(130)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, min(p, c - h)), min(1.0, max(p, c + h))


def fit_logit(df: pd.DataFrame, y: str, terms: list[str], target: str,
              fact_fe: bool, label: str) -> tuple[dict, str]:
    """Logistic GLM with cluster-robust SE by subject_id."""
    if len(df) < 30 or df[y].nunique() < 2:
        return ({"model": label, "n": len(df), "status": "skipped"},
                f"[{label}] skipped (n={len(df)}, outcome levels={df[y].nunique()})")
    X = df[terms].astype(float).copy()
    if fact_fe and df["fact"].nunique() > 1:
        d = pd.get_dummies(df["fact"], prefix="fact", drop_first=True,
                           dtype=float)
        X = pd.concat([X, d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.GLM(df[y].astype(float), X,
                     family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": df["subject_id"].values})
    except Exception as exc:  # noqa: BLE001
        return ({"model": label, "n": len(df), "status": f"failed: {exc}"},
                f"[{label}] model failed: {exc}")
    b = float(res.params[target])
    lo, hi = (float(v) for v in res.conf_int().loc[target].values)
    rec = {"model": label, "n": len(df), "events": int(df[y].sum()),
           "clusters": int(df["subject_id"].nunique()), "term": target,
           "OR": math.exp(b), "OR_lo": math.exp(lo), "OR_hi": math.exp(hi),
           "p": float(res.pvalues[target]), "status": "ok"}
    head = (f"[{label}] n={rec['n']} events={rec['events']} "
            f"clusters={rec['clusters']} OR per unit {target}="
            f"{rec['OR']:.3f} (95% CI {rec['OR_lo']:.3f}-{rec['OR_hi']:.3f}) "
            f"p={rec['p']:.4g}")
    return rec, head + "\n" + res.summary().as_text() + "\n"


def decide(rec: dict) -> str:
    if rec.get("status") != "ok":
        return "not estimable"
    if rec["OR"] <= 1:
        return "CONTRADICTED (flat or falling)"
    if rec["OR_lo"] > 1:
        return "SUPPORTED"
    return "INCONCLUSIVE (OR > 1, lower bound <= 1)"


def analyse(per: pd.DataFrame, out: Path, have_lab: bool) -> None:
    log("\n=== P8 analysis ===")
    per = per.copy()
    per["llm_valid"] = per["llm_answer"].isin(["yes", "no"])
    per["excluded"] = ~per["llm_valid"] | per["possibly_truncated"].astype(bool)
    per["miss"] = (per["llm_answer"] == "no").astype(int)
    per["false_yes"] = (per["llm_answer"] == "yes").astype(int)
    per["log_words"] = np.log(per["n_words"].clip(lower=1))

    ok = per[~per["excluded"]]
    sets = {
        "PRIMARY documented positives": (
            ok[ok["structured_present"] & ok["text_mentioned"]], "miss"),
        "S1 all structured positives": (ok[ok["structured_present"]], "miss"),
        "S2 unsupported positives": (
            ok[~ok["structured_present"] & ~ok["text_mentioned"]], "false_yes"),
    }

    # absolute rates
    rate_rows = []
    for name, (d, y) in sets.items():
        for fact in FACT_ORDER + ["POOLED"]:
            dd = d if fact == "POOLED" else d[d["fact"] == fact]
            for q in range(4):
                s = dd[dd["q_num"] == q]
                p, lo, hi = wilson(int(s[y].sum()), len(s))
                rate_rows.append({"set": name, "fact": fact, "q_num": q,
                                  "n": len(s), "events": int(s[y].sum()),
                                  "rate": p, "lo95": lo, "hi95": hi})
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(out / "p8_rates_by_stratum.csv", index=False)

    excl = (per.groupby("q_num")
            .agg(note_fact=("excluded", "size"), excluded=("excluded", "sum"),
                 parse_fail=("llm_valid", lambda s: int((~s).sum())))
            .reset_index())

    recs, txt = [], []
    prim, _ = sets["PRIMARY documented positives"]
    r, t = fit_logit(prim, "miss", ["q_num"], "q_num", True, "PRIMARY")
    recs.append(r); txt.append(t)
    for name, key in (("S1", "S1 all structured positives"),
                      ("S2", "S2 unsupported positives")):
        d, y = sets[key]
        r2, t2 = fit_logit(d, y, ["q_num"], "q_num", True, name)
        recs.append(r2); txt.append(t2)
    r3, t3 = fit_logit(prim, "miss", ["q_num", "log_words"], "q_num", True,
                       "S3 primary + log(words)")
    recs.append(r3); txt.append(t3)
    if have_lab and "lab_sofa" in prim.columns:
        d4 = prim[prim["fact"].isin(["vasopressor", "rbc_transfusion"])
                  & prim["lab_sofa"].notna()]
        r4, t4 = fit_logit(d4, "miss", ["lab_sofa"], "lab_sofa", True,
                           "S4 lab-only SOFA (per point), vaso + rbc")
    else:
        r4, t4 = ({"model": "S4", "status": "skipped (no lab SOFA)"},
                  "[S4] skipped (no lab SOFA)")
    recs.append(r4); txt.append(t4)
    for fact in FACT_ORDER:
        r5, t5 = fit_logit(prim[prim["fact"] == fact], "miss", ["q_num"],
                           "q_num", False, f"S5 {fact}")
        recs.append(r5); txt.append(t5)
    d6 = per[per["structured_present"] & per["text_mentioned"]].copy()
    d6["miss"] = ((d6["llm_answer"] != "yes") | d6["excluded"]).astype(int)
    r6, t6 = fit_logit(d6, "miss", ["q_num"], "q_num", True,
                       "S6 exclusions counted as misses")
    recs.append(r6); txt.append(t6)

    (out / "p8_models.txt").write_text("\n".join(txt))
    pd.DataFrame(recs).to_csv(out / "p8_models.csv", index=False)

    verdict = decide(recs[0])
    pr = rates[(rates["set"] == "PRIMARY documented positives")
               & (rates["fact"] == "POOLED")]
    summ = [ANALYSIS_PLAN, "", "RESULT", "",
            "Exclusions (note x fact) by quartile:", excl.to_string(index=False),
            "",
            "Primary set, pooled miss rate by quartile (Wilson 95% CI):",
            pr[["q_num", "n", "events", "rate", "lo95", "hi95"]].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"),
            ""]
    for r_ in recs:
        if r_.get("status") == "ok":
            summ.append(f"{r_['model']:<45} OR {r_['OR']:.3f} "
                        f"({r_['OR_lo']:.3f}-{r_['OR_hi']:.3f}) p={r_['p']:.4g} "
                        f"n={r_['n']} events={r_['events']}")
        else:
            summ.append(f"{r_['model']:<45} {r_.get('status')}")
    summ += ["", f"PRIMARY DECISION: {verdict}"]
    (out / "p8_summary.txt").write_text("\n".join(summ) + "\n")
    for l in summ[summ.index("RESULT"):]:
        log(l)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (name, (_, y)) in zip(axes, sets.items()):
        sub = rates[rates["set"] == name]
        for fact in FACT_ORDER + ["POOLED"]:
            s = sub[sub["fact"] == fact].sort_values("q_num")
            pooled = fact == "POOLED"
            ax.errorbar(s["q_num"] + 1, s["rate"],
                        yerr=[(s["rate"] - s["lo95"]).clip(lower=0),
                              (s["hi95"] - s["rate"]).clip(lower=0)],
                        marker="s" if pooled else "o",
                        color="black" if pooled else None,
                        linewidth=2 if pooled else 1, capsize=3,
                        alpha=1 if pooled else 0.7,
                        label="pooled" if pooled else fact)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("derangement quartile")
        ax.set_ylabel("LLM 'no'" if y == "miss" else "LLM 'yes'")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "p8_error_by_stratum.png", dpi=140)
    plt.close(fig)
    log(f"\nwritten to {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", type=Path, default=DEF_MIMIC_ROOT)
    ap.add_argument("--mimic-note", type=Path, default=DEF_MIMIC_NOTE)
    ap.add_argument("--acuity", type=Path, default=DEF_ACUITY)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--ollama-url", default=DEF_OLLAMA)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", type=int, default=0,
                    help="run the LLM on N notes (stratified) and stop")
    ap.add_argument("--analyse-only", action="store_true",
                    help="no LLM calls; analyse the notes already cached")
    ap.add_argument("--skip-lab-sofa", action="store_true")
    args = ap.parse_args()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    plan_path = args.out / "analysis_plan.txt"
    if plan_path.exists() and plan_path.read_text() != ANALYSIS_PLAN:
        sys.exit(f"[fatal] {plan_path} differs from the plan in this script. "
                 f"The plan is frozen once written; use a new --out.")
    plan_path.write_text(ANALYSIS_PLAN)
    check_prompt_stamp(args.out)
    plan_hash = hashlib.sha256(ANALYSIS_PLAN.encode()).hexdigest()[:16]

    cohort = build_cohort(args.mimic_root.expanduser(), args.acuity.expanduser())
    itemids = resolve_itemids(args.mimic_root.expanduser(), args.out)
    struct = structured_presence(args.mimic_root.expanduser(), itemids,
                                 set(cohort["hadm_id"]))
    notes = load_notes(args.mimic_note.expanduser(), set(cohort["hadm_id"]))
    notes["note_id"] = notes["note_id"].astype(str)
    cohort = cohort[cohort["hadm_id"].isin(set(notes["hadm_id"]))]
    log("    notes per quartile (q_num): "
        + str({int(k): int(v) for k, v in
               cohort["q_num"].value_counts().sort_index().items()}))

    manifest = {
        "script": Path(__file__).name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_sha256_16": plan_hash,
        "mimic_root": str(args.mimic_root), "mimic_note": str(args.mimic_note),
        "acuity": str(args.acuity), "n_notes": int(len(notes)),
        "num_ctx": args.num_ctx, "seed": args.seed,
        "pandas": pd.__version__, "numpy": np.__version__,
        "statsmodels": statsmodels.__version__,
    }

    cache_path = args.out / "llm_cache.jsonl"
    manifest["prompt_sha256_16"] = prompt_sha()
    nid_q = dict(zip(notes["note_id"],
                     notes["hadm_id"].map(dict(zip(cohort["hadm_id"],
                                                   cohort["q_num"])))))
    rng = random.Random(args.seed)
    order = sorted(notes["note_id"])
    rng.shuffle(order)   # random order: an interrupted run is a random subsample

    if not args.analyse_only:
        manifest["ollama"] = ollama_manifest(args.ollama_url, args.model)
        log(f"\nOllama {manifest['ollama'].get('ollama_version')}  model "
            f"{args.model}  digest {str(manifest['ollama'].get('digest'))[:12]}  "
            f"{manifest['ollama'].get('details', {}).get('quantization_level')}")
        if args.pilot:
            per_q = max(1, args.pilot // 4)
            pick = []
            for q in range(4):
                pick += [n for n in order if nid_q.get(n) == q][:per_q]
            run_llm(notes, args, cache_path, pick)
            cache = read_cache(cache_path)
            recs = [cache[n] for n in pick if n in cache]
            secs = [r["seconds"] for r in recs if r.get("seconds") is not None]
            log("\nPILOT")
            log(f"    notes run          {len(recs)}")
            log(f"    seconds/note       median {np.median(secs):.1f}  "
                f"max {np.max(secs):.1f}" if secs else "    no timings")
            log(f"    parse failures     "
                f"{sum(any(v is None for v in r['answers'].values()) for r in recs)}")
            log(f"    possibly truncated {sum(r['possibly_truncated'] for r in recs)}")
            log(f"    max prompt tokens  "
                f"{max((r.get('prompt_eval_count') or 0) for r in recs)}")
            if secs:
                log(f"    ETA full run       "
                    f"{np.median(secs) * (len(notes) - len(cache)) / 3600:.1f} h "
                    f"for {len(notes) - len(cache)} remaining notes")
            (args.out / "manifest_pilot.json").write_text(
                json.dumps(manifest, indent=2, default=str))
            return
        run_llm(notes, args, cache_path, order)

    cache = read_cache(cache_path)
    if not cache:
        sys.exit("[fatal] no LLM answers cached; run without --analyse-only")
    log(f"\n{len(cache)} of {len(notes)} notes have LLM answers")

    have_lab = False
    lab = pd.DataFrame()
    if not args.skip_lab_sofa:
        lab = lab_sofa(args.mimic_root.expanduser(), cohort, args.out)
        have_lab = not lab.empty

    ment = mention_flags(notes)
    grid = (cohort.drop(columns=["intime"]).assign(_k=1)
            .merge(pd.DataFrame({"fact": FACT_ORDER, "_k": 1}), on="_k")
            .drop(columns="_k"))
    per = (grid.merge(notes[["note_id", "hadm_id", "n_words", "n_chars"]],
                      on="hadm_id", how="inner")
           .merge(ment[["note_id", "fact", "text_mentioned"]],
                  on=["note_id", "fact"], how="left")
           .merge(struct, on=["hadm_id", "fact"], how="left"))
    per["structured_present"] = per["structured_present"].astype("boolean").fillna(False).astype(bool)
    per["text_mentioned"] = per["text_mentioned"].astype("boolean").fillna(False).astype(bool)
    per = per[per["note_id"].isin(cache.keys())].copy()
    per["llm_answer"] = [cache[n]["answers"].get(f)
                         for n, f in zip(per["note_id"], per["fact"])]
    per["possibly_truncated"] = [bool(cache[n].get("possibly_truncated"))
                                 for n in per["note_id"]]
    per["prompt_eval_count"] = [cache[n].get("prompt_eval_count")
                                for n in per["note_id"]]
    if have_lab:
        per = per.merge(lab[["stay_id", "lab_sofa", "n_components_observed"]],
                        on="stay_id", how="left")
    per.to_csv(args.out / "p8_per_note_fact.csv", index=False)

    manifest["n_notes_answered"] = len(cache)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                       default=str))
    analyse(per, args.out, have_lab)
    log("\nCAVEATS")
    log("  * itemids resolved by d_items label regex -- inspect itemids.txt")
    log("  * 'documented' is a surface regex with a 40-char negation window")
    log("  * reference covers the ICU stay; the question covers the admission")
    log("  * truncation flag is heuristic (prompt tokens, words x 1.6)")


if __name__ == "__main__":
    main()
