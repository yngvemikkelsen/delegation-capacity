# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
# ]
# ///
"""
extension_p8_vaso_any_v3.py -- POST HOC check requested in review (Paper 24,
Frontiers, Reviewer 2, comment 1). Not part of the prespecified P8 analysis.

QUESTION
    In P8 the delegate was asked whether the patient received "any
    intravenous norepinephrine, epinephrine, phenylephrine, vasopressin or
    dopamine". Many notes record vasopressor use only as "pressors" or by
    brand name. Does the miss rate, and its gradient across disturbance
    quartiles, change when the delegate is asked about ANY vasopressor?

WHAT CHANGES
    Only the vasopressor question. Model, digest, decoding (temperature 0,
    seed 42, think off, JSON schema, num_ctx 32768, num_predict 200), system
    prompt, prompt frame, the other two questions and the notes are those of
    extension_p8_error_demand_v2.py.
        v2: Did the patient receive any intravenous norepinephrine,
            epinephrine, phenylephrine, vasopressin or dopamine?
        v3: Did the patient receive any intravenous vasopressor (for example
            norepinephrine, epinephrine, phenylephrine, vasopressin or
            dopamine), including treatment recorded only as pressors or
            pressor support?

INPUTS (read only; nothing in them is changed)
    --p8-dir   ~/extension_results/p8                   p8_per_note_fact.csv
    --adj-dir  ~/extension_results/p8_adjudication      key/diag2_notes.csv
    --mimic-note  MIMIC-IV-Note 2.2 note folder          discharge.csv.gz

RUN ORDER (resumable; an interrupted run continues from llm_cache.jsonl)
    Notes are run in three blocks, each in random order:
      A  vasopressor primary set (in ICU record and flagged in the note)
      B  vasopressor absent from the ICU record
      C  the remaining notes (in the ICU record, not flagged in the note)
    --scope A | AB | ABC (default ABC). The analysis uses whatever is cached.
        uv run --offline extension_p8_vaso_any_v3.py
        uv run --offline extension_p8_vaso_any_v3.py --scope A
        uv run --offline extension_p8_vaso_any_v3.py --analyse-only

OUTPUTS (in --out, default ~/extension_results/p8_vaso_any); aggregates only,
no note text
    analysis_plan_v3.txt, prompt_sha.txt, manifest_v3.json
    llm_cache.jsonl              one line per note (answers, token counts)
    v3_summary.txt               all results below
    v3_rates.csv                 rates with Wilson 95% CIs
    v3_models.csv                logistic models (OR per quartile step)

Ollama must be running with qwen3:30b pulled.
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

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

HOME = Path.home()
DEF_P8 = HOME / "extension_results" / "p8"
DEF_ADJ = HOME / "extension_results" / "p8_adjudication"
DEF_MIMIC_NOTE = HOME / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"
DEF_OUT = HOME / "extension_results" / "p8_vaso_any"
DEF_OLLAMA = "http://localhost:11434"
DEF_MODEL = "qwen3:30b"
EXPECTED_DIGEST_PREFIX = "ad815644918f"

FACT_ORDER = ["vasopressor", "renal_replacement", "rbc_transfusion"]
QUESTIONS = {
    # v3: only this question differs from extension_p8_error_demand_v2.py
    "vasopressor": "Did the patient receive any intravenous vasopressor (for "
                   "example norepinephrine, epinephrine, phenylephrine, "
                   "vasopressin or dopamine), including treatment recorded "
                   "only as pressors or pressor support?",
    # identical to v2
    "renal_replacement": "Did the patient receive renal replacement therapy "
                         "(haemodialysis, CRRT, CVVH, CVVHD, CVVHDF, SCUF or "
                         "peritoneal dialysis)?",
    "rbc_transfusion": "Did the patient receive a transfusion of red blood cells "
                       "(packed red blood cells)?",
}
SYSTEM_PROMPT = ("You are a careful clinical chart abstractor. You answer "
                 "strictly from the document provided.")
CATEGORIES = ["generic", "brand_only", "umbrella_only", "loose_only"]

PLAN = """\
P8 v3 POST HOC CHECK (REVIEW REQUEST) -- ANY-VASOPRESSOR QUESTION
(written before the first LLM call; do not edit after the run starts)

Change      Only the vasopressor question (see script header). Everything
            else as extension_p8_error_demand_v2.py.
Notes       The 4,078 notes of the P8 run; reference standard, documentation
            flag, quartiles and vocabulary categories are read from the P8 and
            post hoc outputs, not recomputed.
Endpoints   (descriptive; no decision rule; reported as post hoc)
  E1  Vasopressor primary set (in ICU record and flagged in the note):
      miss rate (answer "no") under v2 and v3, overall and by quartile, Wilson
      95% CIs; paired changes (v2 miss -> v3 hit, v2 hit -> v3 miss).
  E2  E1 by vocabulary category (generic, brand_only, umbrella_only,
      loose_only), v2 and v3.
  E3  Logistic models, cluster-robust SE by subject, OR per quartile step:
      M1 v3 miss ~ q_num; M2 v3 miss ~ q_num + category; M3 v3 miss ~ q_num,
      generic notes only. v2 M1 reproduced as a check (expected 1.156).
  E4  Vasopressor absent from the ICU record: "yes" rate under v2 and v3,
      separately for notes with and without a flagged mention.
  E5  All vasopressor-positive admissions (S1): miss rate v2 and v3; v3 OR.
  E6  Pooled primary model (three facts, fact indicators) with v3 answers;
      agreement of v3 with v2 answers for the two unchanged questions.
  E7  Vasopressor answer error against the ICU reference (all notes run),
      v2 and v3, overall and by quartile.
"""


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


# ---------------------------------------------------------------------------
# inputs from the P8 run and the post hoc diagnostic
# ---------------------------------------------------------------------------

def load_per(p8_dir: Path) -> pd.DataFrame:
    path = p8_dir / "p8_per_note_fact.csv"
    if not path.exists():
        sys.exit(f"[fatal] {path} not found")
    per = pd.read_csv(path, dtype={"note_id": str}, low_memory=False)
    need = {"note_id", "subject_id", "hadm_id", "q_num", "fact",
            "structured_present", "text_mentioned", "llm_answer",
            "possibly_truncated"}
    miss = need - set(per.columns)
    if miss:
        sys.exit(f"[fatal] {path} lacks columns {sorted(miss)}")
    for c in ("structured_present", "text_mentioned", "possibly_truncated"):
        per[c] = per[c].astype(str).str.lower() == "true"
    per["llm_answer"] = per["llm_answer"].astype(str).str.lower()
    per["q_num"] = per["q_num"].astype(int)
    log(f"    P8 per-note table <- {path}")
    log(f"    notes {per.note_id.nunique()}  rows {len(per)}")
    return per


def load_categories(adj_dir: Path) -> pd.DataFrame:
    path = adj_dir / "key" / "diag2_notes.csv"
    if not path.exists():
        sys.exit(f"[fatal] {path} not found (written by "
                 f"extension_p8_vaso_diagnostic_v2.py)")
    cat = pd.read_csv(path, dtype={"note_id": str})
    if not {"note_id", "category"} <= set(cat.columns):
        sys.exit(f"[fatal] {path} lacks note_id/category")
    log(f"    vocabulary categories <- {path} ({len(cat)} notes)")
    return cat[["note_id", "category"]]


def load_texts(note_dir: Path, note_ids: set[str]) -> dict[str, str]:
    path = find_table(note_dir, "discharge")
    log(f"    discharge <- {path}")
    out: dict[str, str] = {}
    for ch in pd.read_csv(path, usecols=["note_id", "text"], dtype={"note_id": str},
                          chunksize=20_000, low_memory=False):
        h = ch[ch["note_id"].isin(note_ids)]
        out.update(zip(h["note_id"], h["text"].fillna("").astype(str)))
    missing = note_ids - set(out)
    if missing:
        sys.exit(f"[fatal] {len(missing)} note_ids not in the discharge table")
    return out


# ---------------------------------------------------------------------------
# Ollama (as v2; only the question text differs)
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
    qs = "\n".join(f"{i}. {f}: {QUESTIONS[f]}"
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
                     f"prompt. Move that folder aside or use a new --out.")
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


def run_llm(texts: dict[str, str], words: dict[str, int], args,
            cache_path: Path, order: list[tuple[str, str]]) -> None:
    done = read_cache(cache_path)
    todo = [(n, b) for n, b in order if n not in done]
    log(f"\nLLM: {len(done)} cached, {len(todo)} to run "
        f"(model {args.model}, num_ctx {args.num_ctx})")
    for blk in "ABC":
        k = sum(1 for _, b in todo if b == blk)
        if k:
            log(f"    block {blk}: {k} to run")
    if not todo:
        return
    t_start = time.time()
    with cache_path.open("a", encoding="utf-8") as fh:
        try:
            for i, (nid, blk) in enumerate(todo, 1):
                rec = ask_llm(args.ollama_url, args.model, texts[nid],
                              args.num_ctx, args.seed, args.timeout)
                pe = rec.get("prompt_eval_count") or 0
                rec["note_id"] = nid
                rec["block"] = blk
                rec["n_words"] = int(words[nid])
                rec["possibly_truncated"] = bool(
                    pe + 300 >= args.num_ctx
                    or words[nid] * 1.6 > args.num_ctx - 600)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                nxt = todo[i][1] if i < len(todo) else None
                if i in (5, 20) or i % 100 == 0 or i == len(todo) or nxt != blk:
                    el = time.time() - t_start
                    eta = el / i * (len(todo) - i)
                    left_blk = sum(1 for _, b in todo[i:] if b == blk)
                    log(f"    {i}/{len(todo)}  block {blk} ({left_blk} left in "
                        f"block)  {el / i:.1f} s/note  ETA all {eta / 3600:.1f} h")
        except KeyboardInterrupt:
            log("\n[interrupted] progress is cached; rerun to continue")
            sys.exit(130)


# ---------------------------------------------------------------------------
# statistics (as v2)
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
              dummies: str | None, label: str) -> dict:
    """Logistic GLM, cluster-robust SE by subject_id; optional categorical."""
    if len(df) < 30 or df[y].nunique() < 2:
        return {"model": label, "n": len(df), "status": "skipped"}
    X = df[terms].astype(float).copy()
    if dummies and df[dummies].nunique() > 1:
        d = pd.get_dummies(df[dummies], prefix=dummies, drop_first=True, dtype=float)
        X = pd.concat([X, d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": df["subject_id"].values})
    except Exception as exc:  # noqa: BLE001
        return {"model": label, "n": len(df), "status": f"failed: {exc}"}
    b = float(res.params[target])
    lo, hi = (float(v) for v in res.conf_int().loc[target].values)
    return {"model": label, "n": len(df), "events": int(df[y].sum()),
            "clusters": int(df["subject_id"].nunique()), "OR": math.exp(b),
            "OR_lo": math.exp(lo), "OR_hi": math.exp(hi),
            "p": float(res.pvalues[target]), "status": "ok"}


def fmt_model(r: dict) -> str:
    if r.get("status") != "ok":
        return f"{r['model']:<58} {r.get('status')} (n={r.get('n')})"
    return (f"{r['model']:<58} OR {r['OR']:.3f} ({r['OR_lo']:.3f}-{r['OR_hi']:.3f}) "
            f"p={r['p']:.3g} n={r['n']} events={r['events']}")


def rate_rows(d: pd.DataFrame, y: str, label: str, by: str | None = None) -> list[dict]:
    rows = []
    groups = [("all", d)] + ([(str(k), g) for k, g in d.groupby(by)] if by else [])
    for k, g in groups:
        p, lo, hi = wilson(int(g[y].sum()), len(g))
        rows.append({"set": label, "group": k, "n": len(g), "events": int(g[y].sum()),
                     "rate": p, "lo95": lo, "hi95": hi})
    return rows


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyse(per: pd.DataFrame, cat: pd.DataFrame, cache: dict[str, dict],
            out: Path) -> None:
    L: list[str] = ["POST HOC CHECK -- any-vasopressor question (v3) vs P8 (v2)",
                    "Requested in review; not part of the prespecified P8 analysis.", ""]
    rates: list[dict] = []
    models: list[dict] = []

    def v3_answer(nid: str, fact: str):
        rec = cache.get(nid)
        if rec is None or rec.get("possibly_truncated"):
            return None
        return rec["answers"].get(fact)

    per = per.copy()
    per["v3_answer"] = [v3_answer(n, f) for n, f in zip(per["note_id"], per["fact"])]
    per["v2_valid"] = per["llm_answer"].isin(["yes", "no"]) & ~per["possibly_truncated"]
    per["v3_valid"] = per["v3_answer"].isin(["yes", "no"])
    both = per[per["v2_valid"] & per["v3_valid"]].copy()
    both["v2_miss"] = (both["llm_answer"] == "no").astype(int)
    both["v3_miss"] = (both["v3_answer"] == "no").astype(int)
    both["v2_yes"] = 1 - both["v2_miss"]
    both["v3_yes"] = 1 - both["v3_miss"]

    nv = per.drop_duplicates("note_id")
    L.append(f"Notes in P8: {nv.note_id.nunique()}; answered under v3: "
             f"{sum(1 for n in nv.note_id if n in cache)}; "
             f"v3 parse failures: {int((per['fact'].eq('vasopressor') & per['note_id'].isin(list(cache)) & ~per['v3_valid']).sum())}")
    L.append("")

    v = both[both["fact"] == "vasopressor"].copy()
    prim = v[v["structured_present"] & v["text_mentioned"]].merge(cat, on="note_id",
                                                                    how="left")
    if prim["category"].isna().any():
        L.append(f"[warning] {int(prim['category'].isna().sum())} primary-set notes "
                 f"without a vocabulary category")
    prim["q1"] = prim["q_num"] + 1

    # E1
    L.append(f"E1  Vasopressor primary set answered under v3: {len(prim)} notes "
             f"(of 1,699 in P8)")
    for tag, y in (("v2", "v2_miss"), ("v3", "v3_miss")):
        rr = rate_rows(prim, y, f"E1 primary {tag}", "q1")
        rates += rr
        L.append(f"    {tag} miss: " + "  ".join(
            f"{r['group']}: {r['events']}/{r['n']} {100 * r['rate']:.1f}% "
            f"({100 * r['lo95']:.1f}-{100 * r['hi95']:.1f})" for r in rr))
    t = pd.crosstab(prim["v2_miss"].map({1: "v2 miss", 0: "v2 hit"}),
                    prim["v3_miss"].map({1: "v3 miss", 0: "v3 hit"}))
    L.append("    paired answers:\n" + "\n".join("      " + s for s in
                                                 t.to_string().splitlines()))
    L.append("")

    # E2
    L.append("E2  By vocabulary category (miss rate, %, Wilson 95% CI)")
    for c in CATEGORIES:
        g = prim[prim["category"] == c]
        if g.empty:
            continue
        r2 = rate_rows(g, "v2_miss", f"E2 {c} v2", "q1")
        r3 = rate_rows(g, "v3_miss", f"E2 {c} v3", "q1")
        rates += r2 + r3
        a, b = r2[0], r3[0]
        L.append(f"    {c:<14} n={a['n']:<5} v2 {100 * a['rate']:5.1f} "
                 f"({100 * a['lo95']:.1f}-{100 * a['hi95']:.1f})   v3 "
                 f"{100 * b['rate']:5.1f} ({100 * b['lo95']:.1f}-{100 * b['hi95']:.1f})"
                 f"   v3 by quartile " + ", ".join(f"{100 * r['rate']:.1f}"
                                                   for r in r3[1:]))
    L.append("")

    # E3
    L.append("E3  Models (OR per quartile step, cluster-robust by subject)")
    models.append(fit_logit(prim, "v2_miss", ["q_num"], "q_num", None,
                            "M1 v2 miss ~ q (check: P8 S5 = 1.156)"))
    models.append(fit_logit(prim, "v3_miss", ["q_num"], "q_num", None,
                            "M1 v3 miss ~ q"))
    models.append(fit_logit(prim.dropna(subset=["category"]), "v3_miss", ["q_num"],
                            "q_num", "category", "M2 v3 miss ~ q + category"))
    models.append(fit_logit(prim[prim["category"] == "generic"], "v3_miss",
                            ["q_num"], "q_num", None, "M3 v3 miss ~ q, generic notes"))
    L += ["    " + fmt_model(r) for r in models[-4:]]
    L.append("")

    # E4
    neg = v[~v["structured_present"]]
    L.append(f"E4  Vasopressor absent from the ICU record: {len(neg)} notes answered")
    for lab, g in (("flagged mention", neg[neg["text_mentioned"]]),
                   ("no flagged mention", neg[~neg["text_mentioned"]])):
        r2 = rate_rows(g, "v2_yes", f"E4 {lab} v2")[0]
        r3 = rate_rows(g, "v3_yes", f"E4 {lab} v3")[0]
        rates += [r2, r3]
        L.append(f"    {lab:<20} n={r2['n']:<5} 'yes' v2 {r2['events']} "
                 f"({100 * r2['rate']:.1f}%)   v3 {r3['events']} ({100 * r3['rate']:.1f}%)")
    L.append("")

    # E5
    s1 = v[v["structured_present"]]
    r2 = rate_rows(s1, "v2_miss", "E5 S1 v2", "q_num")
    r3 = rate_rows(s1, "v3_miss", "E5 S1 v3", "q_num")
    rates += r2 + r3
    L.append(f"E5  Vasopressor in ICU record (S1; complete only after block C): n={len(s1)}  miss v2 "
             f"{100 * r2[0]['rate']:.1f}%  v3 {100 * r3[0]['rate']:.1f}%")
    models.append(fit_logit(s1, "v3_miss", ["q_num"], "q_num", None, "S1 v3 miss ~ q"))
    L.append("    " + fmt_model(models[-1]))
    L.append("")

    # E6
    pp = both[both["structured_present"] & both["text_mentioned"]]
    models.append(fit_logit(pp, "v3_miss", ["q_num"], "q_num", "fact",
                            "PRIMARY pooled, v3 answers (fact indicators)"))
    models.append(fit_logit(pp, "v2_miss", ["q_num"], "q_num", "fact",
                            "PRIMARY pooled, v2 answers, same notes (check 1.136)"))
    L.append("E6  Pooled primary model (complete only when all three blocks have run)")
    L += ["    " + fmt_model(r) for r in models[-2:]]
    for f in ("renal_replacement", "rbc_transfusion"):
        g = both[both["fact"] == f]
        agree = (g["llm_answer"] == g["v3_answer"]).mean() if len(g) else float("nan")
        L.append(f"    {f:<18} unchanged question: v3 = v2 answer in "
                 f"{100 * agree:.2f}% of {len(g)} notes")
    L.append("")

    # E7
    L.append("E7  Vasopressor answer error against the ICU reference "
             "(notes answered under v3)")
    v["ref"] = v["structured_present"].astype(int)
    v["v2_err"] = (v["v2_yes"] != v["ref"]).astype(int)
    v["v3_err"] = (v["v3_yes"] != v["ref"]).astype(int)
    for tag in ("v2", "v3"):
        e = v.groupby("q_num")[f"{tag}_err"].mean()
        L.append(f"    {tag}: overall {100 * v[f'{tag}_err'].mean():.1f}%  by quartile "
                 + " ".join(f"{100 * e.get(q, float('nan')):.1f}" for q in range(4))
                 + f"  (n={len(v)})")
    L.append("    (complete only when all three blocks have run)")

    (out / "v3_summary.txt").write_text("\n".join(L) + "\n")
    pd.DataFrame(rates).to_csv(out / "v3_rates.csv", index=False)
    pd.DataFrame(models).to_csv(out / "v3_models.csv", index=False)
    log("")
    for line in L:
        log(line)
    log(f"\nwritten to {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p8-dir", type=Path, default=DEF_P8)
    ap.add_argument("--adj-dir", type=Path, default=DEF_ADJ)
    ap.add_argument("--mimic-note", type=Path, default=DEF_MIMIC_NOTE)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--ollama-url", default=DEF_OLLAMA)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scope", choices=["A", "AB", "ABC"], default="ABC")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    plan_path = out / "analysis_plan_v3.txt"
    if plan_path.exists() and plan_path.read_text() != PLAN:
        sys.exit(f"[fatal] {plan_path} differs from the plan in this script; "
                 f"use a new --out.")
    plan_path.write_text(PLAN)
    check_prompt_stamp(out)

    log("loading inputs")
    per = load_per(args.p8_dir.expanduser())
    cat = load_categories(args.adj_dir.expanduser())
    vaso = per[per["fact"] == "vasopressor"]
    blk_a = sorted(vaso.loc[vaso["structured_present"] & vaso["text_mentioned"], "note_id"])
    blk_b = sorted(vaso.loc[~vaso["structured_present"], "note_id"])
    blk_c = sorted(set(per["note_id"]) - set(blk_a) - set(blk_b))
    log(f"    block A (vasopressor primary set) {len(blk_a)}; block B (absent "
        f"from ICU record) {len(blk_b)}; block C (rest) {len(blk_c)}")
    if len(blk_a) != 1699:
        log(f"[warning] block A has {len(blk_a)} notes; the P8 primary set had 1,699")

    manifest = {"script": Path(__file__).name,
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "plan_sha256_16": hashlib.sha256(PLAN.encode()).hexdigest()[:16],
                "prompt_sha256_16": prompt_sha(), "scope": args.scope,
                "num_ctx": args.num_ctx, "seed": args.seed,
                "pandas": pd.__version__, "numpy": np.__version__,
                "statsmodels": statsmodels.__version__}
    cache_path = out / "llm_cache.jsonl"

    if not args.analyse_only:
        manifest["ollama"] = ollama_manifest(args.ollama_url, args.model)
        dig = str(manifest["ollama"].get("digest"))
        log(f"\nOllama {manifest['ollama'].get('ollama_version')}  model {args.model}  "
            f"digest {dig[:12]}  "
            f"{manifest['ollama'].get('details', {}).get('quantization_level')}")
        if not dig.startswith(EXPECTED_DIGEST_PREFIX):
            sys.exit(f"[fatal] model digest {dig[:12]} differs from the P8 run "
                     f"({EXPECTED_DIGEST_PREFIX}); the comparison would not be like "
                     f"for like")
        rng = random.Random(args.seed)
        order: list[tuple[str, str]] = []
        for name, blk in (("A", blk_a), ("B", blk_b), ("C", blk_c)):
            if name not in args.scope:
                continue
            ids = list(blk)
            rng.shuffle(ids)
            order += [(n, name) for n in ids]
        need = {n for n, _ in order}
        log("\nreading discharge summaries")
        texts = load_texts(args.mimic_note.expanduser(), need)
        words = {n: len(t.split()) for n, t in texts.items()}
        run_llm(texts, words, args, cache_path, order)

    cache = read_cache(cache_path)
    if not cache:
        sys.exit("[fatal] no v3 answers cached; run without --analyse-only")
    manifest["n_notes_answered"] = len(cache)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "manifest_v3.json").write_text(json.dumps(manifest, indent=2, default=str))
    analyse(per, cat, cache, out)


if __name__ == "__main__":
    main()
