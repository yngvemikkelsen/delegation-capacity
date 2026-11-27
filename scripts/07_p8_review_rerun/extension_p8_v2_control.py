# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
# ]
# ///
"""
extension_p8_v2_control.py -- CONTROL for the any-vasopressor check (Paper 24,
Frontiers, Reviewer 2, comment 1). Not part of the prespecified P8 analysis.

WHY
    The P8 run (October 2026) used Ollama 0.35.0. The v3 rerun uses Ollama
    0.40.15 with the same model weights (digest ad815644918f). A change in the
    runtime can change answers even at temperature 0, so a v3-vs-P8 difference
    could reflect the runtime as well as the question. This script asks the
    ORIGINAL P8 question again under the current runtime. Then:
        runtime effect   original P8 answers  vs  this rerun (same question)
        wording effect   this rerun           vs  v3 (same runtime)

WHAT IS ASKED
    Exactly the P8 prompt of extension_p8_error_demand_v2.py (all three
    questions unchanged), same model, decoding and notes.

INPUTS (read only)
    --p8-dir   ~/extension_results/p8                   p8_per_note_fact.csv
    --adj-dir  ~/extension_results/p8_adjudication      key/diag2_notes.csv
    --v3-dir   ~/extension_results/p8_vaso_any          llm_cache.jsonl (v3)
    --mimic-note  MIMIC-IV-Note 2.2 note folder          discharge.csv.gz

RUN (resumable). Default --scope A: the 1,699 vasopressor primary-set notes.
        uv run --offline extension_p8_v2_control.py
        uv run --offline extension_p8_v2_control.py --analyse-only
    To start it automatically when the v3 run has finished:
        caffeinate -i bash -c 'while pgrep -f "[e]xtension_p8_vaso_any_v3" \
            >/dev/null; do sleep 60; done; uv run --offline extension_p8_v2_control.py'
    (the [e] keeps pgrep from matching the waiting command itself)

OUTPUTS (in --out, default ~/extension_results/p8_v2_control); aggregates only
    analysis_plan_control.txt, prompt_sha.txt, manifest_control.json
    llm_cache.jsonl, control_summary.txt, control_rates.csv, control_models.csv
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
DEF_OUT = HOME / "extension_results" / "p8_v2_control"
DEF_V3 = HOME / "extension_results" / "p8_vaso_any"
DEF_OLLAMA = "http://localhost:11434"
DEF_MODEL = "qwen3:30b"
EXPECTED_DIGEST_PREFIX = "ad815644918f"

FACT_ORDER = ["vasopressor", "renal_replacement", "rbc_transfusion"]
QUESTIONS = {
    # all three identical to extension_p8_error_demand_v2.py
    "vasopressor": "Did the patient receive any intravenous norepinephrine, "
                   "epinephrine, phenylephrine, vasopressin or dopamine?",
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
P8 v2 CONTROL RERUN (REVIEW REQUEST) -- ORIGINAL QUESTION, CURRENT RUNTIME
(written before the first LLM call; do not edit after the run starts)

Prompt      Identical to extension_p8_error_demand_v2.py.
Notes       Block A: the 1,699 vasopressor primary-set notes of P8 (default).
Endpoints   (descriptive; reported as post hoc)
  C1  Runtime effect: agreement of this rerun with the original P8 answers,
      per fact; vasopressor primary-set miss rate, original vs rerun, overall
      and by quartile; OR per quartile (M1) for both.
  C2  Wording effect at equal runtime: v3 vs this rerun on the same notes;
      miss rate overall, by quartile and by vocabulary category; paired
      changes; M1 and M2 (q + category) for both.
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

def answers(cache: dict[str, dict], nid: str, fact: str):
    rec = cache.get(nid)
    if rec is None or rec.get("possibly_truncated"):
        return None
    a = rec["answers"].get(fact)
    return a if a in ("yes", "no") else None


def analyse(per: pd.DataFrame, cat: pd.DataFrame, ctl: dict[str, dict],
            v3: dict[str, dict], out: Path) -> None:
    L: list[str] = ["CONTROL RERUN -- original P8 question under the current runtime",
                    "Requested in review; not part of the prespecified P8 analysis.", ""]
    rates: list[dict] = []
    models: list[dict] = []
    per = per.copy()
    per["orig"] = per["llm_answer"].where(per["llm_answer"].isin(["yes", "no"])
                                          & ~per["possibly_truncated"])
    per["ctl"] = [answers(ctl, n, f) for n, f in zip(per["note_id"], per["fact"])]
    per["v3"] = [answers(v3, n, f) for n, f in zip(per["note_id"], per["fact"])]

    # C1 runtime effect
    L.append("C1  Runtime effect (same question): original P8 answers (Ollama 0.35.0) "
             "vs this rerun")
    for f in FACT_ORDER:
        g = per[(per["fact"] == f) & per["orig"].notna() & per["ctl"].notna()]
        if g.empty:
            continue
        same = (g["orig"] == g["ctl"]).mean()
        L.append(f"    {f:<18} notes {len(g):<5} identical answers {100 * same:.2f}%  "
                 f"(yes->no {int(((g['orig'] == 'yes') & (g['ctl'] == 'no')).sum())}, "
                 f"no->yes {int(((g['orig'] == 'no') & (g['ctl'] == 'yes')).sum())})")
    v = per[(per["fact"] == "vasopressor") & per["structured_present"]
            & per["text_mentioned"]].merge(cat, on="note_id", how="left")
    a = v[v["orig"].notna() & v["ctl"].notna()].copy()
    a["orig_miss"] = (a["orig"] == "no").astype(int)
    a["ctl_miss"] = (a["ctl"] == "no").astype(int)
    a["q1"] = a["q_num"] + 1
    L.append(f"    vasopressor primary set, notes with both answers: {len(a)}")
    for tag, y in (("original", "orig_miss"), ("rerun", "ctl_miss")):
        rr = rate_rows(a, y, f"C1 {tag}", "q1")
        rates += rr
        L.append(f"    {tag:<8} miss: " + "  ".join(
            f"{r['group']}: {r['events']}/{r['n']} {100 * r['rate']:.1f}%" for r in rr))
    models.append(fit_logit(a, "orig_miss", ["q_num"], "q_num", None,
                            "C1 M1 original miss ~ q (check 1.156)"))
    models.append(fit_logit(a, "ctl_miss", ["q_num"], "q_num", None,
                            "C1 M1 rerun miss ~ q"))
    L += ["    " + fmt_model(r) for r in models[-2:]]
    L.append("")

    # C2 wording effect at equal runtime
    b = v[v["ctl"].notna() & v["v3"].notna()].copy()
    b["ctl_miss"] = (b["ctl"] == "no").astype(int)
    b["v3_miss"] = (b["v3"] == "no").astype(int)
    b["q1"] = b["q_num"] + 1
    L.append(f"C2  Wording effect at equal runtime: v3 (any vasopressor) vs this rerun "
             f"(original question); notes with both answers: {len(b)}")
    for tag, y in (("rerun", "ctl_miss"), ("v3", "v3_miss")):
        rr = rate_rows(b, y, f"C2 {tag}", "q1")
        rates += rr
        L.append(f"    {tag:<8} miss: " + "  ".join(
            f"{r['group']}: {r['events']}/{r['n']} {100 * r['rate']:.1f}% "
            f"({100 * r['lo95']:.1f}-{100 * r['hi95']:.1f})" for r in rr))
    if len(b):
        t = pd.crosstab(b["ctl_miss"].map({1: "rerun miss", 0: "rerun hit"}),
                        b["v3_miss"].map({1: "v3 miss", 0: "v3 hit"}))
        L.append("    paired answers:\n" + "\n".join("      " + s for s in
                                                     t.to_string().splitlines()))
    L.append("    by vocabulary category (miss %, rerun -> v3)")
    for c in CATEGORIES:
        g = b[b["category"] == c]
        if g.empty:
            continue
        r2 = rate_rows(g, "ctl_miss", f"C2 {c} rerun")[0]
        r3 = rate_rows(g, "v3_miss", f"C2 {c} v3")[0]
        rates += [r2, r3]
        L.append(f"      {c:<14} n={r2['n']:<5} {100 * r2['rate']:5.1f} "
                 f"({100 * r2['lo95']:.1f}-{100 * r2['hi95']:.1f}) -> {100 * r3['rate']:5.1f} "
                 f"({100 * r3['lo95']:.1f}-{100 * r3['hi95']:.1f})")
    for tag, y in (("rerun", "ctl_miss"), ("v3", "v3_miss")):
        models.append(fit_logit(b, y, ["q_num"], "q_num", None, f"C2 M1 {tag} miss ~ q"))
        models.append(fit_logit(b.dropna(subset=["category"]), y, ["q_num"], "q_num",
                                "category", f"C2 M2 {tag} miss ~ q + category"))
    L += ["    " + fmt_model(r) for r in models[-4:]]

    (out / "control_summary.txt").write_text("\n".join(L) + "\n")
    pd.DataFrame(rates).to_csv(out / "control_rates.csv", index=False)
    pd.DataFrame(models).to_csv(out / "control_models.csv", index=False)
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
    ap.add_argument("--v3-dir", type=Path, default=DEF_V3)
    ap.add_argument("--mimic-note", type=Path, default=DEF_MIMIC_NOTE)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--ollama-url", default=DEF_OLLAMA)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scope", choices=["A", "AB", "ABC"], default="A")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    plan_path = out / "analysis_plan_control.txt"
    if plan_path.exists() and plan_path.read_text() != PLAN:
        sys.exit(f"[fatal] {plan_path} differs from the plan in this script; "
                 f"use a new --out.")
    plan_path.write_text(PLAN)
    check_prompt_stamp(out)
    mpath = args.p8_dir.expanduser() / "manifest.json"
    if mpath.exists():
        p8_sha = json.loads(mpath.read_text()).get("prompt_sha256_16")
        if p8_sha and p8_sha != prompt_sha():
            sys.exit(f"[fatal] prompt differs from the P8 run ({prompt_sha()} vs {p8_sha})")
        log(f"    prompt identical to the P8 run (sha {prompt_sha()})")
    else:
        log(f"[warning] {mpath} not found; prompt identity not checked")

    log("loading inputs")
    per = load_per(args.p8_dir.expanduser())
    cat = load_categories(args.adj_dir.expanduser())
    vaso = per[per["fact"] == "vasopressor"]
    blk_a = sorted(vaso.loc[vaso["structured_present"] & vaso["text_mentioned"], "note_id"])
    blk_b = sorted(vaso.loc[~vaso["structured_present"], "note_id"])
    blk_c = sorted(set(per["note_id"]) - set(blk_a) - set(blk_b))
    log(f"    block A {len(blk_a)}; block B {len(blk_b)}; block C {len(blk_c)}; "
        f"scope {args.scope}")

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
                     f"({EXPECTED_DIGEST_PREFIX})")
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

    ctl = read_cache(cache_path)
    if not ctl:
        sys.exit("[fatal] no control answers cached; run without --analyse-only")
    v3_cache = read_cache(args.v3_dir.expanduser() / "llm_cache.jsonl")
    log(f"\ncontrol answers {len(ctl)}; v3 answers {len(v3_cache)}")
    manifest["n_notes_answered"] = len(ctl)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "manifest_control.json").write_text(json.dumps(manifest, indent=2, default=str))
    analyse(per, cat, ctl, v3_cache, out)


if __name__ == "__main__":
    main()
