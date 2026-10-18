# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
#   "matplotlib>=3.8",
#   "torch>=2.2",
#   "transformers>=4.40",
# ]
# ///
"""
extension_p4_label_provenance.py -- test of P4 (retraining on accepted output).

PROPOSITION (specification P4)
    Label provenance and accepted-output contamination predict changes on a
    held-out, independent outcome set after actual retraining. Without label
    reuse the pathway is absent.

DESIGN (computational experiment on real data; deference is the only
simulated component)
    Delegate      the LLM abstraction from extension_p8_error_demand.py. Its
                  real answers, with their real error structure across
                  derangement strata, are the outputs a clinician reviews.
    Review        R14 simplified case (s = 1, r = 0): an erroneous LLM answer
                  is overridden to the reference with probability q;
                  otherwise it is accepted. Correct answers are accepted.
                  Two prespecified scenarios:
                    constant  q = 0.5 in every quartile
                    load      q = 0.8, 0.6, 0.4, 0.2 for quartiles 1-4
                              (same mean; verification falls as derangement
                              rises, R12)
    Provenance    each training label comes from the accepted-output channel
                  with probability omega, otherwise from the independent
                  reference (ICU event tables). omega in {0, .25, .5, .75, 1}.
                  Common random numbers across omega and scenarios.
    Retraining    a classifier per fact is trained on those labels: note
                  embedding (BGE-base-en-v1.5, CLS pooling, L2-normalised,
                  512-token chunks, mean and max over chunks), PCA fitted on
                  the training split, ridge logistic (statsmodels GLM,
                  elastic net with L1_wt = 0; intercept unpenalised).
    Test          held-out patients (subject-level split), independent
                  reference labels only. AUROC and Brier, overall and by
                  quartile.
    Replication   R repetitions of split, override and provenance draws.

ENDPOINTS (written to analysis_plan_p4.txt before the experiment)
    PRIMARY   per fact, constant scenario: change in test Brier score at
              omega = 1 versus omega = 0, paired within repetition; mean and
              2.5-97.5 percentile interval over repetitions. Brier is the
              primary because it is a proper score: contamination that shifts
              predicted risk without reordering cases leaves AUROC unchanged.
              supported     2.5th percentile > 0 (worse on the reference)
              contradicted  mean <= 0
              inconclusive  mean > 0, 2.5th percentile <= 0
    SECONDARY change in AUROC and in calibration-in-the-large (mean predicted
              minus observed); dose-response over omega; under the load
              scenario, whether the Brier loss is larger in quartile 4 than in
              quartile 1 compared with the constant scenario (difference in
              differences); realised training-label contamination by quartile.
    The intervals describe variation over the experiment's random draws, not
    sampling uncertainty about a population.

INPUTS
    --p8-dir   output folder of extension_p8_error_demand.py (needs
               p8_per_note_fact.csv)
    --mimic-note  to reload the note texts for embedding

USAGE
    uv run extension_p4_label_provenance.py
    uv run extension_p4_label_provenance.py --reps 100

OUTPUTS (in --out)
    analysis_plan_p4.txt, manifest_p4.json
    embeddings_<encoder>.npz       cached note embeddings
    p4_runs.csv                    one row per rep x fact x scenario x omega
    p4_contamination.csv           realised label contamination by quartile
    p4_summary.csv / p4_summary.txt
    p4_omega_effect.png            figure, dpi=140
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
DEF_P8 = HOME / "extension_results" / "p8"
DEF_OUT = HOME / "extension_results" / "p4"
DEF_MIMIC_NOTE = HOME / "physionet.org" / "files" / "mimic-iv-note" / "2.2" / "note"
DEF_ENCODER = "BAAI/bge-base-en-v1.5"
FACT_ORDER = ["vasopressor", "renal_replacement", "rbc_transfusion"]

PLAN = """\
P4 LABEL PROVENANCE -- PRESPECIFIED ANALYSIS PLAN
(written before the experiment; do not edit after the full run starts)

Delegate outputs : LLM answers from the P8 run (notes with a valid answer for
                   all three facts and not flagged as possibly truncated).
Reference        : ICU event tables (structured_present).
Review           : erroneous answer overridden with probability q (s=1, r=0).
Scenarios        : constant q = {qc}; load q by quartile = {ql}.
Omega grid       : {omega}
Classifier       : {encoder} embeddings (CLS, L2, 512-token chunks, mean+max),
                   PCA k = {pcs} on the training split, standardised, ridge
                   logistic alpha = {alpha} (intercept unpenalised).
Split            : by subject_id, test fraction {test_frac}.
Repetitions      : {reps}, seed {seed}.

PRIMARY  per fact, constant scenario: Brier(omega=1) - Brier(omega=0) on
         the independent test reference, paired within repetition; mean,
         2.5 and 97.5 percentiles.
         supported     2.5th percentile > 0
         contradicted  mean <= 0
         inconclusive  mean > 0 and 2.5th percentile <= 0
SECONDARY dAUROC; d calibration-in-the-large; dose-response over omega;
         load vs constant scenario, difference in differences of
         dBrier(omega=1) between quartile 4 and quartile 1; realised label
         contamination by quartile.
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
# data from the P8 run
# ---------------------------------------------------------------------------

def load_p8(p8_dir: Path) -> pd.DataFrame:
    path = p8_dir / "p8_per_note_fact.csv"
    if not path.exists():
        sys.exit(f"[fatal] {path} not found; run extension_p8_error_demand.py "
                 f"to completion first")
    per = pd.read_csv(path, dtype={"note_id": str}, low_memory=False)
    need = {"note_id", "subject_id", "hadm_id", "q_num", "fact",
            "structured_present", "llm_answer", "possibly_truncated"}
    miss = need - set(per.columns)
    if miss:
        sys.exit(f"[fatal] {path} lacks columns {sorted(miss)}")
    per["structured_present"] = per["structured_present"].astype(str).str.lower() == "true"
    per["possibly_truncated"] = per["possibly_truncated"].astype(str).str.lower() == "true"
    per["llm_answer"] = per["llm_answer"].astype(str).str.lower()
    per["valid"] = per["llm_answer"].isin(["yes", "no"]) & ~per["possibly_truncated"]
    ok_notes = per.groupby("note_id")["valid"].all()
    ok_notes = set(ok_notes[ok_notes].index)
    log(f"    P8 per-note table <- {path}")
    log(f"    notes: {per.note_id.nunique()}; usable (all three answers valid, "
        f"not truncated): {len(ok_notes)}")
    per = per[per["note_id"].isin(ok_notes)]
    wide = per.pivot_table(index=["note_id", "subject_id", "hadm_id", "q_num"],
                           columns="fact",
                           values=["structured_present", "llm_answer"],
                           aggfunc="first")
    out = pd.DataFrame(index=wide.index).reset_index()
    for f in FACT_ORDER:
        out[f"ref_{f}"] = wide[("structured_present", f)].astype(bool).values.astype(int)
        out[f"llm_{f}"] = (wide[("llm_answer", f)].values == "yes").astype(int)
    out["q_num"] = out["q_num"].astype(int)
    for f in FACT_ORDER:
        err = out[f"llm_{f}"] != out[f"ref_{f}"]
        log(f"    {f:<18} prevalence {out[f'ref_{f}'].mean():.3f}  "
            f"LLM error vs reference {err.mean():.3f}  by quartile "
            + " ".join(f"{err[out.q_num == q].mean():.3f}" for q in range(4)))
    return out


def load_texts(note_dir: Path, note_ids: set[str]) -> dict[str, str]:
    path = find_table(note_dir, "discharge")
    log(f"    discharge       <- {path}")
    texts: dict[str, str] = {}
    for chunk in pd.read_csv(path, usecols=["note_id", "text"],
                             dtype={"note_id": str}, chunksize=20_000,
                             low_memory=False):
        hit = chunk[chunk["note_id"].isin(note_ids)]
        texts.update(zip(hit["note_id"], hit["text"].fillna("").astype(str)))
    missing = note_ids - set(texts)
    if missing:
        sys.exit(f"[fatal] {len(missing)} note_ids not found in discharge table")
    return texts


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def pick_device(pref: str):
    import torch
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed(texts: dict[str, str], note_ids: list[str], encoder: str,
          device_pref: str, batch_size: int, cache: Path) -> np.ndarray:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["note_ids"].tolist(), range(len(z["note_ids"]))))
        if z["encoder"].item() == encoder and all(n in have for n in note_ids):
            log(f"    embeddings from cache {cache}")
            return z["X"][[have[n] for n in note_ids]]
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = pick_device(device_pref)
    log(f"    encoding {len(note_ids)} notes with {encoder} on {device}")
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        sys.exit(f"[fatal] {encoder} tokenizer has no CLS/SEP token")
    win = 510
    chunks, owner = [], []
    for i, nid in enumerate(note_ids):
        ids = tok(texts[nid], add_special_tokens=False,
                  truncation=False)["input_ids"]
        if not ids:
            ids = [pad_id]
        for s in range(0, len(ids), win):
            chunks.append([cls_id] + ids[s:s + win] + [sep_id])
            owner.append(i)
    log(f"    {len(chunks)} chunks ({len(chunks)/len(note_ids):.1f} per note)")
    order = np.argsort([len(c) for c in chunks])  # length-sorted batches
    vecs = [None] * len(chunks)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            L = max(len(chunks[j]) for j in idx)
            ii = torch.full((len(idx), L), pad_id, dtype=torch.long)
            am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):
                c = chunks[j]
                ii[r, :len(c)] = torch.tensor(c)
                am[r, :len(c)] = 1
            h = model(input_ids=ii.to(device),
                      attention_mask=am.to(device)).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h, p=2, dim=1).float().cpu().numpy()
            for r, j in enumerate(idx):
                vecs[j] = h[r]
            if (b // batch_size) % 50 == 0:
                log(f"      {b + len(idx)}/{len(chunks)} chunks "
                    f"({time.time() - t0:.0f} s)")
    V = np.vstack(vecs)
    owner = np.asarray(owner)
    d = V.shape[1]
    X = np.zeros((len(note_ids), 2 * d), dtype=np.float32)
    for i in range(len(note_ids)):
        v = V[owner == i]
        X[i, :d] = v.mean(axis=0)
        X[i, d:] = v.max(axis=0)
    np.savez_compressed(cache, X=X, note_ids=np.asarray(note_ids),
                        encoder=np.asarray(encoder))
    return X


# ---------------------------------------------------------------------------
# learner and metrics
# ---------------------------------------------------------------------------

def pca_fit(Xtr: np.ndarray, k: int):
    mu = Xtr.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    W = Vt[:k].T
    Ztr = (Xtr - mu) @ W
    sd = Ztr.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, W, sd


def pca_apply(X, mu, W, sd):
    return ((X - mu) @ W) / sd


def ridge_logit(Z: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xc = sm.add_constant(Z, has_constant="add")
    pen = np.full(Xc.shape[1], alpha)
    pen[0] = 0.0
    res = sm.GLM(y.astype(float), Xc, family=sm.families.Binomial()
                 ).fit_regularized(alpha=pen, L1_wt=0.0)
    return np.asarray(res.params)


def predict(params: np.ndarray, Z: np.ndarray) -> np.ndarray:
    eta = sm.add_constant(Z, has_constant="add") @ params
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, X: np.ndarray, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    omegas = [float(v) for v in args.omega.split(",")]
    q_load = np.array([float(v) for v in args.q_load.split(",")])
    if len(q_load) != 4:
        sys.exit("[fatal] --q-load needs four values (quartiles 1-4)")
    scen = {"constant": np.full(4, args.q_constant), "load": q_load}
    rng = np.random.default_rng(args.seed)
    subjects = df["subject_id"].unique()
    qn = df["q_num"].to_numpy()
    rows, crow = [], []
    t0 = time.time()
    for rep in range(args.reps):
        test_subj = set(rng.choice(subjects, size=int(round(len(subjects) * args.test_frac)),
                                   replace=False))
        te = df["subject_id"].isin(test_subj).to_numpy()
        tr = ~te
        mu, W, sd = pca_fit(X[tr], args.pcs)
        Ztr, Zte = pca_apply(X[tr], mu, W, sd), pca_apply(X[te], mu, W, sd)
        u_over = rng.random(tr.sum())
        u_prov = rng.random(tr.sum())
        for fact in FACT_ORDER:
            ref_tr = df[f"ref_{fact}"].to_numpy()[tr]
            llm_tr = df[f"llm_{fact}"].to_numpy()[tr]
            ref_te = df[f"ref_{fact}"].to_numpy()[te]
            if ref_tr.sum() < 10 or ref_te.sum() < 5:
                continue
            err = llm_tr != ref_tr
            for sname, qv in scen.items():
                q_tr = qv[qn[tr]]
                accepted = np.where(err & (u_over < q_tr), ref_tr, llm_tr)
                for om in omegas:
                    lab = np.where(u_prov < om, accepted, ref_tr)
                    if lab.min() == lab.max():
                        continue
                    params = ridge_logit(Ztr, lab, args.alpha)
                    p = predict(params, Zte)
                    rec = {"rep": rep, "fact": fact, "scenario": sname,
                           "omega": om, "n_train": int(tr.sum()),
                           "n_test": int(te.sum()),
                           "train_contam": float((lab != ref_tr).mean()),
                           "auroc": auroc(ref_te, p),
                           "brier": float(np.mean((p - ref_te) ** 2)),
                           "citl": float(p.mean() - ref_te.mean())}
                    for q in range(4):
                        m = qn[te] == q
                        rec[f"brier_q{q + 1}"] = (float(np.mean((p[m] - ref_te[m]) ** 2))
                                                  if m.any() else float("nan"))
                        rec[f"auroc_q{q + 1}"] = auroc(ref_te[m], p[m])
                    rows.append(rec)
                    for q in range(4):
                        m = qn[tr] == q
                        crow.append({"rep": rep, "fact": fact, "scenario": sname,
                                     "omega": om, "quartile": q + 1,
                                     "contamination": float((lab[m] != ref_tr[m]).mean())
                                     if m.any() else float("nan")})
        if rep in (0, 4) or (rep + 1) % 10 == 0:
            el = time.time() - t0
            log(f"    rep {rep + 1}/{args.reps}  {el / (rep + 1):.1f} s/rep  "
                f"ETA {el / (rep + 1) * (args.reps - rep - 1) / 60:.1f} min")
    return pd.DataFrame(rows), pd.DataFrame(crow)


def summarise(runs: pd.DataFrame, contam: pd.DataFrame, out: Path) -> None:
    base = runs[runs["omega"] == 0].set_index(["rep", "fact", "scenario"])
    mcols = (["brier", "auroc", "citl"] + [f"brier_q{q}" for q in range(1, 5)]
             + [f"auroc_q{q}" for q in range(1, 5)])
    d = runs.join(base[mcols], on=["rep", "fact", "scenario"], rsuffix="_0")
    for c in mcols:
        d[f"d_{c}"] = d[c] - d[f"{c}_0"]

    def pct(s: pd.Series, q: float) -> float:
        s = s.dropna()
        return float(np.percentile(s, q)) if len(s) else float("nan")

    srows = []
    for (fact, sc, om), g in d.groupby(["fact", "scenario", "omega"]):
        rec = {"fact": fact, "scenario": sc, "omega": om, "reps": len(g),
               "train_contam": g["train_contam"].mean(),
               "brier": g["brier"].mean(), "auroc": g["auroc"].mean()}
        for c in mcols:
            rec[f"d_{c}"] = g[f"d_{c}"].mean()
            rec[f"d_{c}_lo"] = pct(g[f"d_{c}"], 2.5)
            rec[f"d_{c}_hi"] = pct(g[f"d_{c}"], 97.5)
        srows.append(rec)
    summ = pd.DataFrame(srows)
    summ.to_csv(out / "p4_summary.csv", index=False)

    lines = ["PRIMARY (constant scenario, omega = 1 vs 0, test Brier score on "
             "the independent reference; positive = worse)", ""]
    for fact in FACT_ORDER:
        r = summ[(summ.fact == fact) & (summ.scenario == "constant")
                 & (summ.omega == 1.0)]
        if r.empty:
            lines.append(f"  {fact:<18} not estimable")
            continue
        r = r.iloc[0]
        if r["d_brier_lo"] > 0:
            v = "SUPPORTED"
        elif r["d_brier"] <= 0:
            v = "CONTRADICTED"
        else:
            v = "INCONCLUSIVE"
        lines.append(f"  {fact:<18} dBrier {r['d_brier']:+.5f} "
                     f"[{r['d_brier_lo']:+.5f}, {r['d_brier_hi']:+.5f}]  "
                     f"dAUROC {r['d_auroc']:+.4f} [{r['d_auroc_lo']:+.4f}, "
                     f"{r['d_auroc_hi']:+.4f}]  dCITL {r['d_citl']:+.4f}  "
                     f"label contamination {r['train_contam']:.3f}  -> {v}")
    for metric in ("brier", "auroc"):
        lines += ["", f"DOSE-RESPONSE (mean d{metric.upper()} by omega)", ""]
        piv = summ.pivot_table(index=["fact", "scenario"], columns="omega",
                               values=f"d_{metric}")
        lines.append(piv.to_string(float_format=lambda v: f"{v:+.5f}"))

    lines += ["", "LOAD vs CONSTANT: quartile 4 minus quartile 1 Brier loss at "
              "omega = 1", "(positive = larger Brier loss in quartile 4 under load)", ""]
    w = d[d["omega"] == 1.0].copy()
    w["q4_minus_q1"] = w["d_brier_q4"] - w["d_brier_q1"]
    for fact in FACT_ORDER:
        a = w[(w.fact == fact) & (w.scenario == "load")].set_index("rep")["q4_minus_q1"]
        b = w[(w.fact == fact) & (w.scenario == "constant")].set_index("rep")["q4_minus_q1"]
        did = (a - b).dropna()
        if len(did):
            lines.append(f"  {fact:<18} DiD {did.mean():+.5f} "
                         f"[{np.percentile(did, 2.5):+.5f}, "
                         f"{np.percentile(did, 97.5):+.5f}]  reps {len(did)}")
        else:
            lines.append(f"  {fact:<18} not estimable")

    lines += ["", "REALISED TRAINING-LABEL CONTAMINATION at omega = 1 "
              "(mean over reps)", ""]
    c = (contam[contam["omega"] == 1.0]
         .groupby(["fact", "scenario", "quartile"])["contamination"].mean()
         .unstack("quartile"))
    lines.append(c.to_string(float_format=lambda v: f"{v:.4f}"))
    (out / "p4_summary.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        log(l)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for (fact, sc), g in summ.groupby(["fact", "scenario"]):
        g = g.sort_values("omega")
        axes[0].errorbar(g["omega"], g["d_brier"],
                         yerr=[(g["d_brier"] - g["d_brier_lo"]).clip(lower=0),
                               (g["d_brier_hi"] - g["d_brier"]).clip(lower=0)],
                         marker="o", capsize=3,
                         linestyle="-" if sc == "constant" else "--",
                         label=f"{fact} ({sc})")
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_xlabel("omega (share of labels from accepted output)")
    axes[0].set_ylabel("change in test Brier vs omega = 0 (up = worse)")
    axes[0].set_title("Label provenance and retrained performance", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    s1 = summ[summ["omega"] == 1.0]
    xs = np.arange(1, 5)
    for (fact, sc), g in s1.groupby(["fact", "scenario"]):
        vals = [g[f"d_brier_q{q}"].iloc[0] for q in xs]
        axes[1].plot(xs, vals, marker="o",
                     linestyle="-" if sc == "constant" else "--",
                     label=f"{fact} ({sc})")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("derangement quartile (test patients)")
    axes[1].set_ylabel("change in Brier at omega = 1")
    axes[1].set_title("Where the loss falls", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "p4_omega_effect.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p8-dir", type=Path, default=DEF_P8)
    ap.add_argument("--mimic-note", type=Path, default=DEF_MIMIC_NOTE)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--encoder", default=DEF_ENCODER)
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pcs", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--omega", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--q-constant", type=float, default=0.5)
    ap.add_argument("--q-load", default="0.8,0.6,0.4,0.2")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    plan = PLAN.format(qc=args.q_constant, ql=args.q_load, omega=args.omega,
                       encoder=args.encoder, pcs=args.pcs, alpha=args.alpha,
                       test_frac=args.test_frac, reps=args.reps, seed=args.seed)
    plan_path = args.out / "analysis_plan_p4.txt"
    if plan_path.exists() and plan_path.read_text() != plan:
        sys.exit(f"[fatal] {plan_path} differs from this run's settings. "
                 f"The plan is frozen once written; use a new --out.")
    plan_path.write_text(plan)

    log("loading P8 delegate outputs")
    df = load_p8(args.p8_dir.expanduser())
    if len(df) < 200:
        sys.exit(f"[fatal] only {len(df)} usable notes; finish the P8 run first")
    note_ids = df["note_id"].tolist()
    texts = load_texts(args.mimic_note.expanduser(), set(note_ids))
    safe = args.encoder.replace("/", "__")
    X = embed(texts, note_ids, args.encoder, args.device, args.batch_size,
              args.out / f"embeddings_{safe}.npz")
    if args.pcs >= min(X.shape):
        sys.exit(f"[fatal] --pcs {args.pcs} too large for {X.shape}")

    log("\nrunning experiment")
    runs, contam = run(df, X, args)
    if runs.empty:
        sys.exit("[fatal] no estimable runs (too few positives per fact?)")
    runs.to_csv(args.out / "p4_runs.csv", index=False)
    contam.to_csv(args.out / "p4_contamination.csv", index=False)
    log("")
    summarise(runs, contam, args.out)

    import torch
    import transformers
    manifest = {
        "script": Path(__file__).name,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_sha256_16": hashlib.sha256(plan.encode()).hexdigest()[:16],
        "p8_dir": str(args.p8_dir), "n_notes": len(df),
        "n_subjects": int(df["subject_id"].nunique()),
        "encoder": args.encoder, "pandas": pd.__version__,
        "numpy": np.__version__, "statsmodels": statsmodels.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    (args.out / "manifest_p4.json").write_text(json.dumps(manifest, indent=2))
    log(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
