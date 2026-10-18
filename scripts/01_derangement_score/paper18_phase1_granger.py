#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scipy",
#     "statsmodels",
#     "pyarrow",
#     "matplotlib",
# ]
# ///
"""
Paper 18 Phase 1: Granger causality of within-patient recursive feedback
between S, D_v1, and O_v1, pooled across MICU stays (LOS >= 72h cohort).

Preprocessing decisions baked in (justified in v1 validation review)
====================================================================

1. STAY-START TRUNCATION
   Drop hours before all five core LOCF vitals are non-null:
   {hr_locf, map_locf, spo2_locf, gcs_locf, rr_locf}.
   Rationale: prevents O_v1 "phantom transitions" when first GCS/SpO2
   reading arrives, which would otherwise correlate spuriously with S
   (the admission-flurry chartevents that include those first readings).

2. LOS CAP AT 720 HOURS (30 days)
   Truncate per-stay analysis window to hour_idx in [T_warmup,
   T_warmup + 720). Rationale: VAR assumes stationarity; trajectories
   over weeks/months are non-stationary by clinical course (multiple
   regime shifts, sedation cycles, weaning trials) and don't isolate
   BCST P2 feedback. Capping preserves >99% of analyzable hours
   (median post-72h LOS is 138h, well below cap).

3. MIN-EFFECTIVE-LOS FILTER
   After preprocessing, drop stays with < 50 remaining hours.
   Rationale: VAR(6) with 3 endogenous variables has 21 params per
   equation; 50 obs gives ~2.5x parameters-to-obs ratio (loose lower
   bound). Below this, F-tests are unreliable.

4. ACUITY PROXY FROM FIRST 24h POST-WARMUP
   Not from per_stay_full stay-wide extremes (which leak future
   information — patient's worst vital over entire stay correlates
   with outcome). Computed from hourly_series LOCF values in the
   first 24h after warmup. Five vital components (HR/MAP/RR/SpO2/
   GCS extremes) scored by within-cohort quartile (0-3), plus age
   tertile (0-2) and vent_first_24h (0-1). Summed -> 0-18 range.
   Not calibrated against mortality — used ONLY for stratification.

Pipeline
========
Load   -> v1 outputs (hourly_series.parquet, cohort.parquet) + per_stay_full.csv
Preproc-> truncate, cap, filter, reindex within each stay
Acuity -> compute 0-18 proxy, derive within-cohort quartiles
Granger-> per-stay: log1p(S), ADF/diff if non-stationary, VAR(p) by BIC,
          test 6 directions (S<->O, D<->O, S<->D)
Pool   -> proportion p<0.05 per direction, binomial test vs null 0.05,
          median F + IQR, stratified by acuity quartile

Outputs (in --out-dir)
======================
paper18_phase1_per_stay.csv      one row per stay, all test stats
paper18_phase1_pooled.csv        6 directions x (overall + 4 acuity strata)
paper18_phase1_acuity.csv        per-stay acuity proxy + components
paper18_phase1_summary.txt       narrative interpretation + checklist
paper18_phase1_diagnostics.png   F-stat distributions, acuity calibration

Usage
=====
  uv run paper18_phase1_granger.py \\
      --hourly  ~/projects/BCST/paper18_results/paper18_hourly_series.parquet \\
      --cohort  ~/projects/BCST/paper18_results/paper18_cohort.parquet \\
      --per-stay ~/projects/BCST/bcst_mimic_p1_robustness_results/per_stay_full.csv \\
      --out-dir ~/projects/BCST/paper18_results

  # Smoke test with a sample:
  uv run paper18_phase1_granger.py ... --sample-n-stays 100

  # Tune the analysis window and warmup tolerance:
  uv run paper18_phase1_granger.py ... --analysis-window-h 480 --min-effective-h 72
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Quiet statsmodels' chatter on small-sample warnings (we report them ourselves)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Constants — preprocessing decisions baked in
# -----------------------------------------------------------------------------

CORE_LOCF_VITALS = ["hr_locf", "map_locf", "spo2_locf", "gcs_locf", "rr_locf"]
ANALYSIS_WINDOW_H = 720    # 30 days from warmup
MIN_EFFECTIVE_H = 50       # below this, VAR(6) is unreliable
MAX_LAG = 6                # transfer doc Section 8: max p=6
ACUITY_WINDOW_H = 24       # first 24h post-warmup for baseline severity

# Granger directions to test. Tuple = (label, caused_col, causing_col)
GRANGER_DIRECTIONS = [
    ("s_to_o", "O", "S"),
    ("o_to_s", "S", "O"),
    ("d_to_o", "O", "D"),
    ("o_to_d", "D", "O"),
    ("s_to_d", "D", "S"),
    ("d_to_s", "S", "D"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Stage 1: load + preprocess
# =============================================================================

def load_inputs(
    hourly_path: Path, cohort_path: Path, per_stay_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load v1 outputs + per_stay_full for vent_first_24h and anchor_age."""
    log(f"loading {hourly_path}")
    hourly = pd.read_parquet(hourly_path)
    log(f"  hourly: {len(hourly):,} rows, {hourly['stay_id'].nunique():,} stays")

    log(f"loading {cohort_path}")
    cohort = pd.read_parquet(cohort_path)
    log(f"  cohort: {len(cohort):,} stays")

    log(f"loading {per_stay_path} (subset of columns)")
    # Only the columns we need from per_stay
    ps_cols = ["stay_id", "anchor_age", "vent_first_24h", "anchor_year_group", "died_in_hosp"]
    ps = pd.read_csv(per_stay_path, usecols=ps_cols)
    log(f"  per_stay subset: {len(ps):,} rows")

    return hourly, cohort, ps


def preprocess_stay_truncate_and_cap(
    hourly: pd.DataFrame, attrition: list[str],
    analysis_window_h: int = ANALYSIS_WINDOW_H,
    min_effective_h: int = MIN_EFFECTIVE_H,
) -> pd.DataFrame:
    """Decisions 1-3: warmup-truncate, LOS-cap, min-effective-LOS filter.

    Adds `hour_eff` (0-indexed within the analysis window of each stay).
    """
    # Ensure required LOCF columns exist
    missing_locf = [c for c in CORE_LOCF_VITALS if c not in hourly.columns]
    if missing_locf:
        sys.exit(f"[fatal] hourly_series missing required LOCF columns: {missing_locf}")

    # Sort by stay/hour for deterministic groupby ops
    hourly = hourly.sort_values(["stay_id", "hour_idx"]).reset_index(drop=True)

    # T_warmup per stay = first hour_idx where ALL five core LOCF vitals are non-null
    all_present = hourly[CORE_LOCF_VITALS].notna().all(axis=1)
    hourly_with_flag = hourly.assign(_all_vitals=all_present)
    # First hour_idx where _all_vitals is True, per stay
    first_complete = (
        hourly_with_flag.loc[hourly_with_flag["_all_vitals"]]
                        .groupby("stay_id", sort=False)["hour_idx"]
                        .min()
                        .rename("t_warmup")
    )
    n_input_stays = hourly["stay_id"].nunique()
    n_with_warmup = len(first_complete)
    attrition.append(f"stays with a hour-with-all-5-LOCF-vitals: "
                     f"{n_with_warmup:,} / {n_input_stays:,} "
                     f"({n_with_warmup/n_input_stays:.1%})")

    # Drop stays that never reach a complete-vitals hour
    dropped_no_warmup = n_input_stays - n_with_warmup
    if dropped_no_warmup > 0:
        attrition.append(f"  -> dropped {dropped_no_warmup:,} stays "
                         f"(never reached all-5-vitals window)")

    hourly = hourly.merge(first_complete, on="stay_id", how="inner")

    # Apply Decision 1: drop hours before warmup
    n_before = len(hourly)
    hourly = hourly[hourly["hour_idx"] >= hourly["t_warmup"]].copy()
    n_dropped_warmup = n_before - len(hourly)
    attrition.append(f"Decision 1 (warmup truncation): dropped "
                     f"{n_dropped_warmup:,} hours below T_warmup")

    # Apply Decision 2: LOS cap
    hourly["hour_eff"] = hourly["hour_idx"] - hourly["t_warmup"]
    n_before = len(hourly)
    hourly = hourly[hourly["hour_eff"] < analysis_window_h].copy()
    n_dropped_cap = n_before - len(hourly)
    attrition.append(f"Decision 2 (LOS cap at {analysis_window_h}h): dropped "
                     f"{n_dropped_cap:,} hours above cap")

    # Per-stay effective length
    eff_lens = hourly.groupby("stay_id", sort=False).size().rename("n_eff")

    # Apply Decision 3: min-effective-LOS filter
    keep_stays = eff_lens[eff_lens >= min_effective_h].index
    n_before_stays = hourly["stay_id"].nunique()
    hourly = hourly[hourly["stay_id"].isin(keep_stays)].copy()
    n_after_stays = hourly["stay_id"].nunique()
    attrition.append(f"Decision 3 (min effective LOS >= {min_effective_h}h): "
                     f"{n_after_stays:,} / {n_before_stays:,} stays kept "
                     f"({n_after_stays/n_before_stays:.1%})")

    # Effective-LOS distribution post-filter
    eff_lens_kept = hourly.groupby("stay_id", sort=False).size()
    attrition.append(f"effective LOS post-preproc: min={eff_lens_kept.min()}, "
                     f"median={int(eff_lens_kept.median())}, "
                     f"mean={eff_lens_kept.mean():.1f}, "
                     f"max={eff_lens_kept.max()}, "
                     f"total patient-hours={len(hourly):,}")

    return hourly


# =============================================================================
# Stage 2: acuity proxy (Decision 4)
# =============================================================================

def compute_acuity_proxy(
    hourly: pd.DataFrame, per_stay: pd.DataFrame, attrition: list[str]
) -> pd.DataFrame:
    """First-24h-post-warmup baseline severity. Returns one row per stay.

    Score = sum of:
      - 5 vital components, each 0-3 by within-cohort quartile
        (worst quartile = 3)
      - age tertile, 0-2 (older = higher)
      - vent_first_24h, 0/1
    Range: 0-18.
    """
    # Restrict to first ACUITY_WINDOW_H effective hours per stay
    first_window = hourly[hourly["hour_eff"] < ACUITY_WINDOW_H].copy()

    # Define worst-case statistic per vital
    # (hr/rr extremes both directions; map/spo2/gcs low side primarily)
    vital_specs = [
        ("hr_locf", "extreme"),    # furthest from 80
        ("map_locf", "min"),       # lowest = worst
        ("rr_locf", "extreme"),    # furthest from 18
        ("spo2_locf", "min"),      # lowest = worst
        ("gcs_locf", "min"),       # lowest = worst
    ]

    rows = []
    for stay_id, g in first_window.groupby("stay_id", sort=False):
        row: dict = {"stay_id": stay_id, "n_acuity_hours": len(g)}
        for col, mode in vital_specs:
            x = g[col].dropna()
            if x.empty:
                row[f"acuity_{col}"] = np.nan
                continue
            if mode == "min":
                row[f"acuity_{col}"] = x.min()
            elif mode == "extreme":
                # Distance from clinical centre (HR 80, RR 18) — symmetric extremes
                centre = 80.0 if col == "hr_locf" else 18.0
                row[f"acuity_{col}"] = (x - centre).abs().max()
        rows.append(row)
    raw_components = pd.DataFrame(rows)

    # Convert each raw value to a 0-3 quartile score WITHIN cohort
    # (worst = 3, best = 0). For 'min' style, worst = LOW so reverse.
    score_cols = []
    for col, mode in vital_specs:
        raw_col = f"acuity_{col}"  # e.g. acuity_hr_locf
        score_col = f"score_{col.replace('_locf', '')}"  # e.g. score_hr
        score_cols.append(score_col)
        x = raw_components[raw_col]
        # qcut: 4 quantiles, labelled 0-3.
        try:
            ranks = pd.qcut(x, q=4, labels=False, duplicates="drop")
        except ValueError:
            ranks = pd.Series(np.nan, index=x.index)
        if mode == "min":
            # Reverse: lowest quartile = highest score
            ranks = 3 - ranks
        raw_components[score_col] = ranks.astype("Float64")

    # Merge per_stay covariates
    acuity = raw_components.merge(
        per_stay[["stay_id", "anchor_age", "vent_first_24h"]],
        on="stay_id", how="left",
    )

    # Age tertile (0-2)
    try:
        acuity["score_age"] = pd.qcut(
            acuity["anchor_age"], q=3, labels=False, duplicates="drop"
        ).astype("Float64")
    except ValueError:
        acuity["score_age"] = np.nan

    # Vent score
    acuity["score_vent"] = acuity["vent_first_24h"].fillna(0).astype("Float64")

    all_score_cols = score_cols + ["score_age", "score_vent"]
    acuity["acuity_proxy_24h"] = acuity[all_score_cols].sum(axis=1, min_count=len(all_score_cols))

    # Drop stays where the proxy couldn't be computed
    n_with_proxy = acuity["acuity_proxy_24h"].notna().sum()
    attrition.append(f"acuity proxy computed for {n_with_proxy:,} / "
                     f"{len(acuity):,} stays")

    # Quartile of the proxy itself
    try:
        acuity["acuity_quartile"] = pd.qcut(
            acuity["acuity_proxy_24h"], q=4, labels=[1, 2, 3, 4], duplicates="drop"
        )
    except ValueError:
        acuity["acuity_quartile"] = pd.NA

    return acuity


def _qcut_safe(x: pd.Series, q: int) -> pd.Series:
    """qcut that falls back to NaN if duplicates prevent splitting."""
    try:
        return pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=x.index)


# =============================================================================
# Stage 3: per-stay Granger
# =============================================================================

@dataclass
class GrangerResult:
    stay_id: int
    n_obs: int
    status: str
    p_lag: int = 0
    adf_s_p: float = np.nan
    adf_d_p: float = np.nan
    adf_o_p: float = np.nan
    diff_s: bool = False
    diff_d: bool = False
    diff_o: bool = False
    # 6 directions x (F, p)
    f_s_to_o: float = np.nan; p_s_to_o: float = np.nan
    f_o_to_s: float = np.nan; p_o_to_s: float = np.nan
    f_d_to_o: float = np.nan; p_d_to_o: float = np.nan
    f_o_to_d: float = np.nan; p_o_to_d: float = np.nan
    f_s_to_d: float = np.nan; p_s_to_d: float = np.nan
    f_d_to_s: float = np.nan; p_d_to_s: float = np.nan


def analyse_one_stay(args: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """Worker function — must be top-level and pickle-safe for multiprocessing."""
    # Import here so workers don't crash on import order
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.api import VAR

    stay_id, s_raw, d_raw, o_raw = args
    n0 = len(s_raw)

    # log1p(S) — S is right-skewed counts
    s = np.log1p(s_raw.astype(float))
    d = d_raw.astype(float)
    o = o_raw.astype(float)

    def adf_p(x: np.ndarray) -> float:
        # ADF requires non-constant
        if np.std(x) < 1e-12 or len(x) < 12:
            return np.nan
        try:
            return adfuller(x, autolag="AIC")[1]
        except Exception:
            return np.nan

    adf_s_p, adf_d_p, adf_o_p = adf_p(s), adf_p(d), adf_p(o)
    diff_s = (not np.isnan(adf_s_p)) and adf_s_p > 0.05
    diff_d = (not np.isnan(adf_d_p)) and adf_d_p > 0.05
    diff_o = (not np.isnan(adf_o_p)) and adf_o_p > 0.05

    if diff_s: s = np.diff(s)
    if diff_d: d = np.diff(d)
    if diff_o: o = np.diff(o)

    # Align lengths if some were differenced and others not
    n = min(len(s), len(d), len(o))
    s, d, o = s[-n:], d[-n:], o[-n:]

    result = dict(
        stay_id=int(stay_id), n_obs=int(n),
        adf_s_p=adf_s_p, adf_d_p=adf_d_p, adf_o_p=adf_o_p,
        diff_s=diff_s, diff_d=diff_d, diff_o=diff_o,
        p_lag=0, status="ok",
    )

    if n < 20 or np.std(s) < 1e-10 or np.std(d) < 1e-10 or np.std(o) < 1e-10:
        result["status"] = "insufficient_or_constant"
        return result

    Y = pd.DataFrame({"S": s, "D": d, "O": o})
    try:
        model = VAR(Y)
        # Cap maxlags so VAR has degrees of freedom: need n - p*k > 0 with margin
        max_p = min(MAX_LAG, max(1, (n - 10) // 4))
        order = model.select_order(maxlags=max_p)
        p_lag = int(order.bic)
        if p_lag == 0:
            result["status"] = "lag_0"
            return result
        result["p_lag"] = p_lag
        fitted = model.fit(p_lag)

        for name, caused, causing in GRANGER_DIRECTIONS:
            try:
                t = fitted.test_causality(caused=caused, causing=[causing], kind="f")
                result[f"f_{name}"] = float(t.test_statistic)
                result[f"p_{name}"] = float(t.pvalue)
            except Exception:
                result[f"f_{name}"] = np.nan
                result[f"p_{name}"] = np.nan
        return result
    except Exception as e:
        result["status"] = f"var_fail:{type(e).__name__}"
        return result


def run_granger_parallel(hourly: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    """Per-stay Granger analysis, parallel across stays."""
    # Build inputs as tuples (small, pickle-cheap)
    log("packing per-stay arrays...")
    inputs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for stay_id, g in hourly.groupby("stay_id", sort=False):
        g_sorted = g.sort_values("hour_eff")
        inputs.append((
            int(stay_id),
            g_sorted["S"].to_numpy(),
            g_sorted["D_v1"].to_numpy(),
            g_sorted["O_v1"].to_numpy(),
        ))
    log(f"  packed {len(inputs):,} stays; launching pool with {n_workers} workers")

    t0 = time.time()
    results: list[dict] = []
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(analyse_one_stay, inputs, chunksize=20), 1):
                results.append(r)
                if i % max(1, len(inputs) // 20) == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(inputs) - i) / rate
                    log(f"  {i:,} / {len(inputs):,} done, "
                        f"rate {rate:.1f}/s, eta {eta:.0f}s")
    else:
        for i, item in enumerate(inputs, 1):
            results.append(analyse_one_stay(item))
            if i % max(1, len(inputs) // 20) == 0:
                log(f"  {i:,} / {len(inputs):,} done (serial)")

    log(f"Granger fits done in {time.time()-t0:.0f}s")
    return pd.DataFrame(results)


# =============================================================================
# Stage 4: pooled analysis
# =============================================================================

def pool_granger(per_stay_results: pd.DataFrame, acuity: pd.DataFrame) -> pd.DataFrame:
    """Per direction × (overall, acuity quartile): n, prop p<0.05, binomial p,
    median F (with IQR)."""
    from scipy.stats import binom

    # Only OK stays count toward pooling
    ok = per_stay_results[per_stay_results["status"] == "ok"].copy()
    ok = ok.merge(
        acuity[["stay_id", "acuity_quartile"]], on="stay_id", how="left"
    )

    rows = []
    strata = [("overall", ok)]
    for q in [1, 2, 3, 4]:
        strata.append((f"acuity_q{q}", ok[ok["acuity_quartile"] == q]))

    for direction, _, _ in GRANGER_DIRECTIONS:
        for stratum_label, stratum in strata:
            f_col = f"f_{direction}"
            p_col = f"p_{direction}"
            valid = stratum[stratum[p_col].notna()]
            n = len(valid)
            if n == 0:
                rows.append({"direction": direction, "stratum": stratum_label,
                             "n": 0, "n_sig": 0, "prop_sig": np.nan,
                             "binom_p": np.nan, "f_median": np.nan,
                             "f_iqr_lo": np.nan, "f_iqr_hi": np.nan})
                continue
            n_sig = int((valid[p_col] < 0.05).sum())
            prop_sig = n_sig / n
            # One-sided binomial: P(X >= n_sig | p=0.05)
            binom_p = float(binom.sf(n_sig - 1, n, 0.05))
            f_q = valid[f_col].quantile([0.25, 0.5, 0.75])
            rows.append({
                "direction": direction,
                "stratum": stratum_label,
                "n": n,
                "n_sig": n_sig,
                "prop_sig": prop_sig,
                "binom_p": binom_p,
                "f_median": float(f_q.iloc[1]),
                "f_iqr_lo": float(f_q.iloc[0]),
                "f_iqr_hi": float(f_q.iloc[2]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Stage 5: reporting
# =============================================================================

def write_summary(
    out_dir: Path,
    attrition: list[str],
    per_stay_results: pd.DataFrame,
    pooled: pd.DataFrame,
    runtime_s: float,
) -> None:
    lines: list[str] = []
    lines.append("Paper 18 Phase 1 — Granger causality summary")
    lines.append("=" * 70)
    lines.append(f"runtime: {runtime_s:.1f}s")
    lines.append("")

    lines.append("## 1. Preprocessing attrition")
    lines.extend("  " + a for a in attrition)
    lines.append("")

    lines.append("## 2. Per-stay analysis status")
    status_counts = per_stay_results["status"].value_counts()
    for status, n in status_counts.items():
        lines.append(f"  {status}: {n:,} ({n/len(per_stay_results):.1%})")
    lines.append("")

    ok = per_stay_results[per_stay_results["status"] == "ok"]
    if not ok.empty:
        lines.append("## 3. Lag order selection (BIC, max 6)")
        lag_counts = ok["p_lag"].value_counts().sort_index()
        for lag, n in lag_counts.items():
            lines.append(f"  p={lag}: {n:,} stays ({n/len(ok):.1%})")
        lines.append("")

        lines.append("## 4. Differencing for stationarity")
        for col in ["diff_s", "diff_d", "diff_o"]:
            prop = ok[col].mean()
            lines.append(f"  {col} (ADF p>0.05): {prop:.1%} of stays")
        lines.append("")

    lines.append("## 5. Pooled Granger results (significance at p<0.05)")
    lines.append("")
    lines.append("  direction  | stratum     |     n | n_sig | prop_sig | binom_p   | F median (IQR)")
    lines.append("  -----------|-------------|-------|-------|----------|-----------|---------------")
    for _, r in pooled.iterrows():
        if pd.isna(r["prop_sig"]):
            continue
        lines.append(
            f"  {r['direction']:<10} | {r['stratum']:<11} | "
            f"{r['n']:>5,} | {r['n_sig']:>5,} | {r['prop_sig']:>7.1%} | "
            f"{r['binom_p']:>.2e} | "
            f"{r['f_median']:>5.2f} ({r['f_iqr_lo']:.2f}-{r['f_iqr_hi']:.2f})"
        )
    lines.append("")

    lines.append("## 6. Interpretation (substantive detection criteria)")
    overall = pooled[pooled["stratum"] == "overall"]
    # Substantive detection requires BOTH:
    #   prop_sig > 0.10  (>= 2x null rate, real effect size)
    #   binom_p  < 0.001 (not chance at this N)
    # At large N a 1pp excess produces binom_p << 0.001 by itself;
    # without the effect-size floor, every direction is "detected".
    EFFSIZE_FLOOR = 0.10
    PVAL_FLOOR = 0.001
    lines.append(f"  Thresholds: prop_sig > {EFFSIZE_FLOOR:.0%} AND binom_p < {PVAL_FLOOR}")
    lines.append("")

    sig_directions = overall[
        (overall["binom_p"] < PVAL_FLOOR) &
        (overall["prop_sig"] > EFFSIZE_FLOOR)
    ]
    weak_directions = overall[
        (overall["binom_p"] < PVAL_FLOOR) &
        (overall["prop_sig"] <= EFFSIZE_FLOOR) &
        (overall["prop_sig"] > 0.05)
    ]
    null_directions = overall[overall["binom_p"] >= PVAL_FLOOR]

    lines.append(f"  Substantively detected: {len(sig_directions)} / 6")
    if len(sig_directions) > 0:
        for _, r in sig_directions.iterrows():
            lines.append(f"    {r['direction']:<10}  prop_sig={r['prop_sig']:.1%}  "
                         f"F_median={r['f_median']:.2f}")
    lines.append("")
    lines.append(f"  Statistically distinguishable but small (binom p<{PVAL_FLOOR} "
                 f"but prop_sig <= {EFFSIZE_FLOOR:.0%}): {len(weak_directions)}")
    if len(weak_directions) > 0:
        for _, r in weak_directions.iterrows():
            lines.append(f"    {r['direction']:<10}  prop_sig={r['prop_sig']:.1%}  "
                         f"(excess over null: {r['prop_sig']-0.05:+.1%})")
    lines.append("")
    lines.append(f"  At-or-below null: {len(null_directions)}")
    if len(null_directions) > 0:
        lines.append(f"    {', '.join(null_directions['direction'].tolist())}")
    lines.append("")

    # Bidirectional check uses substantive criterion only
    s_o_pair = {"s_to_o", "o_to_s"}
    d_o_pair = {"d_to_o", "o_to_d"}
    s_d_pair = {"s_to_d", "d_to_s"}
    detected = set(sig_directions["direction"].tolist())
    lines.append(f"  Substantive bidirectional S<->O: {s_o_pair.issubset(detected)}")
    lines.append(f"  Substantive bidirectional D<->O: {d_o_pair.issubset(detected)}")
    lines.append(f"  Substantive bidirectional S<->D: {s_d_pair.issubset(detected)}")
    lines.append("")

    lines.append("## 7. Caveats")
    lines.append("  - Granger causality is predictive, not strictly causal.")
    lines.append("  - S, D_v1, O_v1 are constructed measures with mechanical")
    lines.append("    contemporaneous correlations. VAR Granger tests LAGGED")
    lines.append("    relationships, so contemporaneous correlation does not")
    lines.append("    inflate the F-statistics directly. However, common time-")
    lines.append("    varying confounders (sedation cycles, day/night patterns)")
    lines.append("    can drive both S and O in lagged fashion — not tested.")
    lines.append("    Phase 3 (G-methods MSM) addresses this if pursued.")
    lines.append("  - At N this large, even small effects yield tiny binomial")
    lines.append("    p-values. Effect SIZE (prop_sig minus 0.05) and stratum")
    lines.append("    consistency matter more than the binomial p itself.")
    lines.append("  - Stays with status != 'ok' are excluded from pooling but")
    lines.append("    listed in per_stay CSV for inspection.")
    lines.append("")

    lines.append("## 8. Next step")
    lines.append("  Review pooled.csv §5 table. If S<->O bidirectional with")
    lines.append("  effect-modification across acuity quartiles is detected,")
    lines.append("  proceed to Phase 2 (cross-lagged multilevel models) to")
    lines.append("  estimate signed coefficients. Otherwise, the Phase 1 null")
    lines.append("  result is itself the paper.")

    txt_path = out_dir / "paper18_phase1_summary.txt"
    txt_path.write_text("\n".join(lines))
    log(f"summary: wrote {txt_path}")


def write_diagnostics(
    out_dir: Path, per_stay_results: pd.DataFrame, acuity: pd.DataFrame
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("diagnostics: matplotlib not available, skipping figure")
        return

    ok = per_stay_results[per_stay_results["status"] == "ok"]
    if ok.empty:
        log("diagnostics: no OK stays, skipping figure")
        return

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=140)
    for ax, (direction, _, _) in zip(axes.flatten()[:6], GRANGER_DIRECTIONS):
        f_col = f"f_{direction}"
        p_col = f"p_{direction}"
        f = ok[f_col].dropna()
        if len(f) == 0:
            continue
        ax.hist(np.log1p(f), bins=50)
        n_sig = (ok[p_col] < 0.05).sum()
        ax.set_title(f"{direction}  (n_sig/n = {n_sig:,}/{len(f):,} "
                     f"= {n_sig/len(f):.1%})")
        ax.set_xlabel("log1p(F-statistic)")

    # Acuity histogram
    if "acuity_proxy_24h" in acuity.columns:
        axes[2, 0].hist(acuity["acuity_proxy_24h"].dropna(), bins=30)
        axes[2, 0].set_title("acuity_proxy_24h distribution")
    # Lag order
    axes[2, 1].hist(ok["p_lag"], bins=np.arange(0, MAX_LAG + 2) - 0.5)
    axes[2, 1].set_title(f"VAR lag (BIC, max {MAX_LAG})")
    axes[2, 1].set_xlabel("selected p")
    # n_obs distribution
    axes[2, 2].hist(ok["n_obs"], bins=40)
    axes[2, 2].set_title("effective n per stay")
    axes[2, 2].set_xlabel("hours")

    fig.tight_layout()
    out = out_dir / "paper18_phase1_diagnostics.png"
    fig.savefig(out)
    log(f"diagnostics: wrote {out}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 18 Phase 1 Granger analysis")
    ap.add_argument("--hourly", type=Path, required=True,
                    help="paper18_hourly_series.parquet (from v1)")
    ap.add_argument("--cohort", type=Path, required=True,
                    help="paper18_cohort.parquet (from v1)")
    ap.add_argument("--per-stay", type=Path, required=True,
                    help="per_stay_full.csv (for anchor_age, vent_first_24h)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--sample-n-stays", type=int, default=None,
                    help="Random-sample stays before analysis (testing)")
    ap.add_argument("--analysis-window-h", type=int, default=ANALYSIS_WINDOW_H,
                    help=f"Override LOS cap (default {ANALYSIS_WINDOW_H})")
    ap.add_argument("--min-effective-h", type=int, default=MIN_EFFECTIVE_H,
                    help=f"Override min effective LOS (default {MIN_EFFECTIVE_H})")
    ap.add_argument("--use-o-strict", action="store_true",
                    help="Recompute O on RAW (not LOCF) vitals — sensitivity. "
                         "Replaces O_v1 column before Granger analysis. "
                         "Tests whether D->O survives without LOCF persistence.")
    args = ap.parse_args()

    log(f"config: analysis_window_h={args.analysis_window_h}, "
        f"min_effective_h={args.min_effective_h}, max_lag={MAX_LAG}")

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attrition: list[str] = []

    # 1. Load
    hourly, cohort, per_stay = load_inputs(args.hourly, args.cohort, args.per_stay)
    attrition.append(f"input hourly: {len(hourly):,} rows, "
                     f"{hourly['stay_id'].nunique():,} stays")

    if args.sample_n_stays:
        rng = np.random.default_rng(17)
        keep = rng.choice(
            hourly["stay_id"].unique(), size=min(args.sample_n_stays, hourly["stay_id"].nunique()),
            replace=False,
        )
        hourly = hourly[hourly["stay_id"].isin(keep)].copy()
        attrition.append(f"--sample-n-stays={args.sample_n_stays} applied -> "
                         f"{hourly['stay_id'].nunique():,} stays")

    # --use-o-strict: recompute O from RAW vitals (no LOCF persistence)
    if args.use_o_strict:
        log("computing O_strict from RAW vitals (sensitivity)")
        required_raw = ["map", "spo2", "gcs"]
        missing_raw = [c for c in required_raw if c not in hourly.columns]
        if missing_raw:
            sys.exit(f"[fatal] --use-o-strict needs raw vital columns; "
                     f"missing: {missing_raw}")
        o_strict = np.zeros(len(hourly), dtype="int64")
        map_low = (hourly["map"] < 70).fillna(False).to_numpy()
        spo2_low = (hourly["spo2"] < 92).fillna(False).to_numpy()
        gcs = hourly["gcs"].to_numpy()
        cns = np.where(
            np.isnan(gcs), 0,
            np.where(gcs < 6, 3, np.where(gcs < 10, 2, np.where(gcs < 13, 1, 0))),
        ).astype("int64")
        o_strict = map_low.astype("int64") + spo2_low.astype("int64") + cns
        hourly["O_v1_locf_original"] = hourly["O_v1"]
        hourly["O_v1"] = o_strict
        attrition.append(f"--use-o-strict: O_v1 replaced by O_strict "
                         f"(mean orig={hourly['O_v1_locf_original'].mean():.3f}, "
                         f"mean strict={hourly['O_v1'].mean():.3f})")

    # 2. Preprocess (Decisions 1-3)
    hourly = preprocess_stay_truncate_and_cap(
        hourly, attrition,
        analysis_window_h=args.analysis_window_h,
        min_effective_h=args.min_effective_h,
    )

    # 3. Acuity proxy (Decision 4)
    acuity = compute_acuity_proxy(hourly, per_stay, attrition)
    acuity_out = args.out_dir / "paper18_phase1_acuity.csv"
    acuity.to_csv(acuity_out, index=False)
    log(f"acuity: wrote {acuity_out} ({len(acuity):,} stays)")

    # 4. Per-stay Granger
    per_stay_results = run_granger_parallel(hourly, args.n_workers)
    per_stay_out = args.out_dir / "paper18_phase1_per_stay.csv"
    per_stay_results.to_csv(per_stay_out, index=False)
    log(f"per_stay: wrote {per_stay_out} ({len(per_stay_results):,} rows)")

    # 5. Pool
    pooled = pool_granger(per_stay_results, acuity)
    pooled_out = args.out_dir / "paper18_phase1_pooled.csv"
    pooled.to_csv(pooled_out, index=False)
    log(f"pooled: wrote {pooled_out} ({len(pooled):,} rows)")

    # 6. Summary + diagnostics
    runtime_s = time.time() - t0
    write_summary(args.out_dir, attrition, per_stay_results, pooled, runtime_s)
    write_diagnostics(args.out_dir, per_stay_results, acuity)

    log("=" * 70)
    log(f"DONE in {runtime_s:.1f}s — REVIEW paper18_phase1_summary.txt")


if __name__ == "__main__":
    main()
