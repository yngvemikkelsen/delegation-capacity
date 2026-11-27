# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "statsmodels>=0.14",
#   "matplotlib>=3.8",
#   "pillow>=8",
# ]
# ///
"""
extension_figures_v2.py -- revised figures for Paper 24 (Frontiers revision,
Reviewer 1, minor comment 1).

Figure 1 (P8)  Observed miss rates by disturbance quartile (symbols, Wilson
               95% CIs) with fitted trend lines from the logistic models and
               the odds ratios printed in each panel:
                 pooled line  marginal prediction of the pooled model
                              (fact indicators): the model's predicted
                              probability, averaged over the panel's note-fact
                              pairs, with the quartile set to 1, 2, 3, 4
                 fact lines   panel A only: one logistic model per fact (S5)
               Models are refitted here from p8_per_note_fact.csv exactly as in
               extension_p8_error_demand_v2.py (logistic GLM, cluster-robust SE
               by subject); the printed ORs are checked against p8_models.csv.
Figure 2 (P4)  As before (A: change in Brier score by omega; B: by quartile at
               omega = 1) with a plain-language key, plus a new panel C: change
               in calibration-in-the-large (mean predicted probability minus
               observed rate, percentage points), which shows what the Brier
               loss means in practice (under-calling).
Axis labels use "Disturbance quartile" (manuscript wording since v18).
No analysis is changed; every plotted value is written to figure_values_v2.csv.

INPUTS
    --p8-dir  ~/extension_results/p8   p8_per_note_fact.csv, p8_rates_by_stratum.csv,
                                       p8_models.csv
    --p4-dir  ~/extension_results/p4   p4_summary.csv
OUTPUTS (in --out, default ~/extension_results/figures_v2)
    Figure1_P8_error_by_quartile_v2.tif / .png   180 mm wide, 300 dpi, RGB
    Figure2_P4_retraining_v2.tif / .png
    figure_values_v2.csv, figure_models_v2.txt
USAGE
    uv run --offline extension_figures_v2.py          (under 1 min)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image

HOME = Path.home()
DEF_P8 = HOME / "extension_results" / "p8"
DEF_P4 = HOME / "extension_results" / "p4"
DEF_OUT = HOME / "extension_results" / "figures_v2"

FACTS = ["vasopressor", "renal_replacement", "rbc_transfusion"]
LABEL = {"vasopressor": "Vasopressor", "renal_replacement": "Renal replacement therapy",
         "rbc_transfusion": "Red-cell transfusion", "POOLED": "All three facts"}
SHORT = {"vasopressor": "Vasopressor", "renal_replacement": "RRT",
         "rbc_transfusion": "Transfusion", "POOLED": "Pooled"}
COLOR = {"vasopressor": "#0072B2", "renal_replacement": "#D55E00",
         "rbc_transfusion": "#009E73", "POOLED": "#000000"}
MARKER = {"vasopressor": "o", "renal_replacement": "^", "rbc_transfusion": "D",
          "POOLED": "s"}
# (set key in p8_rates_by_stratum.csv, panel title, y label, outcome, model key)
P8_SETS = [
    ("PRIMARY documented positives", "A  In record and note", "Answered ‘no’ (miss)",
     "miss", "PRIMARY"),
    ("S1 all structured positives", "B  In record", "Answered ‘no’ (miss)",
     "miss", "S1"),
    ("S2 unsupported positives", "C  Not in record or note", "Answered ‘yes’",
     "false_yes", "S2"),
]
MM = 1 / 25.4


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "savefig.dpi": 300,
    })


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)
    with Image.open(out / f"{stem}.png") as im:
        rgb = im.convert("RGB")
        rgb.save(out / f"{stem}.png", dpi=(300, 300))
        rgb.save(out / f"{stem}.tif", dpi=(300, 300), compression="tiff_lzw")
        w, h = rgb.size
    print(f"    {out / stem}.tif / .png  {w}x{h} px = {w / 300 * 25.4:.0f} x "
          f"{h / 300 * 25.4:.0f} mm")


def need(path: Path, cols: set[str]) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[fatal] {path} not found")
    df = pd.read_csv(path, dtype={"note_id": str} if "note_id" in cols else None,
                     low_memory=False)
    miss = cols - set(df.columns)
    if miss:
        sys.exit(f"[fatal] {path} lacks columns {sorted(miss)}")
    return df


def fmt_or(r: dict) -> str:
    return f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})"


# ---------------------------------------------------------------------------
# P8 models, refitted as in extension_p8_error_demand_v2.py
# ---------------------------------------------------------------------------

def p8_sets(per: pd.DataFrame) -> dict[str, pd.DataFrame]:
    for c in ("structured_present", "text_mentioned", "possibly_truncated"):
        per[c] = per[c].astype(str).str.lower() == "true"
    per["llm_answer"] = per["llm_answer"].astype(str).str.lower()
    per["excluded"] = ~per["llm_answer"].isin(["yes", "no"]) | per["possibly_truncated"]
    per["miss"] = (per["llm_answer"] == "no").astype(int)
    per["false_yes"] = (per["llm_answer"] == "yes").astype(int)
    ok = per[~per["excluded"]]
    return {"PRIMARY": ok[ok["structured_present"] & ok["text_mentioned"]],
            "S1": ok[ok["structured_present"]],
            "S2": ok[~ok["structured_present"] & ~ok["text_mentioned"]]}


def fit(df: pd.DataFrame, y: str, fact_fe: bool) -> dict:
    X = df[["q_num"]].astype(float).copy()
    if fact_fe and df["fact"].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(df["fact"], prefix="fact", drop_first=True,
                                         dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["subject_id"].values})
    b = float(res.params["q_num"])
    lo, hi = (float(v) for v in res.conf_int().loc["q_num"].values)
    # fitted trend: average predicted probability with q_num set to 0..3
    curve = []
    for q in range(4):
        Xq = X.copy()
        Xq["q_num"] = float(q)
        curve.append(float(np.mean(res.predict(Xq))))
    return {"OR": math.exp(b), "lo": math.exp(lo), "hi": math.exp(hi),
            "p": float(res.pvalues["q_num"]), "n": len(df), "events": int(df[y].sum()),
            "curve": curve}


def figure1(p8_dir: Path, out: Path, values: list[dict], mlog: list[str]) -> None:
    rates = need(p8_dir / "p8_rates_by_stratum.csv",
                 {"set", "fact", "q_num", "n", "events", "rate", "lo95", "hi95"})
    per = need(p8_dir / "p8_per_note_fact.csv",
               {"note_id", "subject_id", "q_num", "fact", "structured_present",
                "text_mentioned", "llm_answer", "possibly_truncated"})
    sets = p8_sets(per)
    ref = None
    mpath = p8_dir / "p8_models.csv"
    if mpath.exists():
        ref = pd.read_csv(mpath)

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 82 * MM))
    offs = {"vasopressor": -0.12, "renal_replacement": -0.04,
            "rbc_transfusion": 0.04, "POOLED": 0.12}
    for ax, (key, title, ylab, y, mkey) in zip(axes, P8_SETS):
        sub = rates[rates["set"] == key]
        if sub.empty:
            sys.exit(f"[fatal] set '{key}' not in p8_rates_by_stratum.csv")
        d = sets[mkey]
        pooled = fit(d, y, True)
        lines = {"POOLED": pooled}
        if mkey == "PRIMARY":
            for f in FACTS:
                lines[f] = fit(d[d["fact"] == f], y, False)
        # check against the P8 run
        for k, r in lines.items():
            name = {"PRIMARY": "PRIMARY", "S1": "S1", "S2": "S2"}[mkey] if k == "POOLED" \
                else f"S5 {k}"
            chk = ""
            if ref is not None and "model" in ref.columns:
                m = ref[ref["model"].astype(str).str.startswith(name)]
                if not m.empty and "OR" in m.columns:
                    chk = f"  P8 run OR {float(m['OR'].iloc[0]):.3f}"
            mlog.append(f"Figure 1 {title[0]} {SHORT[k]:<12} OR {r['OR']:.3f} "
                        f"({r['lo']:.3f}-{r['hi']:.3f}) n={r['n']} events={r['events']}"
                        f"  fitted % {[round(100 * c, 1) for c in r['curve']]}{chk}")

        ymax = 0.0
        for fact in FACTS + ["POOLED"]:
            s = sub[sub["fact"] == fact].sort_values("q_num")
            x = s["q_num"].to_numpy() + 1 + offs[fact]
            yv = s["rate"].to_numpy() * 100
            lo = (s["rate"] - s["lo95"]).clip(lower=0).to_numpy() * 100
            hi = (s["hi95"] - s["rate"]).clip(lower=0).to_numpy() * 100
            ymax = max(ymax, float(np.nanmax(s["hi95"].to_numpy() * 100)))
            isp = fact == "POOLED"
            ax.errorbar(x, yv, yerr=[lo, hi], color=COLOR[fact], marker=MARKER[fact],
                        markersize=4 if isp else 3.2, capsize=2, elinewidth=0.7,
                        linestyle="none", zorder=3 if isp else 2)
            if fact in lines:
                ax.plot(np.arange(1, 5), np.array(lines[fact]["curve"]) * 100,
                        color=COLOR[fact], linewidth=1.5 if isp else 0.9,
                        linestyle="-" if isp else "--", zorder=4 if isp else 2)
            for r in s.itertuples():
                values.append({"figure": "1", "panel": title[0], "series": LABEL[fact],
                               "kind": "observed", "x": int(r.q_num) + 1, "n": int(r.n),
                               "events": int(r.events), "value_pct": r.rate * 100,
                               "lo_pct": r.lo95 * 100, "hi_pct": r.hi95 * 100})
            if fact in lines:
                for q, c in enumerate(lines[fact]["curve"]):
                    values.append({"figure": "1", "panel": title[0], "series": LABEL[fact],
                                   "kind": "fitted", "x": q + 1, "value_pct": 100 * c})
        # odds ratios, printed above the data
        txt = ["OR per quartile (95% CI)"]
        order = (["POOLED"] + FACTS) if mkey == "PRIMARY" else ["POOLED"]
        txt += [f"{SHORT[k]}: {fmt_or(lines[k])}" for k in order]
        ax.set_ylim(-0.03 * ymax * 1.75, ymax * 1.75)
        ax.text(0.03, 0.985, "\n".join(txt), transform=ax.transAxes, va="top",
                ha="left", fontsize=6.4, linespacing=1.25,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#BBBBBB", "linewidth": 0.5})
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.6, 4.4)
        ax.set_xlabel("Disturbance quartile")
        ax.set_ylabel(ylab + ", %")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    handles = [Line2D([], [], color=COLOR[f], marker=MARKER[f], linestyle="none",
                      label=LABEL[f]) for f in FACTS + ["POOLED"]]
    handles += [Line2D([], [], color="black", linewidth=1.5, label="Fitted trend, pooled"),
                Line2D([], [], color="grey", linewidth=0.9, linestyle="--",
                       label="Fitted trend, per fact (A)")]
    fig.tight_layout(w_pad=1.2, rect=(0, 0.14, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "Figure1_P8_error_by_quartile_v2")


def figure2(p4_dir: Path, out: Path, values: list[dict]) -> None:
    cols = {"fact", "scenario", "omega", "d_brier", "d_brier_lo", "d_brier_hi",
            "d_citl", "d_citl_lo", "d_citl_hi"}
    cols |= {f"d_brier_q{q}{s}" for q in range(1, 5) for s in ("", "_lo", "_hi")}
    summ = need(p4_dir / "p4_summary.csv", cols)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 86 * MM))
    ls = {"constant": "-", "load": "--"}
    off_s = {"constant": -0.012, "load": 0.012}
    off_f = {"vasopressor": -0.045, "renal_replacement": 0.0, "rbc_transfusion": 0.045}

    def by_omega(ax, metric: str, scale: float, panel: str) -> None:
        for fact in FACTS:
            for sc in ("constant", "load"):
                g = summ[(summ["fact"] == fact) & (summ["scenario"] == sc)].sort_values("omega")
                if g.empty:
                    continue
                x = g["omega"].to_numpy() + off_s[sc] + off_f[fact]
                yv = g[f"d_{metric}"].to_numpy() * scale
                lo = g[f"d_{metric}_lo"].to_numpy() * scale
                hi = g[f"d_{metric}_hi"].to_numpy() * scale
                ax.errorbar(x, yv, yerr=[np.clip(yv - lo, 0, None), np.clip(hi - yv, 0, None)],
                            color=COLOR[fact], marker=MARKER[fact], markersize=3.2,
                            linestyle=ls[sc], capsize=2, elinewidth=0.7, linewidth=1.0,
                            markerfacecolor=COLOR[fact] if sc == "constant" else "white")
                for om, a, b, c in zip(g["omega"], yv, lo, hi):
                    values.append({"figure": "2", "panel": panel,
                                   "series": f"{LABEL[fact]}, {sc} review", "x": om,
                                   "value": a, "lo": b, "hi": c})
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel("Share of training labels\nfrom reviewed output (ω)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    def key(ax, text: str, loc: str) -> None:
        y, va = (0.985, "top") if loc == "top" else (0.015, "bottom")
        ax.text(0.03, y, text, transform=ax.transAxes, va=va, ha="left", fontsize=6.2,
                linespacing=1.2, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                                       "edgecolor": "#BBBBBB", "linewidth": 0.5})

    ax = axes[0]
    by_omega(ax, "brier", 1.0, "A")
    ax.set_ylabel("Change in test Brier score vs ω = 0")
    ax.set_title("A  Label provenance", loc="left", fontweight="bold")
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.55 * (hi_ - lo_))
    key(ax, "Brier score: squared error of the\npredicted probability. Above 0:\n"
            "the retrained model is worse", "top")

    ax = axes[1]
    s1 = summ[np.isclose(summ["omega"], 1.0)]
    for fact in FACTS:
        for sc in ("constant", "load"):
            g = s1[(s1["fact"] == fact) & (s1["scenario"] == sc)]
            if g.empty:
                continue
            g = g.iloc[0]
            xs = np.arange(1, 5) + off_s[sc] * 6 + off_f[fact] * 4
            yv = np.array([g[f"d_brier_q{q}"] for q in range(1, 5)])
            lo = np.array([g[f"d_brier_q{q}_lo"] for q in range(1, 5)])
            hi = np.array([g[f"d_brier_q{q}_hi"] for q in range(1, 5)])
            ax.errorbar(xs, yv, yerr=[np.clip(yv - lo, 0, None), np.clip(hi - yv, 0, None)],
                        color=COLOR[fact], marker=MARKER[fact], markersize=3.2,
                        linestyle=ls[sc], capsize=2, elinewidth=0.7, linewidth=1.0,
                        markerfacecolor=COLOR[fact] if sc == "constant" else "white")
            for q in range(4):
                values.append({"figure": "2", "panel": "B",
                               "series": f"{LABEL[fact]}, {sc} review", "x": q + 1,
                               "value": yv[q], "lo": lo[q], "hi": hi[q]})
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Disturbance quartile\nof test patients")
    ax.set_ylabel("Change in test Brier score at ω = 1")
    ax.set_title("B  By quartile, ω = 1", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    ax = axes[2]
    by_omega(ax, "citl", 100.0, "C")
    ax.set_ylabel("Change in predicted minus observed\nrate vs ω = 0, percentage points")
    ax.set_title("C  What it means", loc="left", fontweight="bold")
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_ - 0.55 * (hi_ - lo_), hi_)
    key(ax, "Below 0: the retrained model\npredicts the treatment less\n"
            "often than it occurred", "bottom")

    fh = [Line2D([], [], color=COLOR[f], marker=MARKER[f], linewidth=1.0,
                 label=LABEL[f]) for f in FACTS]
    sh = [Line2D([], [], color="grey", linestyle="-", linewidth=1.2,
                 label="Constant review (filled)"),
          Line2D([], [], color="grey", linestyle="--", linewidth=1.2,
                 label="Load-dependent review (open)")]
    handles = [fh[0], sh[0], fh[1], sh[1], fh[2]]
    fig.tight_layout(w_pad=2.0, rect=(0, 0.12, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "Figure2_P4_retraining_v2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p8-dir", type=Path, default=DEF_P8)
    ap.add_argument("--p4-dir", type=Path, default=DEF_P4)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    args = ap.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    style()
    values: list[dict] = []
    mlog: list[str] = []
    print("writing figures")
    figure1(args.p8_dir.expanduser(), out, values, mlog)
    figure2(args.p4_dir.expanduser(), out, values)
    pd.DataFrame(values).to_csv(out / "figure_values_v2.csv", index=False)
    (out / "figure_models_v2.txt").write_text("\n".join(mlog) + "\n")
    print("\n".join(mlog))
    print(f"    {out / 'figure_values_v2.csv'}")


if __name__ == "__main__":
    main()
