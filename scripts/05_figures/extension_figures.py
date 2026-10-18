# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "matplotlib>=3.8",
#   "pillow>=8",
# ]
# ///
"""
extension_figures.py -- submission figures for the extension paper.

Redraws Figure 1 (P8) and Figure 2 (P4) from the saved result tables, with
clean labels, at 300 dpi and 180 mm width (Frontiers double column), RGB. No
analysis is recomputed: every plotted value is read from the CSVs written
by the analysis scripts.

INPUTS
    --p8-dir  folder of extension_p8_error_demand_v2.py   (p8_rates_by_stratum.csv)
    --p4-dir  folder of extension_p4_label_provenance.py  (p4_summary.csv)

OUTPUTS (in --out, default ~/extension_results/figures)
    Figure1_P8_error_by_quartile.tif / .png
    Figure2_P4_retraining.tif / .png
    figure_values.csv   every plotted value with its interval, for checking

USAGE
    uv run --offline extension_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

HOME = Path.home()
DEF_P8 = HOME / "extension_results" / "p8"
DEF_P4 = HOME / "extension_results" / "p4"
DEF_OUT = HOME / "extension_results" / "figures"

FACTS = ["vasopressor", "renal_replacement", "rbc_transfusion"]
LABEL = {"vasopressor": "Vasopressor", "renal_replacement": "Renal replacement therapy",
         "rbc_transfusion": "Red-cell transfusion", "POOLED": "All three facts"}
# Okabe-Ito palette (colour-blind safe); markers differ as well
COLOR = {"vasopressor": "#0072B2", "renal_replacement": "#D55E00",
         "rbc_transfusion": "#009E73", "POOLED": "#000000"}
MARKER = {"vasopressor": "o", "renal_replacement": "^", "rbc_transfusion": "D",
          "POOLED": "s"}
P8_SETS = [
    ("PRIMARY documented positives", "A  In record and note",
     "Answered ‘no’ (miss)"),
    ("S1 all structured positives", "B  In record",
     "Answered ‘no’ (miss)"),
    ("S2 unsupported positives", "C  Not in record or note",
     "Answered ‘yes’"),
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
    # fixed canvas (no tight bbox), so the width stays exactly 180 mm
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
    df = pd.read_csv(path)
    miss = cols - set(df.columns)
    if miss:
        sys.exit(f"[fatal] {path} lacks columns {sorted(miss)}")
    return df


def figure1(p8_dir: Path, out: Path, values: list[dict]) -> None:
    rates = need(p8_dir / "p8_rates_by_stratum.csv",
                 {"set", "fact", "q_num", "n", "events", "rate", "lo95", "hi95"})
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 72 * MM))
    offs = {"vasopressor": -0.12, "renal_replacement": -0.04,
            "rbc_transfusion": 0.04, "POOLED": 0.12}
    for ax, (key, title, ylab) in zip(axes, P8_SETS):
        sub = rates[rates["set"] == key]
        if sub.empty:
            sys.exit(f"[fatal] set '{key}' not in p8_rates_by_stratum.csv")
        for fact in FACTS + ["POOLED"]:
            s = sub[sub["fact"] == fact].sort_values("q_num")
            x = s["q_num"].to_numpy() + 1 + offs[fact]
            y = s["rate"].to_numpy() * 100
            lo = (s["rate"] - s["lo95"]).clip(lower=0).to_numpy() * 100
            hi = (s["hi95"] - s["rate"]).clip(lower=0).to_numpy() * 100
            pooled = fact == "POOLED"
            ax.errorbar(x, y, yerr=[lo, hi], color=COLOR[fact], marker=MARKER[fact],
                        markersize=4 if pooled else 3.2, capsize=2, elinewidth=0.7,
                        linewidth=1.6 if pooled else 0.9, zorder=3 if pooled else 2)
            for r in s.itertuples():
                values.append({"figure": "1", "panel": title[0], "series": LABEL[fact],
                               "x": int(r.q_num) + 1, "n": int(r.n),
                               "events": int(r.events), "value_pct": r.rate * 100,
                               "lo_pct": r.lo95 * 100, "hi_pct": r.hi95 * 100})
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.6, 4.4)
        top = ax.get_ylim()[1]
        ax.set_ylim(-0.03 * top, top)
        ax.set_xlabel("Derangement quartile")
        ax.set_ylabel(ylab + ", %")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    handles = [Line2D([], [], color=COLOR[f], marker=MARKER[f],
                      linewidth=1.6 if f == "POOLED" else 0.9, label=LABEL[f])
               for f in FACTS + ["POOLED"]]
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "Figure1_P8_error_by_quartile")


def figure2(p4_dir: Path, out: Path, values: list[dict]) -> None:
    cols = {"fact", "scenario", "omega", "d_brier", "d_brier_lo", "d_brier_hi"}
    cols |= {f"d_brier_q{q}{s}" for q in range(1, 5) for s in ("", "_lo", "_hi")}
    summ = need(p4_dir / "p4_summary.csv", cols)
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 80 * MM))
    ls = {"constant": "-", "load": "--"}
    off_s = {"constant": -0.012, "load": 0.012}
    off_f = {"vasopressor": -0.045, "renal_replacement": 0.0, "rbc_transfusion": 0.045}
    ax = axes[0]
    for fact in FACTS:
        for sc in ("constant", "load"):
            g = summ[(summ["fact"] == fact) & (summ["scenario"] == sc)].sort_values("omega")
            if g.empty:
                continue
            x = g["omega"].to_numpy() + off_s[sc] + off_f[fact]
            ax.errorbar(x, g["d_brier"], yerr=[(g["d_brier"] - g["d_brier_lo"]).clip(lower=0),
                                               (g["d_brier_hi"] - g["d_brier"]).clip(lower=0)],
                        color=COLOR[fact], marker=MARKER[fact], markersize=3.2,
                        linestyle=ls[sc], capsize=2, elinewidth=0.7, linewidth=1.0,
                        markerfacecolor=COLOR[fact] if sc == "constant" else "white")
            for r in g.itertuples():
                values.append({"figure": "2", "panel": "A",
                               "series": f"{LABEL[fact]}, {sc} review", "x": r.omega,
                               "value": r.d_brier, "lo": r.d_brier_lo, "hi": r.d_brier_hi})
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("Share of training labels from reviewed output (ω)")
    ax.set_ylabel("Change in test Brier score vs ω = 0")
    ax.set_title("A  Label provenance", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    ax = axes[1]
    s1 = summ[np.isclose(summ["omega"], 1.0)]
    for fact in FACTS:
        for sc in ("constant", "load"):
            g = s1[(s1["fact"] == fact) & (s1["scenario"] == sc)]
            if g.empty:
                continue
            g = g.iloc[0]
            xs = np.arange(1, 5) + off_s[sc] * 6 + off_f[fact] * 4
            y = np.array([g[f"d_brier_q{q}"] for q in range(1, 5)])
            lo = np.array([g[f"d_brier_q{q}_lo"] for q in range(1, 5)])
            hi = np.array([g[f"d_brier_q{q}_hi"] for q in range(1, 5)])
            ax.errorbar(xs, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                        color=COLOR[fact], marker=MARKER[fact], markersize=3.2,
                        linestyle=ls[sc], capsize=2, elinewidth=0.7, linewidth=1.0,
                        markerfacecolor=COLOR[fact] if sc == "constant" else "white")
            for q in range(4):
                values.append({"figure": "2", "panel": "B",
                               "series": f"{LABEL[fact]}, {sc} review", "x": q + 1,
                               "value": y[q], "lo": lo[q], "hi": hi[q]})
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Derangement quartile of test patients")
    ax.set_ylabel("Change in test Brier score at ω = 1")
    ax.set_title("B  Where the loss falls (ω = 1)", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    fh = [Line2D([], [], color=COLOR[f], marker=MARKER[f], linewidth=1.0,
                 label=LABEL[f]) for f in FACTS]
    sh = [Line2D([], [], color="grey", linestyle="-", linewidth=1.2,
                 label="Constant review (filled)"),
          Line2D([], [], color="grey", linestyle="--", linewidth=1.2,
                 label="Load-dependent review (open)")]
    # column-major fill: facts on the first row, scenarios on the second
    handles = [fh[0], sh[0], fh[1], sh[1], fh[2]]
    fig.tight_layout(w_pad=1.5, rect=(0, 0.13, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "Figure2_P4_retraining")


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
    print("writing figures")
    figure1(args.p8_dir.expanduser(), out, values)
    figure2(args.p4_dir.expanduser(), out, values)
    pd.DataFrame(values).to_csv(out / "figure_values.csv", index=False)
    print(f"    {out / 'figure_values.csv'}")


if __name__ == "__main__":
    main()
