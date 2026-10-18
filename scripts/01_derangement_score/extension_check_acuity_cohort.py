# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
# ]
# ///
"""
extension_check_acuity_cohort.py -- describes the cohort behind
paper18_phase1_acuity.csv from the file itself, so the manuscript can state
the source cohort without relying on which version of the build script was
used. It also lists the candidate build scripts with their dates and hashes,
to identify the one that most plausibly wrote the file.

Prints aggregates only (counts, ranges); no patient-level data.

CHECKS
  1. size of the file: stays, admissions, patients
  2. ICU unit: first_careunit of each stay (MIMIC-IV icustays)
  3. length of stay: minimum, centiles, share >= 72 hours
  4. first MICU stay of the admission? (rank by intime among MICU stays)
  5. age: minimum anchor_age
  6. score: range of acuity_proxy_24h and of each component; quartile counts
     and cut points
  7. hours of data behind the score (n_acuity_hours)
  8. candidate build scripts under --projects: path, size, modified time,
     SHA-256 (first 12), markers found in the text, and whether they predate
     the acuity file

USAGE
  uv run --offline extension_check_acuity_cohort.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
DEF_ACUITY = HOME / "projects" / "BCST" / "paper18_results" / "paper18_phase1_acuity.csv"
DEF_MIMIC = HOME / "physionet.org" / "files" / "mimiciv" / "3.1"
DEF_PROJECTS = HOME / "projects"
MICU_RE = r"\(MICU\)$"
MARKERS = ["MICU", "72", "MIN_EFFECTIVE", "ANALYSIS_WINDOW", "warmup", "qcut",
           "vent_first_24h", "anchor_age", "acuity_proxy_24h", "acuity_quartile",
           "paper18_phase1_acuity.csv"]


def find_table(root: Path, name: str) -> Path:
    hits = [p for ext in (".csv.gz", ".csv") for p in root.rglob(f"{name}{ext}")]
    if not hits:
        sys.exit(f"[fatal] {name} not found under {root}")
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0]


def stamp(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acuity", type=Path, default=DEF_ACUITY)
    ap.add_argument("--mimic-root", type=Path, default=DEF_MIMIC)
    ap.add_argument("--projects", type=Path, default=DEF_PROJECTS)
    args = ap.parse_args()
    acu_path = args.acuity.expanduser()
    if not acu_path.exists():
        sys.exit(f"[fatal] {acu_path} not found")
    ac = pd.read_csv(acu_path, low_memory=False)
    print(f"acuity file   {acu_path}")
    print(f"              modified {stamp(acu_path)}, {len(ac)} rows")
    print(f"columns       {list(ac.columns)}")

    icu_path = find_table(args.mimic_root.expanduser(), "icustays")
    icu = pd.read_csv(icu_path, usecols=["subject_id", "hadm_id", "stay_id",
                                          "first_careunit", "intime", "los"])
    icu["intime"] = pd.to_datetime(icu["intime"])
    m = ac[["stay_id"]].merge(icu, on="stay_id", how="left")
    print(f"\n1. stays {ac['stay_id'].nunique()}  admissions {m['hadm_id'].nunique()}  "
          f"patients {m['subject_id'].nunique()}  (not in icustays: "
          f"{int(m['hadm_id'].isna().sum())})")

    print("\n2. first_careunit")
    vc = m["first_careunit"].value_counts(dropna=False)
    for unit, n in vc.head(10).items():
        print(f"     {n:>6}  {unit}")
    micu = m["first_careunit"].astype(str).str.contains(MICU_RE, regex=True)
    print(f"   share matching {MICU_RE}: {micu.mean():.4f}")

    los_h = m["los"] * 24
    print("\n3. length of stay, hours: min {:.1f}  p1 {:.1f}  p5 {:.1f}  median {:.1f}  "
          "max {:.1f}".format(los_h.min(), los_h.quantile(0.01), los_h.quantile(0.05),
                               los_h.median(), los_h.max()))
    print(f"   share >= 72 h: {(los_h >= 72).mean():.4f}   share >= 48 h: "
          f"{(los_h >= 48).mean():.4f}")

    mi = icu[icu["first_careunit"].astype(str).str.contains(MICU_RE, regex=True)].copy()
    mi["rank_in_hadm"] = mi.groupby("hadm_id")["intime"].rank(method="first")
    r = m[["stay_id"]].merge(mi[["stay_id", "rank_in_hadm"]], on="stay_id", how="left")
    print("\n4. rank among MICU stays of the same admission (1 = first):")
    print("   " + r["rank_in_hadm"].value_counts(dropna=False).sort_index()
          .to_string().replace("\n", "\n   "))
    allstays = icu.groupby("hadm_id")["stay_id"].size()
    multi = m["hadm_id"].map(allstays)
    print(f"   admissions with more than one ICU stay (any unit): "
          f"{int((multi > 1).sum())} of {len(m)} stays")

    if "anchor_age" in ac.columns:
        print(f"\n5. anchor_age: min {ac['anchor_age'].min()}  median "
              f"{ac['anchor_age'].median()}  max {ac['anchor_age'].max()}")

    print("\n6. score")
    if "acuity_proxy_24h" in ac.columns:
        s = ac["acuity_proxy_24h"]
        print(f"   acuity_proxy_24h: min {s.min()}  max {s.max()}  mean {s.mean():.2f}  "
              f"missing {int(s.isna().sum())}")
    for c in [c for c in ac.columns if c.startswith("score_")]:
        v = ac[c]
        print(f"   {c:<12} values {sorted(v.dropna().unique().tolist())}")
    if {"acuity_quartile", "acuity_proxy_24h"} <= set(ac.columns):
        g = (ac.groupby("acuity_quartile")["acuity_proxy_24h"]
             .agg(["size", "min", "max"]))
        print("   quartile: n, score min, score max")
        print("   " + g.to_string().replace("\n", "\n   "))

    if "n_acuity_hours" in ac.columns:
        h = ac["n_acuity_hours"]
        print(f"\n7. n_acuity_hours: min {h.min()}  median {h.median()}  max {h.max()}")

    print("\n8. candidate build scripts under", args.projects.expanduser())
    acu_time = acu_path.stat().st_mtime
    cands = sorted({p for pat in ("paper18x_acuity_build*.py", "paper18*acuity*.py",
                                  "paper18_phase1*.py")
                    for p in args.projects.expanduser().rglob(pat)},
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        print("   none found")
    for p in cands:
        txt = p.read_text(errors="replace")
        sha = hashlib.sha256(txt.encode()).hexdigest()[:12]
        found = [k for k in MARKERS if re.search(re.escape(k), txt)]
        before = "before" if p.stat().st_mtime <= acu_time else "AFTER"
        print(f"   {stamp(p)}  {before:<6} {p.stat().st_size:>7} B  sha {sha}  {p}")
        print(f"        markers: {', '.join(found) if found else '-'}")
    prior = [p for p in cands if p.stat().st_mtime <= acu_time]
    if prior:
        print(f"\n   latest script modified before the acuity file: {prior[-1]}")


if __name__ == "__main__":
    main()
