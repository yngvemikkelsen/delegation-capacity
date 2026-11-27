# delegation-capacity

Code for:

Mikkelsen Y. *From delegation to usable capacity: extending the supply–demand production condition to human-to-machine delegation.* Frontiers in Health Services (Hypothesis and Theory), submitted 2026, revised 2026.

The repository holds the scripts behind the two empirical tests in the paper (P8, error–demand coupling; P4, retraining on reviewed output), the post hoc checks of the P8 vasopressor result, the checks added in review, the disturbance composite used for stratification, the figures, the analysis plans and the console output of the analysed runs. **It contains no MIMIC data.**

## Data access

The analyses use MIMIC-IV 3.1 and MIMIC-IV-Note 2.2, available to credentialed users through PhysioNet under the MIMIC data use agreements. All processing, including the language model, runs on a local machine; no note text leaves it.

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/). Every script declares its dependencies inline (PEP 723), so `uv run <script>.py` sets up its own environment.
- P8: [Ollama](https://ollama.com) with `qwen3:30b` (analysed run: Ollama 0.35.0, Q4_K_M, digest `ad815644918f`, which on ollama.com is also the tag `qwen3:30b-a3b-thinking-2507-q4_K_M`). The reruns added in review (steps 8 and 8a) used the same digest under Ollama 0.40.15.
- P4: the encoder `BAAI/bge-base-en-v1.5`, downloaded from Hugging Face on first run.
- Default paths: `~/physionet.org/files/mimiciv/3.1`, `~/physionet.org/files/mimic-iv-note/2.2/note`, outputs under `~/extension_results/`. Each script accepts flags to change them (`--help`).

## Run order

| Step | Script | What it does | Paper |
|---|---|---|---|
| 1 | `scripts/01_derangement_score/paper18_phase1_granger.py` | Computes the disturbance composite (`paper18_phase1_acuity.csv`) for 5,878 medical ICU stays. Inputs are the hourly series and cohort files of the earlier hourly time-series analysis and a per-stay file with age and ventilation in the first 24 hours; the script's Granger analysis is not used here. | 5.1 |
| 1a | `scripts/01_derangement_score/extension_check_acuity_cohort.py` | Describes the composite file (unit, length of stay, age, score range, quartile cut points) and identifies the script that wrote it. Output: `results/logs/acuity_cohort_check.log`. | 5.1 |
| 2 | `scripts/02_p8_error_demand/extension_p8_error_demand_v2.py` | P8: the delegate answers three questions per discharge summary; misses by disturbance quartile, prespecified primary and secondary models. `--pilot 40` runs the pilot. | 5.2–5.4, 6.1 |
| 3 | `scripts/03_p8_post_hoc/extension_p8_vaso_diagnostic_v2.py` | Post hoc checks of the vasopressor result. Without arguments: context classes, vocabulary groups and the blinded review sheet. With `--labels <file>`: analysis of the blinded reading. | 5.4, 6.2 |
| 3a | `scripts/03_p8_post_hoc/extension_p8_review_tool.py` | Turns the review sheet into a local review page and exports the labels. | 5.4 |
| 4 | `scripts/04_p4_label_provenance/extension_p4_label_provenance.py` | P4: retraining on reviewed output; reads the P8 outputs. | 5.5, 6.3 |
| 5 | `scripts/05_figures/extension_figures.py` | Figures 1 and 2 as submitted (300 dpi, 180 mm) from the saved result tables. | Figures (submitted) |
| 6 | `scripts/05_figures/extension_figures_v2.py` | Figures 1 and 2 as revised: fitted trend lines and odds ratios in Figure 1; a plain-language key and a calibration panel (C) in Figure 2. Refits the P8 models from the per-note table and checks them against the P8 run. | Figures |
| 7 | `scripts/06_review_checks/extension_revision_checks_v2.py` | Checks added in review: (A) replay of the P4 random draws to count the errors that simulated review corrected and left; (B1) reference events by status and amount; (B2) treatment recorded outside the ICU stay. No language model, no note text. | 5.3, 6.1, 6.3; S6.1–S6.3 |
| 8 | `scripts/07_p8_review_rerun/extension_p8_vaso_any_v3.py` | Rerun of the delegate with only the vasopressor question broadened to any vasopressor, including pressor support. | 5.4, 6.2; S6.4 |
| 8a | `scripts/07_p8_review_rerun/extension_p8_v2_control.py` | Control for step 8: the original P8 prompt (hash-checked) under the updated Ollama version, for the 1,699 primary-set vasopressor notes. | 5.4, 6.2; S6.4 |

`scripts/02_p8_error_demand/pilot/extension_p8_error_demand.py` is the first version, used only for the 40-note pilot. Its answers were discarded: after the pilot a dialysis-catheter item was removed from the reference and peritoneal dialysis was added to the question (paper, section 5.6).

## Analysis plans

`analysis_plans/` holds the plan texts exactly as the scripts write them: the P8 plan before the first model call, the P4 plan before the experiment. The scripts refuse to run if an existing plan file differs, and record its SHA-256 in the run manifest. The plans were not registered externally. The P4 primary endpoint was changed from AUROC to Brier score after tests on synthetic data and before any real data were used.

## Results

- `results/logs/` — console output of the analysed runs (aggregates only): pilot, P8, P4, post hoc vocabulary check, blinded reading, figures, cohort check, and, from the revision, the review checks (v1 and v2), the revised figures, the any-vasopressor rerun and its control.
- `results/review_labels.csv` — the reader's labels for the 240 blinded items (review_id, label). Mode 1 of step 3 regenerates the same sample and key from the same inputs with the default seed (20261018).

## Versions

`extension_p8_vaso_diagnostic_v2.py` produced the reported post hoc results. A later v3 differs only in printing a clearer message when the review sheet has no labels, and is not included.

Release 1.1.0 accompanies the revised paper. It adds steps 6–8a and their logs and changes no script of release 1.0.0. `extension_revision_checks.py` (v1) is kept with its log for the record; v2 differs only in excluding a dialysis access procedure code (ICD-9 39.27) from the procedure-code search, and produced the reported figures.

## Terminology

The paper calls the stratifying composite *acute physiological disturbance*; release 1.0.0 and an earlier version of the manuscript called it *derangement*. Folder names, script names, code comments and labels in the release 1.0.0 scripts keep *derangement*. They are left unchanged so that the archived code is the code that produced the results.

## Licence

Code: MIT License (see `LICENSE`). Aggregate results, analysis plans and review labels (`results/`, `analysis_plans/`): CC BY 4.0. No patient-level data are included; data access is governed by the PhysioNet data use agreements.

## Citation

See `CITATION.cff`. The archived release is on Zenodo; the DOI is given in the paper's Data availability statement.
