# delegation-capacity

Code for:

Mikkelsen Y. *From delegation to usable capacity: extending the supply–demand production condition to human-to-machine delegation.* Frontiers in Health Services (Hypothesis and Theory), submitted 2026.

The repository holds the scripts behind the two empirical tests in the paper (P8, error–demand coupling; P4, retraining on reviewed output), the post hoc checks of the P8 vasopressor result, the derangement score used for stratification, the figures, the analysis plans and the console output of the analysed runs. **It contains no MIMIC data.**

## Data access

The analyses use MIMIC-IV 3.1 and MIMIC-IV-Note 2.2, available to credentialed users through PhysioNet under the MIMIC data use agreements. All processing, including the language model, runs on a local machine; no note text leaves it.

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/). Every script declares its dependencies inline (PEP 723), so `uv run <script>.py` sets up its own environment.
- P8: [Ollama](https://ollama.com) with `qwen3:30b` (analysed run: Ollama 0.35.0, Q4_K_M, digest `ad815644918f`).
- P4: the encoder `BAAI/bge-base-en-v1.5`, downloaded from Hugging Face on first run.
- Default paths: `~/physionet.org/files/mimiciv/3.1`, `~/physionet.org/files/mimic-iv-note/2.2/note`, outputs under `~/extension_results/`. Each script accepts flags to change them (`--help`).

## Run order

| Step | Script | What it does | Paper |
|---|---|---|---|
| 1 | `scripts/01_derangement_score/paper18_phase1_granger.py` | Computes the derangement composite (`paper18_phase1_acuity.csv`) for 5,878 medical ICU stays. Inputs are the hourly series and cohort files of the earlier hourly time-series analysis and a per-stay file with age and ventilation in the first 24 hours; the script's Granger analysis is not used here. | 5.1 |
| 1a | `scripts/01_derangement_score/extension_check_acuity_cohort.py` | Describes the composite file (unit, length of stay, age, score range, quartile cut points) and identifies the script that wrote it. Output: `results/logs/acuity_cohort_check.log`. | 5.1 |
| 2 | `scripts/02_p8_error_demand/extension_p8_error_demand_v2.py` | P8: the delegate answers three questions per discharge summary; misses by derangement quartile, prespecified primary and secondary models. `--pilot 40` runs the pilot. | 5.2–5.4, 6.1 |
| 3 | `scripts/03_p8_post_hoc/extension_p8_vaso_diagnostic_v2.py` | Post hoc checks of the vasopressor result. Without arguments: context classes, vocabulary groups and the blinded review sheet. With `--labels <file>`: analysis of the blinded reading. | 5.4, 6.2 |
| 3a | `scripts/03_p8_post_hoc/extension_p8_review_tool.py` | Turns the review sheet into a local review page and exports the labels. | 5.4 |
| 4 | `scripts/04_p4_label_provenance/extension_p4_label_provenance.py` | P4: retraining on reviewed output; reads the P8 outputs. | 5.5, 6.3 |
| 5 | `scripts/05_figures/extension_figures.py` | Figures 1 and 2 (300 dpi, 180 mm) from the saved result tables. | Figures |

`scripts/02_p8_error_demand/pilot/extension_p8_error_demand.py` is the first version, used only for the 40-note pilot. Its answers were discarded: after the pilot a dialysis-catheter item was removed from the reference and peritoneal dialysis was added to the question (paper, section 5.6).

## Analysis plans

`analysis_plans/` holds the plan texts exactly as the scripts write them: the P8 plan before the first model call, the P4 plan before the experiment. The scripts refuse to run if an existing plan file differs, and record its SHA-256 in the run manifest. The plans were not registered externally. The P4 primary endpoint was changed from AUROC to Brier score after tests on synthetic data and before any real data were used.

## Results

- `results/logs/` — console output of the analysed runs (aggregates only): pilot, P8, P4, post hoc vocabulary check, blinded reading, figures, cohort check.
- `results/review_labels.csv` — the reader's labels for the 240 blinded items (review_id, label). Mode 1 of step 3 regenerates the same sample and key from the same inputs with the default seed (20261018).

## Versions

`extension_p8_vaso_diagnostic_v2.py` produced the reported post hoc results. A later v3 differs only in printing a clearer message when the review sheet has no labels, and is not included.

## Citation

See `CITATION.cff`. The archived release is on Zenodo; the DOI is given in the paper's Data availability statement.
