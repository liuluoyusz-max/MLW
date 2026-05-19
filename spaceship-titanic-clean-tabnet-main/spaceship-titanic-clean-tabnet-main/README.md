# Spaceship Titanic Clean TabNet Submission

This repository contains my final clean TabNet model line for the live demo and
code submission. I was responsible for the TabNet training path, and this is
the TabNet version I selected for checking.

## Why This Version

- Model: Group-aware TabNet ensemble.
- Validation: 5-fold `StratifiedGroupKFold` using passenger group IDs.
- Clean OOF accuracy: `0.816864`.
- Known public Kaggle score for this clean line: `0.81342`.
- Scope: raw-data feature engineering, group-aware cross-validation, and TabNet
  probability ensembling only.

The live VSCode demo regenerates a Kaggle-ready CSV from saved clean OOF/test
probability artifacts. The smoke demo proves that the raw-data preprocessing
and TabNet training code path is runnable.

## Files

| Path | Purpose |
| --- | --- |
| `demo_clean_tabnet.py` | Live demo entrypoint; regenerates a Kaggle-ready clean TabNet submission CSV. |
| `demo_full_pipeline.py` | Wrapper for fast, smoke, or full training demo modes. |
| `prepare_tabnet_data.py` | Shared base preprocessing helpers. |
| `prepare_tabnet_data_fe_signal_v1.py` | Preprocesses `train.csv` and `test.csv` for TabNet. |
| `groupcv_tabnet_ensemble.py` | Trains GroupCV TabNet models and builds an OOF ensemble. |
| `train_tabnet.py` | Shared TabNet model/config utilities. |
| `tabnet_groupcv_runs_fe_signal_v1/` | Saved clean model probability artifacts used by the fast demo. |
| `submissions/clean_tabnet_public_0.81342_submission.csv` | Final selected Kaggle submission file. |

## Environment

Python `3.10+` is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation needs to be platform-specific, install `torch` first
from the official PyTorch instructions for your machine, then run the command
above.

## VSCode Live Kaggle Demo

Use this for the live demo and Kaggle submission check. Open this repository
folder in VSCode, then run the command below in the VSCode terminal:

```bash
python demo_clean_tabnet.py --output submissions/live_demo_clean_tabnet_submission.csv
```

On my Mac demo environment, the exact Python command is:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tabnet-mps/bin/python demo_clean_tabnet.py --output submissions/live_demo_clean_tabnet_submission.csv
```

Expected output:

- writes `submissions/live_demo_clean_tabnet_submission.csv`
- recomputes OOF accuracy as `0.816864`
- prints the known Kaggle public score `0.81342`
- prints prediction counts for the generated submission

Submit this generated CSV to Kaggle:

```text
submissions/live_demo_clean_tabnet_submission.csv
```

This live-generated CSV is produced from the saved clean TabNet OOF/test
probability artifacts and matches the selected final clean TabNet submission.

## Optional Fast Artifact Check

If no output path is provided, the same script writes to `demo_outputs/`:

```bash
python demo_clean_tabnet.py
```

This is useful for a quick artifact check, but for the live Kaggle upload demo
use the explicit `submissions/live_demo_clean_tabnet_submission.csv` command
above.

## Raw-Data Smoke Demo

Use this if the reviewer wants to see source code running from raw CSV files:

```bash
python demo_full_pipeline.py --mode smoke --device-name cpu
```

This runs:

1. `prepare_tabnet_data_fe_signal_v1.py`
2. a tiny 2-fold, 5-epoch TabNet training pass
3. submission generation in `demo_outputs/smoke_groupcv_tabnet/`

The smoke run is intentionally short. It proves the code path, but it is not
the final score-producing model.

## Full Training

Only run this when enough time/compute is available:

```bash
python demo_full_pipeline.py --mode train --device-name auto
```

This retrains the clean 3-model GroupCV TabNet ensemble from raw CSV files.

## Final Submission

For the live demo, submit the newly generated file:

```text
submissions/live_demo_clean_tabnet_submission.csv
```

The selected locked reference file is:

```text
submissions/clean_tabnet_public_0.81342_submission.csv
```

Both are generated from the same clean TabNet probabilities. The locked
reference file is equivalent to:

```text
tabnet_groupcv_runs_fe_signal_v1/fe_signal_v1_3model_groupcv_ensemble_submission.csv
```
