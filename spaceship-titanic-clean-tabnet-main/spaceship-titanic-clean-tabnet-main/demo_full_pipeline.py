#!/usr/bin/env python3
"""Run the clean TabNet demo pipeline from code.

Modes:
- fast: regenerate the locked main submission from saved clean model artifacts.
- smoke: run raw-data preprocessing plus a tiny GroupCV TabNet training pass.
- train: run raw-data preprocessing plus the full clean GroupCV TabNet training.

The train mode is the complete TabNet code path for a strict live demo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean TabNet demo pipeline wrapper.")
    parser.add_argument(
        "--mode",
        choices=["fast", "smoke", "train"],
        default="fast",
        help="Demo depth. Use train only when the teacher requires full retraining.",
    )
    parser.add_argument(
        "--device-name",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device passed to pytorch-tabnet for smoke/train modes.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    if args.mode == "fast":
        run([sys.executable, "demo_clean_tabnet.py"])
        return

    run([sys.executable, "prepare_tabnet_data_fe_signal_v1.py"])

    if args.mode == "smoke":
        run(
            [
                sys.executable,
                "groupcv_tabnet_ensemble.py",
                "--data-dir",
                "tabnet_preprocessed_fe_signal_v1",
                "--output-dir",
                "demo_outputs/smoke_groupcv_tabnet",
                "--output-prefix",
                "smoke",
                "--trial-names",
                "baseline_sched_cv",
                "--n-splits",
                "2",
                "--max-epochs",
                "5",
                "--patience",
                "3",
                "--weight-step",
                "1.0",
                "--device-name",
                args.device_name,
            ]
        )
        print("\nSmoke run complete. This proves the raw-data-to-submission code path.")
        print("It is intentionally short and is not the final Kaggle result.")
        return

    run(
        [
            sys.executable,
            "groupcv_tabnet_ensemble.py",
            "--data-dir",
            "tabnet_preprocessed_fe_signal_v1",
            "--output-dir",
            "demo_outputs/full_groupcv_tabnet",
            "--output-prefix",
            "full_clean_tabnet",
            "--trial-names",
            "baseline_sched_cv",
            "baseline_seed7_cv",
            "lower_batch_cv",
            "--n-splits",
            "5",
            "--max-epochs",
            "140",
            "--patience",
            "25",
            "--weight-step",
            "0.01",
            "--device-name",
            args.device_name,
        ]
    )
    print("\nFull clean TabNet training complete.")
    print("Submission: demo_outputs/full_groupcv_tabnet/full_clean_tabnet_groupcv_ensemble_submission.csv")


if __name__ == "__main__":
    main()
