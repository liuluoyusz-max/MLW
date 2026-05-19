#!/usr/bin/env python3
"""Fast demo entrypoint for the clean TabNet submission line.

This script regenerates the Kaggle submission from the saved clean GroupCV
TabNet probability artifacts selected for the TabNet demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "tabnet_groupcv_runs_fe_signal_v1" / "fe_signal_v1_3model_groupcv_summary.json"
OOF_PATH = ROOT / "tabnet_groupcv_runs_fe_signal_v1" / "fe_signal_v1_3model_ensemble_oof_predictions.csv"
TEST_PROB_PATH = (
    ROOT
    / "tabnet_groupcv_runs_fe_signal_v1"
    / "fe_signal_v1_3model_ensemble_test_probabilities.csv"
)
DEFAULT_OUTPUT = ROOT / "demo_outputs" / "clean_tabnet_demo_submission.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the clean TabNet demo submission.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the generated Kaggle submission CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = json.loads(SUMMARY_PATH.read_text())
    oof = pd.read_csv(OOF_PATH)
    test_probs = pd.read_csv(TEST_PROB_PATH)

    required_oof = {"PassengerId", "y_true", "ensemble_prob", "ensemble_pred"}
    required_test = {"PassengerId", "ensemble_prob"}
    missing_oof = required_oof.difference(oof.columns)
    missing_test = required_test.difference(test_probs.columns)
    if missing_oof or missing_test:
        raise ValueError(f"Missing expected columns: oof={missing_oof}, test={missing_test}")

    oof_accuracy = float((oof["y_true"].astype(int) == oof["ensemble_pred"].astype(int)).mean())
    submission = pd.DataFrame(
        {
            "PassengerId": test_probs["PassengerId"],
            "Transported": test_probs["ensemble_prob"] >= 0.5,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    counts = submission["Transported"].value_counts().to_dict()
    print("Clean TabNet demo submission generated")
    print(f"Output: {args.output}")
    print(f"OOF accuracy recomputed from saved artifacts: {oof_accuracy:.6f}")
    print(f"OOF accuracy recorded in summary: {summary['ensemble_oof_accuracy']:.6f}")
    print("Known Kaggle public score for this clean line: 0.81342")
    print(f"Prediction counts: {counts}")
    print("Scope: clean TabNet GroupCV probability artifacts.")


if __name__ == "__main__":
    main()
