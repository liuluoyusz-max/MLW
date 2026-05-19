#!/usr/bin/env python3
"""Train TabNet with StratifiedGroupKFold and build an OOF ensemble."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from train_tabnet import (
    build_extended_search_space,
    load_metadata,
    load_split,
    make_classifier,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "tabnet_preprocessed"
DEFAULT_OUTPUT_DIR = ROOT / "tabnet_groupcv_runs"
DEFAULT_SEED = 42
DEFAULT_TRIAL_NAMES = [
    "baseline_sched_cv",
    "baseline_seed7_cv",
    "lower_batch_cv",
]
MAX_WEIGHT_CANDIDATES = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-aware TabNet CV ensemble.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--trial-names",
        nargs="+",
        default=DEFAULT_TRIAL_NAMES,
        help="Trial names to include, for example baseline_sched_cv lower_batch_cv.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for top-level summary/submission artifacts.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device-name",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device passed to pytorch-tabnet. Use mps on Apple Silicon when torch MPS is available.",
    )
    return parser.parse_args()


def build_groupcv_config_registry(seed: int) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for config in build_extended_search_space(batch_size=1024, virtual_batch_size=128):
        config_copy = copy.deepcopy(config)
        trial_name = f"{config_copy['name']}_cv"
        seed_offset = int(config_copy.pop("seed_offset", 0))
        registry[trial_name] = {
            "trial_name": trial_name,
            "trial_seed": seed + seed_offset,
            "config": config_copy,
        }
    return registry


def resolve_configs(trial_names: list[str], seed: int) -> list[dict]:
    registry = build_groupcv_config_registry(seed)
    missing = [trial_name for trial_name in trial_names if trial_name not in registry]
    if missing:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown trial(s): {missing}. Available trials: {available}")
    return [copy.deepcopy(registry[trial_name]) for trial_name in trial_names]


def with_prefix(prefix: str, filename: str) -> str:
    if not prefix:
        return filename
    return f"{prefix}_{filename}"


def extract_groups(passenger_ids: pd.Series) -> np.ndarray:
    return passenger_ids.str.split("_").str[0].to_numpy()


def probability_to_label(probabilities: np.ndarray) -> np.ndarray:
    return (probabilities >= 0.5).astype(np.int64)


def count_weight_combinations(num_models: int, step: float) -> int:
    scale = int(round(1.0 / step))
    return math.comb(scale + num_models - 1, num_models - 1)


def generate_weight_combinations(num_models: int, step: float) -> list[list[float]]:
    scale = int(round(1.0 / step))
    combos: list[list[float]] = []

    def build(prefix: list[int], remaining_models: int, remaining_sum: int) -> None:
        if remaining_models == 1:
            combos.append([*(count / scale for count in prefix), remaining_sum / scale])
            return
        for count in range(remaining_sum + 1):
            build([*prefix, count], remaining_models - 1, remaining_sum - count)

    build([], num_models, scale)
    return combos


def optimize_weights(
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
    step: float,
) -> tuple[dict, list[dict]]:
    candidates = []
    trial_names = list(model_probs.keys())
    num_candidates = count_weight_combinations(len(trial_names), step)
    if num_candidates > MAX_WEIGHT_CANDIDATES:
        raise ValueError(
            "Weight search would examine "
            f"{num_candidates:,} combinations for {len(trial_names)} models at step {step}. "
            f"Use a larger --weight-step or fewer --trial-names."
        )
    combinations = generate_weight_combinations(len(trial_names), step)
    for weights in combinations:
        ensemble_prob = np.zeros_like(next(iter(model_probs.values())))
        for trial_name, weight in zip(trial_names, weights):
            ensemble_prob += weight * model_probs[trial_name]
        accuracy = float(accuracy_score(y_true, probability_to_label(ensemble_prob)))
        row = {f"weight_{trial_name}": round(weight, 4) for trial_name, weight in zip(trial_names, weights)}
        row["accuracy"] = accuracy
        candidates.append(row)
    best = max(candidates, key=lambda item: item["accuracy"])
    return best, candidates


def load_existing_outputs(
    config_dir: Path,
) -> tuple[dict, np.ndarray, np.ndarray] | None:
    summary_path = config_dir / "config_summary.json"
    oof_path = config_dir / "oof_predictions.csv"
    test_path = config_dir / "test_probabilities.csv"
    if not (summary_path.exists() and oof_path.exists() and test_path.exists()):
        return None

    config_summary = json.loads(summary_path.read_text())
    oof_prob = pd.read_csv(oof_path)["probability"].to_numpy(dtype=np.float64)
    test_prob = pd.read_csv(test_path)["probability"].to_numpy(dtype=np.float64)
    return config_summary, oof_prob, test_prob


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    selected_configs = resolve_configs(args.trial_names, args.seed)

    metadata = load_metadata(args.data_dir)
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    x_full, y_full, train_frame = load_split(
        args.data_dir / "tabnet_full_train.csv",
        feature_columns,
        target_column,
    )
    x_test, _, test_frame = load_split(
        args.data_dir / "tabnet_test.csv",
        feature_columns,
        None,
    )
    groups = extract_groups(train_frame["PassengerId"])

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_results = []
    oof_predictions = {}
    test_predictions = {}

    for config_entry in selected_configs:
        trial_name = config_entry["trial_name"]
        trial_seed = config_entry["trial_seed"]
        config = copy.deepcopy(config_entry["config"])
        config["device_name"] = args.device_name

        print(f"\n=== Config: {trial_name} ===")
        config_dir = output_dir / trial_name
        config_dir.mkdir(parents=True, exist_ok=True)

        existing = load_existing_outputs(config_dir)
        if existing is not None:
            config_summary, oof_prob, test_prob = existing
            print(f"Reusing existing outputs for {trial_name}")
            config_results.append(config_summary)
            oof_predictions[trial_name] = oof_prob
            test_predictions[trial_name] = test_prob
            continue

        oof_prob = np.zeros(len(train_frame), dtype=np.float64)
        test_prob_folds = []
        fold_summaries = []

        for fold_idx, (train_idx, valid_idx) in enumerate(
            splitter.split(x_full, y_full, groups),
            start=1,
        ):
            print(f"\n--- Fold {fold_idx}/{args.n_splits} for {trial_name} ---")
            x_train_fold = x_full[train_idx]
            y_train_fold = y_full[train_idx]
            x_valid_fold = x_full[valid_idx]
            y_valid_fold = y_full[valid_idx]

            fold_seed = trial_seed + fold_idx - 1
            model = make_classifier(metadata, fold_seed, config)
            model.fit(
                x_train_fold,
                y_train_fold,
                eval_set=[(x_train_fold, y_train_fold), (x_valid_fold, y_valid_fold)],
                eval_name=["train", "valid"],
                eval_metric=["accuracy"],
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=config["batch_size"],
                virtual_batch_size=config["virtual_batch_size"],
                num_workers=args.num_workers,
                drop_last=False,
            )

            valid_fold_prob = model.predict_proba(x_valid_fold)[:, 1]
            valid_fold_pred = probability_to_label(valid_fold_prob)
            test_fold_prob = model.predict_proba(x_test)[:, 1]
            best_epoch = int(getattr(model, "best_epoch", args.max_epochs - 1))
            valid_accuracy = float(accuracy_score(y_valid_fold, valid_fold_pred))

            oof_prob[valid_idx] = valid_fold_prob
            test_prob_folds.append(test_fold_prob)

            fold_dir = config_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            model_path = model.save_model(str(fold_dir / "tabnet_model"))
            pd.DataFrame(
                {
                    "PassengerId": train_frame.iloc[valid_idx]["PassengerId"].to_numpy(),
                    "y_true": y_valid_fold,
                    "probability": valid_fold_prob,
                    "prediction": valid_fold_pred,
                }
            ).to_csv(fold_dir / "valid_predictions.csv", index=False)

            fold_summary = {
                "fold": fold_idx,
                "fold_seed": fold_seed,
                "best_epoch": best_epoch,
                "valid_accuracy": valid_accuracy,
                "saved_model": model_path,
                "valid_rows": int(len(valid_idx)),
                "train_rows": int(len(train_idx)),
            }
            fold_summaries.append(fold_summary)
            print(
                f"Fold {fold_idx} accuracy for {trial_name}: "
                f"{valid_accuracy:.6f} at epoch {best_epoch}"
            )

        test_prob_mean = np.mean(np.vstack(test_prob_folds), axis=0)
        oof_pred = probability_to_label(oof_prob)
        oof_accuracy = float(accuracy_score(y_full, oof_pred))
        avg_best_epoch = float(np.mean([fold["best_epoch"] for fold in fold_summaries]))

        pd.DataFrame(
            {
                "PassengerId": train_frame["PassengerId"],
                "y_true": y_full,
                "probability": oof_prob,
                "prediction": oof_pred,
            }
        ).to_csv(config_dir / "oof_predictions.csv", index=False)
        pd.DataFrame(
            {
                "PassengerId": test_frame["PassengerId"],
                "probability": test_prob_mean,
            }
        ).to_csv(config_dir / "test_probabilities.csv", index=False)

        config_summary = {
            "trial_name": trial_name,
            "config": config,
            "trial_seed": trial_seed,
            "oof_accuracy": oof_accuracy,
            "avg_best_epoch": avg_best_epoch,
            "folds": fold_summaries,
        }
        (config_dir / "config_summary.json").write_text(json.dumps(config_summary, indent=2))
        print(f"OOF accuracy for {trial_name}: {oof_accuracy:.6f}")

        config_results.append(config_summary)
        oof_predictions[trial_name] = oof_prob
        test_predictions[trial_name] = test_prob_mean

    best_weight, weight_candidates = optimize_weights(
        y_full,
        oof_predictions,
        args.weight_step,
    )
    ensemble_oof_prob = np.zeros(len(y_full), dtype=np.float64)
    for config_entry in selected_configs:
        trial_name = config_entry["trial_name"]
        ensemble_oof_prob += best_weight[f"weight_{trial_name}"] * oof_predictions[trial_name]
    ensemble_oof_pred = probability_to_label(ensemble_oof_prob)
    ensemble_oof_accuracy = float(accuracy_score(y_full, ensemble_oof_pred))

    ensemble_test_prob = np.zeros(len(test_frame), dtype=np.float64)
    for config_entry in selected_configs:
        trial_name = config_entry["trial_name"]
        ensemble_test_prob += best_weight[f"weight_{trial_name}"] * test_predictions[trial_name]
    submission = pd.DataFrame(
        {
            "PassengerId": test_frame["PassengerId"],
            "Transported": [bool(value) for value in probability_to_label(ensemble_test_prob)],
        }
    )
    submission_path = output_dir / with_prefix(args.output_prefix, "groupcv_ensemble_submission.csv")
    submission.to_csv(submission_path, index=False)

    pd.DataFrame(weight_candidates).to_csv(
        output_dir / with_prefix(args.output_prefix, "ensemble_weight_search.csv"),
        index=False,
    )
    pd.DataFrame(
        {
            "PassengerId": train_frame["PassengerId"],
            "y_true": y_full,
            **{
                f"{config_entry['trial_name']}_prob": oof_predictions[config_entry["trial_name"]]
                for config_entry in selected_configs
            },
            "ensemble_prob": ensemble_oof_prob,
            "ensemble_pred": ensemble_oof_pred,
        }
    ).to_csv(
        output_dir / with_prefix(args.output_prefix, "ensemble_oof_predictions.csv"),
        index=False,
    )
    pd.DataFrame(
        {
            "PassengerId": test_frame["PassengerId"],
            **{
                f"{config_entry['trial_name']}_prob": test_predictions[config_entry["trial_name"]]
                for config_entry in selected_configs
            },
            "ensemble_prob": ensemble_test_prob,
        }
    ).to_csv(
        output_dir / with_prefix(args.output_prefix, "ensemble_test_probabilities.csv"),
        index=False,
    )

    summary = {
        "n_splits": args.n_splits,
        "trial_names": args.trial_names,
        "configs": config_results,
        "best_weight": best_weight,
        "ensemble_oof_accuracy": ensemble_oof_accuracy,
        "submission_path": str(submission_path),
    }
    summary_path = output_dir / with_prefix(args.output_prefix, "groupcv_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    weight_parts = []
    for config_entry in selected_configs:
        trial_name = config_entry["trial_name"]
        weight_parts.append(f"{trial_name}={best_weight[f'weight_{trial_name}']:.2f}")
    weight_message = ", ".join(weight_parts)
    print(
        f"\nGroupCV ensemble OOF accuracy: {ensemble_oof_accuracy:.6f} "
        f"with weights {weight_message}"
    )
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
