#!/usr/bin/env python3
"""Train and evaluate TabNet on the preprocessed Spaceship Titanic dataset."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "tabnet_preprocessed"
DEFAULT_OUTPUT_DIR = ROOT / "tabnet_runs"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TabNet for Spaceship Titanic.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--search-mode",
        choices=["quick", "extended"],
        default="quick",
        help="Use the original small search or a broader tuning sweep.",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--virtual-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device-name",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device passed to pytorch-tabnet. Use mps on Apple Silicon when torch MPS is available.",
    )
    parser.add_argument("--skip-final-train", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir: Path) -> dict:
    return json.loads((data_dir / "tabnet_metadata.json").read_text())


def load_split(
    csv_path: Path,
    feature_columns: list[str],
    target_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    target = None
    if target_column is not None:
        target = frame[target_column].to_numpy(dtype=np.int64)
    return features, target, frame


def build_search_space(batch_size: int, virtual_batch_size: int) -> list[dict]:
    return [
        {
            "name": "baseline",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-4,
            "momentum": 0.02,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 1,
            "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
        {
            "name": "wider",
            "n_d": 24,
            "n_a": 24,
            "n_steps": 5,
            "gamma": 1.5,
            "lambda_sparse": 5e-5,
            "momentum": 0.02,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 1,
            "optimizer_params": {"lr": 1.5e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
        {
            "name": "sparsemax",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 5,
            "gamma": 1.4,
            "lambda_sparse": 1e-4,
            "momentum": 0.02,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 1,
            "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
            "mask_type": "sparsemax",
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
    ]


def build_extended_search_space(batch_size: int, virtual_batch_size: int) -> list[dict]:
    scheduler = {
        "name": "step_lr",
        "params": {
            "step_size": 25,
            "gamma": 0.9,
        },
    }
    return [
        {
            "name": "baseline_sched",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-5,
            "momentum": 0.03,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.5e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
        {
            "name": "baseline_seed7",
            "seed_offset": 7,
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-5,
            "momentum": 0.03,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.5e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
        {
            "name": "baseline_seed21",
            "seed_offset": 21,
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-5,
            "momentum": 0.03,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.5e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
        {
            "name": "deeper_entmax",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 5,
            "gamma": 1.25,
            "lambda_sparse": 1e-5,
            "momentum": 0.03,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.2e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
        },
        {
            "name": "shared_blocks",
            "n_d": 20,
            "n_a": 20,
            "n_steps": 4,
            "gamma": 1.35,
            "lambda_sparse": 5e-6,
            "momentum": 0.02,
            "n_shared": 3,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.0e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
            "cat_emb_dim": [3, 2, 3, 2, 6, 2],
        },
        {
            "name": "lower_batch",
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-5,
            "momentum": 0.03,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.5e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": 512,
            "virtual_batch_size": 64,
        },
        {
            "name": "higher_embed",
            "n_d": 18,
            "n_a": 18,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 5e-6,
            "momentum": 0.02,
            "n_shared": 2,
            "n_independent": 2,
            "clip_value": 2,
            "optimizer_params": {"lr": 1.2e-2, "weight_decay": 1e-5},
            "mask_type": "entmax",
            "scheduler": scheduler,
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
            "cat_emb_dim": [4, 3, 4, 3, 8, 3],
        },
    ]


def resolve_search_space(search_mode: str, batch_size: int, virtual_batch_size: int) -> list[dict]:
    if search_mode == "extended":
        return build_extended_search_space(batch_size, virtual_batch_size)
    return build_search_space(batch_size, virtual_batch_size)


def build_scheduler(config: dict):
    scheduler_config = config.get("scheduler")
    if not scheduler_config:
        return None, None
    if scheduler_config["name"] == "step_lr":
        return torch.optim.lr_scheduler.StepLR, scheduler_config["params"]
    if scheduler_config["name"] == "exponential_lr":
        return torch.optim.lr_scheduler.ExponentialLR, scheduler_config["params"]
    raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")


def make_classifier(metadata: dict, seed: int, config: dict) -> TabNetClassifier:
    scheduler_fn, scheduler_params = build_scheduler(config)
    return TabNetClassifier(
        cat_idxs=metadata["categorical_indices"],
        cat_dims=metadata["categorical_dims"],
        cat_emb_dim=config.get("cat_emb_dim", metadata["suggested_cat_emb_dims"]),
        n_d=config["n_d"],
        n_a=config["n_a"],
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        n_independent=config.get("n_independent", 2),
        n_shared=config.get("n_shared", 2),
        momentum=config.get("momentum", 0.02),
        lambda_sparse=config["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=config["optimizer_params"],
        scheduler_fn=scheduler_fn,
        scheduler_params=scheduler_params or {},
        clip_value=config.get("clip_value", 1),
        mask_type=config["mask_type"],
        device_name=config.get("device_name", "auto"),
        seed=seed,
        verbose=1,
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=["False", "True"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_feature_importance(
    model: TabNetClassifier,
    feature_columns: list[str],
    destination: Path,
) -> None:
    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(destination, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata = load_metadata(args.data_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    x_train, y_train, train_frame = load_split(
        args.data_dir / "tabnet_train_split.csv",
        feature_columns,
        target_column,
    )
    x_valid, y_valid, valid_frame = load_split(
        args.data_dir / "tabnet_valid_split.csv",
        feature_columns,
        target_column,
    )
    x_full, y_full, full_frame = load_split(
        args.data_dir / "tabnet_full_train.csv",
        feature_columns,
        target_column,
    )
    x_test, _, test_frame = load_split(
        args.data_dir / "tabnet_test.csv",
        feature_columns,
        None,
    )

    search_space = resolve_search_space(
        args.search_mode,
        args.batch_size,
        args.virtual_batch_size,
    )
    trial_results = []
    best_result = None

    for idx, config in enumerate(search_space, start=1):
        config["device_name"] = args.device_name
        print(f"\n=== Trial {idx}: {config['name']} ===")
        trial_seed = args.seed + int(config.get("seed_offset", 0))
        model = make_classifier(metadata, trial_seed, config)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_name=["train", "valid"],
            eval_metric=["accuracy"],
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=config["batch_size"],
            virtual_batch_size=config["virtual_batch_size"],
            num_workers=args.num_workers,
            drop_last=False,
        )

        valid_pred = model.predict(x_valid)
        metrics = evaluate_predictions(y_valid, valid_pred)
        best_epoch = int(getattr(model, "best_epoch", args.max_epochs - 1))
        result = {
            "trial_name": config["name"],
            "trial_seed": trial_seed,
            "config": config,
            "best_epoch": best_epoch,
            "metrics": metrics,
        }
        print(
            f"Validation accuracy for {config['name']}: "
            f"{metrics['accuracy']:.6f} at epoch {best_epoch}"
        )

        result_dir = output_dir / f"trial_{idx}_{config['name']}"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        pd.DataFrame(
            {
                "PassengerId": valid_frame["PassengerId"],
                "y_true": y_valid,
                "y_pred": valid_pred,
            }
        ).to_csv(result_dir / "valid_predictions.csv", index=False)
        save_feature_importance(model, feature_columns, result_dir / "feature_importance.csv")
        saved_path = model.save_model(str(result_dir / "tabnet_model"))
        result["saved_model"] = saved_path

        trial_results.append(result)
        if best_result is None or metrics["accuracy"] > best_result["metrics"]["accuracy"]:
            best_result = result

    assert best_result is not None
    summary = {
        "best_trial": best_result["trial_name"],
        "best_accuracy": best_result["metrics"]["accuracy"],
        "trials": trial_results,
    }
    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\nBest trial: {best_result['trial_name']} "
        f"with validation accuracy {best_result['metrics']['accuracy']:.6f}"
    )

    if args.skip_final_train:
        return

    final_config = copy.deepcopy(best_result["config"])
    final_epochs = max(1, int(best_result["best_epoch"]) + 1)
    print(f"\n=== Final training on full dataset for {final_epochs} epochs ===")
    final_model = make_classifier(metadata, best_result["trial_seed"], final_config)
    final_model.fit(
        x_full,
        y_full,
        max_epochs=final_epochs,
        patience=0,
        batch_size=final_config["batch_size"],
        virtual_batch_size=final_config["virtual_batch_size"],
        num_workers=args.num_workers,
        drop_last=False,
    )

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_saved_path = final_model.save_model(str(final_dir / "tabnet_final_model"))
    save_feature_importance(final_model, feature_columns, final_dir / "feature_importance.csv")

    test_pred = final_model.predict(x_test)
    submission = pd.DataFrame(
        {
            "PassengerId": test_frame["PassengerId"],
            "Transported": [bool(value) for value in test_pred],
        }
    )
    submission_path = final_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    final_summary = {
        "chosen_trial": best_result["trial_name"],
        "validation_accuracy": best_result["metrics"]["accuracy"],
        "final_epochs": final_epochs,
        "saved_model": final_saved_path,
        "submission_path": str(submission_path),
        "full_train_rows": int(len(full_frame)),
        "test_rows": int(len(test_frame)),
    }
    (final_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2))
    print(f"Saved final model to {final_saved_path}")
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
