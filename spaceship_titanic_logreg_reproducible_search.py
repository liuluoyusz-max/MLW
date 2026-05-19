import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict


def load_base_module():
    script_path = Path(__file__).with_name("spaceship_titanic_logreg_reproducible.py")
    spec = importlib.util.spec_from_file_location("logreg_reproducible_base", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args():
    # Parse arguments for the script that compares multiple reproducible LR variants.
    parser = argparse.ArgumentParser(
        description="Search reproducible logistic regression variants for Spaceship Titanic."
    )
    parser.add_argument(
        "--train-path",
        default=r"c:\Users\lenovo\Downloads\train.csv",
        help="Path to the raw Kaggle training CSV.",
    )
    parser.add_argument(
        "--test-path",
        default=r"c:\Users\lenovo\Downloads\test.csv",
        help="Path to the raw Kaggle test CSV.",
    )
    parser.add_argument(
        "--sample-submission-path",
        default=r"c:\Users\lenovo\Downloads\sample_submission.csv",
        help="Path to Kaggle's sample submission CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory used to save submissions and logs.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.50,
        help="Minimum threshold scanned on OOF probabilities.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.58,
        help="Maximum threshold scanned on OOF probabilities.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.0025,
        help="Threshold step scanned on OOF probabilities.",
    )
    return parser.parse_args()


def add_feature_set(X_train, X_test, categorical_features, numeric_features, feature_set):
    # Add optional feature bundles on top of the base preprocessing pipeline.
    X_train = X_train.copy()
    X_test = X_test.copy()
    categorical_features = list(categorical_features)
    numeric_features = list(numeric_features)

    if feature_set == "base":
        return X_train, X_test, categorical_features, numeric_features

    if feature_set != "enriched":
        raise ValueError(f"Unsupported feature set: {feature_set}")

    # These interaction features help a linear model use informative category combinations.
    for df in [X_train, X_test]:
        df["HomePlanet_Destination"] = df["HomePlanet"].astype(str) + "_" + df["Destination"].astype(str)
        df["CryoSleep_Deck"] = df["CryoSleep"].astype(str) + "_" + df["Deck"].astype(str)
        df["CryoSleep_Destination"] = df["CryoSleep"].astype(str) + "_" + df["Destination"].astype(str)
        df["HomePlanet_Deck"] = df["HomePlanet"].astype(str) + "_" + df["Deck"].astype(str)
        df["Deck_Destination"] = df["Deck"].astype(str) + "_" + df["Destination"].astype(str)
        df["AgeBand_NoSpend"] = df["AgeBand"].astype(str) + "_" + df["NoSpend"].astype(str)
        df["SpendBand_CryoSleep"] = df["TotalSpendBand"].astype(str) + "_" + df["CryoSleep"].astype(str)
        df["Age_times_NoSpend"] = df["Age"].fillna(0) * df["NoSpend"].fillna(0)
        df["GroupSize_times_NoSpend"] = df["GroupSize"].fillna(0) * df["NoSpend"].fillna(0)
        df["CabinNum_over_GroupSize"] = df["CabinNum"].fillna(0) / df["GroupSize"].replace(0, 1).fillna(1)
        df["LuxurySpend_share"] = df["LuxurySpend"].fillna(0) / (df["TotalSpend"].fillna(0) + 1.0)
        df["EssentialSpend_share"] = df["EssentialSpend"].fillna(0) / (df["TotalSpend"].fillna(0) + 1.0)

    categorical_features.extend(
        [
            "HomePlanet_Destination",
            "CryoSleep_Deck",
            "CryoSleep_Destination",
            "HomePlanet_Deck",
            "Deck_Destination",
            "AgeBand_NoSpend",
            "SpendBand_CryoSleep",
        ]
    )
    numeric_features.extend(
        [
            "Age_times_NoSpend",
            "GroupSize_times_NoSpend",
            "CabinNum_over_GroupSize",
            "LuxurySpend_share",
            "EssentialSpend_share",
        ]
    )
    return X_train, X_test, categorical_features, numeric_features


def evaluate_config(
    X_base,
    y,
    X_test_base,
    test_ids,
    groups,
    categorical_features,
    numeric_features,
    config,
    output_dir,
    sample_submission_df,
    threshold_min,
    threshold_max,
    threshold_step,
):
    # Evaluate one configuration family: feature set, CV seeds, and top-k candidate averaging.
    X_train, X_test, cats, nums = add_feature_set(
        X_base,
        X_test_base,
        categorical_features,
        numeric_features,
        config["feature_set"],
    )

    param_grid = {
        "model__C": config["c_values"],
        "model__class_weight": config["class_weights"],
    }

    oof_prob_list = []
    test_prob_list = []
    model_rows = []

    # Search the best logistic-regression settings under each seed-specific CV split.
    for seed in config["seeds"]:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        pipeline = BASE.build_pipeline(cats, nums)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y, groups=groups)

        cv_results = pd.DataFrame(grid_search.cv_results_).sort_values(
            by=["mean_test_score", "rank_test_score"],
            ascending=[False, True],
        )
        top_params = cv_results["params"].head(config["top_k"]).tolist()

        # Keep several strong candidates instead of a single winner to reduce variance.
        for rank, params in enumerate(top_params, start=1):
            candidate_model = BASE.build_pipeline(cats, nums)
            candidate_model.set_params(**params)
            oof_probabilities = cross_val_predict(
                clone(candidate_model),
                X_train,
                y,
                cv=cv,
                groups=groups,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]
            candidate_model.fit(X_train, y)
            test_probabilities = candidate_model.predict_proba(X_test)[:, 1]

            oof_prob_list.append(oof_probabilities)
            test_prob_list.append(test_probabilities)
            model_rows.append(
                {
                    "config_name": config["name"],
                    "feature_set": config["feature_set"],
                    "seed": seed,
                    "seed_rank": rank,
                    "C": params["model__C"],
                    "class_weight": params["model__class_weight"],
                    "grid_cv_accuracy": cv_results.iloc[rank - 1]["mean_test_score"],
                    "oof_default_050": accuracy_score(y, (oof_probabilities >= 0.5).astype(int)),
                }
            )

    model_df = pd.DataFrame(model_rows).sort_values(
        by=["grid_cv_accuracy", "seed", "seed_rank"],
        ascending=[False, True, True],
    )
    weights = model_df["grid_cv_accuracy"].to_numpy()
    weights = weights / weights.sum()
    model_df["ensemble_weight"] = weights

    # Build the final prediction by weighted probability averaging.
    ensemble_oof_probs = np.average(np.vstack(oof_prob_list), axis=0, weights=weights)
    ensemble_test_probs = np.average(np.vstack(test_prob_list), axis=0, weights=weights)
    best_threshold, tuned_accuracy = BASE.find_best_threshold(
        y,
        ensemble_oof_probs,
        threshold_min,
        threshold_max,
        threshold_step,
    )
    default_accuracy = accuracy_score(y, (ensemble_oof_probs >= 0.5).astype(int))

    submission = sample_submission_df.copy()
    submission["PassengerId"] = test_ids.values
    submission["Transported"] = (ensemble_test_probs >= best_threshold)

    submission_path = output_dir / f"submission_{config['name']}.csv"
    submission.to_csv(submission_path, index=False)

    probability_path = output_dir / f"submission_{config['name']}_probabilities.csv"
    pd.DataFrame(
        {
            "PassengerId": test_ids.values,
            "ensemble_probability": ensemble_test_probs,
        }
    ).to_csv(probability_path, index=False)

    model_df["class_weight_label"] = model_df["class_weight"].apply(BASE.class_weight_label)

    return {
        "name": config["name"],
        "feature_set": config["feature_set"],
        "seeds": ",".join(str(seed) for seed in config["seeds"]),
        "top_k_per_seed": config["top_k"],
        "num_component_models": len(model_df),
        "default_oof_accuracy": default_accuracy,
        "best_threshold": best_threshold,
        "tuned_oof_accuracy": tuned_accuracy,
        "submission_path": str(submission_path.resolve()),
        "probability_path": str(probability_path.resolve()),
        "models": model_df,
    }


def get_configs():
    # Predefined experiment settings used in the final comparison table.
    common_c_values = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    common_class_weights = [None, "balanced", {0: 1.0, 1: 1.02}, {0: 1.0, 1: 1.04}]
    return [
        {
            "name": "logreg_repro_base_top3",
            "feature_set": "base",
            "seeds": [42],
            "top_k": 3,
            "c_values": common_c_values,
            "class_weights": common_class_weights,
        },
        {
            "name": "logreg_repro_base_top4",
            "feature_set": "base",
            "seeds": [42],
            "top_k": 4,
            "c_values": common_c_values,
            "class_weights": common_class_weights,
        },
        {
            "name": "logreg_repro_base_multiseed_top3",
            "feature_set": "base",
            "seeds": [42, 52, 62],
            "top_k": 3,
            "c_values": common_c_values,
            "class_weights": common_class_weights,
        },
        {
            "name": "logreg_repro_enriched_top3",
            "feature_set": "enriched",
            "seeds": [42],
            "top_k": 3,
            "c_values": common_c_values,
            "class_weights": common_class_weights,
        },
        {
            "name": "logreg_repro_enriched_top4",
            "feature_set": "enriched",
            "seeds": [42],
            "top_k": 4,
            "c_values": common_c_values,
            "class_weights": common_class_weights,
        },
        {
            "name": "logreg_repro_enriched_multiseed_top3",
            "feature_set": "enriched",
            "seeds": [42, 52, 62],
            "top_k": 3,
            "c_values": common_c_values,
            "class_weights": common_class_weights,
        },
    ]


def main():
    # Run all predefined configurations and export a compact summary of which one works best.
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_submission_df = pd.read_csv(args.sample_submission_path)

    X_train, y, X_test, test_ids, groups, categorical_features, numeric_features = BASE.build_feature_matrix(
        train_df,
        test_df,
    )

    # Each config corresponds to one clean, rerunnable experiment branch.
    results = []
    all_model_rows = []
    for config in get_configs():
        result = evaluate_config(
            X_train,
            y,
            X_test,
            test_ids,
            groups,
            categorical_features,
            numeric_features,
            config,
            output_dir,
            sample_submission_df,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
        )
        results.append({k: v for k, v in result.items() if k != "models"})
        all_model_rows.append(result["models"])

    summary_df = pd.DataFrame(results).sort_values(
        by=["tuned_oof_accuracy", "default_oof_accuracy"],
        ascending=False,
    )
    summary_path = output_dir / "logreg_reproducible_search_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    model_path = output_dir / "logreg_reproducible_search_models.csv"
    pd.concat(all_model_rows, ignore_index=True).to_csv(model_path, index=False)

    best_row = summary_df.iloc[0]
    best_submission_path = Path(best_row["submission_path"])
    canonical_best_path = output_dir / "submission_logreg_reproducible_search_best.csv"
    pd.read_csv(best_submission_path).to_csv(canonical_best_path, index=False)

    lines = [
        "Model family: Reproducible Logistic Regression Search",
        "Competition: Spaceship Titanic",
        f"Training file: {Path(args.train_path).resolve()}",
        f"Test file: {Path(args.test_path).resolve()}",
        f"Best config name: {best_row['name']}",
        f"Best feature set: {best_row['feature_set']}",
        f"Best config seeds: {best_row['seeds']}",
        f"Best config top-k per seed: {best_row['top_k_per_seed']}",
        f"Best default OOF accuracy: {best_row['default_oof_accuracy']:.6f}",
        f"Best tuned threshold: {best_row['best_threshold']}",
        f"Best tuned OOF accuracy: {best_row['tuned_oof_accuracy']:.6f}",
        f"Best submission path: {canonical_best_path.resolve()}",
        f"Config summary path: {summary_path.resolve()}",
        f"Component model path: {model_path.resolve()}",
    ]
    text_path = output_dir / "logreg_reproducible_search_summary.txt"
    text_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
