import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def parse_args():
    # Allow the pipeline to be rerun from the command line on another machine.
    parser = argparse.ArgumentParser(
        description="Reproducible logistic regression ensemble for Kaggle Spaceship Titanic."
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
        help="Directory used to save the submission file and summary logs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top CV parameter settings to average.",
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


def normalize_boolean_text(series):
    # Normalize different boolean spellings into a consistent string representation.
    mapping = {
        True: "True",
        False: "False",
        "True": "True",
        "False": "False",
        "true": "True",
        "false": "False",
    }
    normalized = series.map(mapping)
    return normalized.where(normalized.notna(), series)


def mode_or_nan(series):
    # Return the mode when available; otherwise preserve the missing value.
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return valid.mode().iloc[0]


def fill_by_group_mode(df, target_col, group_col):
    # Fill a categorical column using the most common value inside each passenger group.
    group_modes = df.groupby(group_col)[target_col].transform(mode_or_nan)
    missing_mask = df[target_col].isna()
    df.loc[missing_mask, target_col] = group_modes[missing_mask]


def add_engineered_features(df):
    # Extract structured information from PassengerId, Cabin, and Name.
    df = df.copy()

    passenger_parts = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["GroupId"] = passenger_parts[0]
    df["GroupNumber"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    cabin_parts = df["Cabin"].astype("string").str.split("/", expand=True)
    df["Deck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["Side"] = cabin_parts[2]

    surname = df["Name"].astype("string").str.split().str[-1]
    df["Surname"] = surname
    missing_surname = df["Surname"].isna()
    df.loc[missing_surname, "Surname"] = "Missing_" + df.loc[missing_surname, "PassengerId"].astype(str)

    return df


def preprocess_combined(train_df, test_df):
    # Process train and test together so both sets follow exactly the same feature logic.
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["_dataset"] = "train"
    test_df["_dataset"] = "test"

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = add_engineered_features(combined)

    for column in ["Age", "GroupNumber", "CabinNum", *SPEND_COLUMNS]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    for column in ["CryoSleep", "VIP", "Transported"]:
        if column in combined.columns:
            combined[column] = normalize_boolean_text(combined[column])
            combined[column] = combined[column].replace({"nan": pd.NA})

    # Build group-level context before fitting the model.
    fill_by_group_mode(combined, "Deck", "GroupId")
    fill_by_group_mode(combined, "Side", "GroupId")
    combined["GroupSize"] = combined.groupby("GroupId")["PassengerId"].transform("size")
    combined["FamilySize"] = combined.groupby("Surname")["PassengerId"].transform("size")
    combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)
    combined["CabinGroupSize"] = combined.groupby("Cabin")["PassengerId"].transform("size").fillna(1)

    spend_sum = combined[SPEND_COLUMNS].fillna(0).sum(axis=1)
    combined["TotalSpend"] = spend_sum
    combined["NoSpend"] = (spend_sum == 0).astype(int)
    combined["SpendPerPerson"] = combined["TotalSpend"] / combined["GroupSize"].clip(lower=1)
    combined["SpendPositiveCount"] = combined[SPEND_COLUMNS].fillna(0).gt(0).sum(axis=1)
    combined["LuxurySpend"] = combined["Spa"].fillna(0) + combined["VRDeck"].fillna(0)
    combined["EssentialSpend"] = (
        combined["RoomService"].fillna(0)
        + combined["FoodCourt"].fillna(0)
        + combined["ShoppingMall"].fillna(0)
    )

    # Log transforms make long-tailed spending features easier for a linear model to use.
    for spend_col in SPEND_COLUMNS:
        combined[f"{spend_col}_log1p"] = np.log1p(combined[spend_col].clip(lower=0))
    combined["TotalSpend_log1p"] = np.log1p(combined["TotalSpend"])

    # Recover important categorical values using group-aware rules first.
    fill_by_group_mode(combined, "HomePlanet", "GroupId")
    fill_by_group_mode(combined, "HomePlanet", "Deck")
    fill_by_group_mode(combined, "Destination", "GroupId")

    # CryoSleep is closely tied to spending, so spending is used as a recovery rule.
    cryo_missing = combined["CryoSleep"].isna()
    combined.loc[cryo_missing & combined["NoSpend"].eq(1), "CryoSleep"] = "True"
    combined.loc[cryo_missing & combined["TotalSpend"].gt(0), "CryoSleep"] = "False"
    fill_by_group_mode(combined, "CryoSleep", "GroupId")

    vip_missing = combined["VIP"].isna()
    combined.loc[vip_missing & combined["CryoSleep"].eq("True"), "VIP"] = "False"
    combined.loc[vip_missing & combined["Age"].lt(18), "VIP"] = "False"

    # Passengers in CryoSleep are assumed not to have onboard spending.
    for spend_col in SPEND_COLUMNS:
        sleep_mask = combined["CryoSleep"].eq("True") & combined[spend_col].isna()
        combined.loc[sleep_mask, spend_col] = 0

    # Bucket continuous variables so logistic regression can capture coarse nonlinear effects.
    combined["Deck"] = combined["Deck"].fillna("Unknown")
    combined["Side"] = combined["Side"].fillna("Unknown")
    combined["DeckSide"] = combined["Deck"].astype(str) + "_" + combined["Side"].astype(str)
    combined["AgeBand"] = pd.cut(
        combined["Age"],
        bins=[-0.1, 12, 18, 25, 35, 50, 65, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "MidAge", "Senior", "Elder"],
    ).astype("string").fillna("Unknown")

    combined["CabinNumBand"] = "Unknown"
    cabin_mask = combined["CabinNum"].notna()
    combined.loc[cabin_mask, "CabinNumBand"] = (
        pd.qcut(combined.loc[cabin_mask, "CabinNum"], q=8, duplicates="drop")
        .astype("string")
        .fillna("Unknown")
    )

    combined["TotalSpendBand"] = "Unknown"
    spend_mask = combined["TotalSpend"].notna()
    combined.loc[spend_mask, "TotalSpendBand"] = (
        pd.qcut(combined.loc[spend_mask, "TotalSpend"], q=6, duplicates="drop")
        .astype("string")
        .fillna("Unknown")
    )

    train_processed = combined.loc[combined["_dataset"] == "train"].drop(columns=["_dataset"])
    test_processed = combined.loc[combined["_dataset"] == "test"].drop(columns=["_dataset", "Transported"])
    return train_processed, test_processed


def build_feature_matrix(train_df, test_df):
    # Build the exact design matrix consumed by the model search and ensemble steps.
    train_processed, test_processed = preprocess_combined(train_df, test_df)

    y = train_processed["Transported"].map({"True": 1, "False": 0}).astype(int)

    feature_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "DeckSide",
        "AgeBand",
        "CabinNumBand",
        "TotalSpendBand",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "GroupNumber",
        "CabinNum",
        "GroupSize",
        "FamilySize",
        "IsAlone",
        "CabinGroupSize",
        "TotalSpend",
        "NoSpend",
        "SpendPerPerson",
        "SpendPositiveCount",
        "LuxurySpend",
        "EssentialSpend",
        "RoomService_log1p",
        "FoodCourt_log1p",
        "ShoppingMall_log1p",
        "Spa_log1p",
        "VRDeck_log1p",
        "TotalSpend_log1p",
    ]

    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "DeckSide",
        "AgeBand",
        "CabinNumBand",
        "TotalSpendBand",
    ]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    X_train = train_processed[feature_columns]
    X_test = test_processed[feature_columns]
    test_ids = test_processed["PassengerId"].copy()
    groups = train_processed["GroupId"].copy()
    return X_train, y, X_test, test_ids, groups, categorical_features, numeric_features


def build_pipeline(categorical_features, numeric_features):
    # One-hot encode categories and standardize numeric features before logistic regression.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def class_weight_label(class_weight):
    # Create readable labels for saved summaries and filenames.
    if class_weight is None:
        return "none"
    if class_weight == "balanced":
        return "balanced"
    positive_weight = class_weight.get(1, 1.0)
    return f"w{positive_weight:.2f}".replace(".", "p")


def find_best_threshold(y_true, probabilities, threshold_min, threshold_max, threshold_step):
    # This competition uses accuracy, so the probability threshold directly affects the score.
    thresholds = np.arange(threshold_min, threshold_max + (threshold_step / 2.0), threshold_step)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, preds)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(np.round(threshold, 4))
    return best_threshold, best_accuracy


def collect_candidate_probabilities(X, y, X_test, groups, categorical_features, numeric_features, cv, top_param_dicts):
    # Train several strong candidates and keep their OOF/test probabilities for averaging.
    candidate_rows = []
    test_prob_list = []
    oof_prob_list = []

    for rank, params in enumerate(top_param_dicts, start=1):
        candidate_model = build_pipeline(categorical_features, numeric_features)
        candidate_model.set_params(**params)

        oof_probabilities = cross_val_predict(
            clone(candidate_model),
            X,
            y,
            cv=cv,
            groups=groups,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        default_cv_accuracy = accuracy_score(y, (oof_probabilities >= 0.5).astype(int))

        candidate_model.fit(X, y)
        test_probabilities = candidate_model.predict_proba(X_test)[:, 1]

        oof_prob_list.append(oof_probabilities)
        test_prob_list.append(test_probabilities)
        candidate_rows.append(
            {
                "rank": rank,
                "C": params["model__C"],
                "class_weight": params["model__class_weight"],
                "oof_cv_accuracy_default_050": default_cv_accuracy,
            }
        )

    return pd.DataFrame(candidate_rows), np.vstack(oof_prob_list), np.vstack(test_prob_list)


def main():
    # End-to-end flow: read raw data, search strong candidates, average them, tune the threshold, and export results.
    args = parse_args()
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_submission_path = Path(args.sample_submission_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(sample_submission_path)

    X, y, X_test, test_ids, groups, categorical_features, numeric_features = build_feature_matrix(
        train_df, test_df
    )
    # Group-aware CV reduces leakage between passengers from the same travel group.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    param_grid = {
        "model__C": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "model__class_weight": [None, "balanced", {0: 1.0, 1: 1.02}, {0: 1.0, 1: 1.04}],
    }

    pipeline = build_pipeline(categorical_features, numeric_features)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X, y, groups=groups)

    cv_results = pd.DataFrame(grid_search.cv_results_).sort_values(
        by=["mean_test_score", "rank_test_score"],
        ascending=[False, True],
    )
    top_param_dicts = cv_results["params"].head(args.top_k).tolist()

    candidate_df, oof_prob_matrix, test_prob_matrix = collect_candidate_probabilities(
        X,
        y,
        X_test,
        groups,
        categorical_features,
        numeric_features,
        cv,
        top_param_dicts,
    )
    top_cv_scores = cv_results["mean_test_score"].head(args.top_k).to_numpy()
    ensemble_weights = top_cv_scores / top_cv_scores.sum()

    # Use weighted probability averaging instead of relying on a single parameter setting.
    ensemble_oof_probs = np.average(oof_prob_matrix, axis=0, weights=ensemble_weights)
    ensemble_test_probs = np.average(test_prob_matrix, axis=0, weights=ensemble_weights)
    best_threshold, ensemble_tuned_cv_accuracy = find_best_threshold(
        y,
        ensemble_oof_probs,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    ensemble_default_cv_accuracy = accuracy_score(y, (ensemble_oof_probs >= 0.5).astype(int))

    submission = sample_submission_df.copy()
    submission["PassengerId"] = test_ids.values
    submission["Transported"] = (ensemble_test_probs >= best_threshold)

    submission_path = output_dir / "submission_logreg_reproducible_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    candidate_df["grid_cv_accuracy"] = top_cv_scores
    candidate_df["ensemble_weight"] = ensemble_weights
    candidate_df["class_weight_label"] = candidate_df["class_weight"].apply(class_weight_label)
    candidate_summary_path = output_dir / "logreg_reproducible_candidates.csv"
    candidate_df.to_csv(candidate_summary_path, index=False)

    probe_df = pd.DataFrame(
        {
            "PassengerId": test_ids.values,
            "ensemble_probability": ensemble_test_probs,
        }
    ).sort_values("ensemble_probability", ascending=False)
    probability_path = output_dir / "submission_logreg_reproducible_probabilities.csv"
    probe_df.to_csv(probability_path, index=False)

    summary_lines = [
        "Model: Reproducible Logistic Regression Ensemble",
        "Competition: Spaceship Titanic",
        f"Training file: {train_path.resolve()}",
        f"Test file: {test_path.resolve()}",
        f"Sample submission file: {sample_submission_path.resolve()}",
        f"Train shape: {train_df.shape}",
        f"Test shape: {test_df.shape}",
        f"Feature count before one-hot encoding: {X.shape[1]}",
        "Validation strategy: StratifiedGroupKFold by GroupId",
        f"Grid-search best CV accuracy: {grid_search.best_score_:.6f}",
        f"Grid-search best parameters: {grid_search.best_params_}",
        f"Top-k ensemble size: {args.top_k}",
        f"Ensemble default 0.50 OOF accuracy: {ensemble_default_cv_accuracy:.6f}",
        f"Ensemble tuned threshold: {best_threshold}",
        f"Ensemble tuned OOF accuracy: {ensemble_tuned_cv_accuracy:.6f}",
        f"Submission path: {submission_path.resolve()}",
        f"Candidate summary path: {candidate_summary_path.resolve()}",
        f"Probability dump path: {probability_path.resolve()}",
    ]

    summary_path = output_dir / "logreg_reproducible_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(candidate_df.to_string(index=False))


if __name__ == "__main__":
    main()
