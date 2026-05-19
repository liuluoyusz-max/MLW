#!/usr/bin/env python3
"""Prepare the compact feature-engineered TabNet dataset.

The added features are raw-data transformations validated through GroupCV.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from prepare_tabnet_data import (
    EXPENSE_COLUMNS,
    PASSENGER_ID,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    TRANSPORT_TARGET,
    VALID_RATIO,
    derive_base_rows,
    read_csv,
    safe_divide,
    stratified_split,
    suggest_embedding_dim,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "tabnet_preprocessed_fe_signal_v1"

BASE_CATEGORICAL_COLUMNS = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
]

SIGNAL_CATEGORICAL_COLUMNS = [
    "AgeBin",
    "SpendBin",
    "GroupSizeBin",
    "CabinRegion",
    "HomePlanetDestination",
    "DeckSide",
    "CryoDestination",
]

CATEGORICAL_COLUMNS = BASE_CATEGORICAL_COLUMNS + SIGNAL_CATEGORICAL_COLUMNS

CONTINUOUS_COLUMNS = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "CabinNum",
    "GroupSize",
    "FamilySize",
    "TotalSpend",
    "SpendPerGroup",
    "ServiceCount",
    "LuxurySpend",
    "BasicSpend",
    "LuxuryRatio",
    "BasicRatio",
    "GroupAvgSpend",
    "GroupNoSpendRate",
]

BINARY_NUMERIC_COLUMNS = [
    "AgeMissing",
    "CabinMissing",
    "NoSpending",
    "CryoSleepMissing",
    "VIPMissing",
    "IsChild",
    "IsSenior",
    "CryoAndNoSpend",
    "AwakeAndNoSpend",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + BINARY_NUMERIC_COLUMNS


def age_bin(age: Optional[float]) -> str:
    if age is None:
        return "Unknown"
    if age < 13:
        return "Child"
    if age < 18:
        return "Teen"
    if age < 30:
        return "YoungAdult"
    if age < 45:
        return "Adult"
    if age < 60:
        return "MiddleAge"
    return "Senior"


def spend_bin(total_spend: float) -> str:
    if total_spend == 0:
        return "zero"
    if total_spend < 500:
        return "low"
    if total_spend < 1000:
        return "medium_low"
    if total_spend < 2000:
        return "medium_high"
    if total_spend < 5000:
        return "high"
    return "very_high"


def group_size_bin(group_size: float) -> str:
    if group_size <= 1:
        return "solo"
    if group_size == 2:
        return "pair"
    if group_size <= 4:
        return "small_group"
    return "large_group"


def cabin_region(cabin_num: Optional[float]) -> str:
    if cabin_num is None:
        return "Unknown"
    if cabin_num < 300:
        return "000_299"
    if cabin_num < 600:
        return "300_599"
    if cabin_num < 900:
        return "600_899"
    if cabin_num < 1200:
        return "900_1199"
    if cabin_num < 1500:
        return "1200_1499"
    return "1500_plus"


def add_signal_group_features(rows: List[Dict[str, object]]) -> None:
    group_counts = Counter(str(row["GroupId"]) for row in rows)
    group_total_spend: Dict[str, float] = defaultdict(float)
    group_no_spend_count: Dict[str, int] = defaultdict(int)

    for row in rows:
        group_id = str(row["GroupId"])
        group_total_spend[group_id] += float(row["TotalSpend"])
        group_no_spend_count[group_id] += int(row["NoSpending"])

    for row in rows:
        group_id = str(row["GroupId"])
        group_size = float(group_counts.get(group_id, 1))
        row["GroupAvgSpend"] = safe_divide(group_total_spend[group_id], group_size)
        row["GroupNoSpendRate"] = safe_divide(float(group_no_spend_count[group_id]), group_size)


def add_signal_features(rows: List[Dict[str, object]]) -> None:
    add_signal_group_features(rows)

    for row in rows:
        age = row["Age"]
        total_spend = float(row["TotalSpend"])
        group_size = float(row["GroupSize"])
        room_service = 0.0 if row["RoomService"] is None else float(row["RoomService"])
        food_court = 0.0 if row["FoodCourt"] is None else float(row["FoodCourt"])
        shopping_mall = 0.0 if row["ShoppingMall"] is None else float(row["ShoppingMall"])
        spa = 0.0 if row["Spa"] is None else float(row["Spa"])
        vr_deck = 0.0 if row["VRDeck"] is None else float(row["VRDeck"])

        luxury_spend = food_court + spa + vr_deck
        basic_spend = room_service + shopping_mall

        row["AgeBin"] = age_bin(age)
        row["SpendBin"] = spend_bin(total_spend)
        row["GroupSizeBin"] = group_size_bin(group_size)
        row["CabinRegion"] = cabin_region(row["CabinNum"])
        row["HomePlanetDestination"] = f"{row['HomePlanet']}__{row['Destination']}"
        row["DeckSide"] = f"{row['Deck']}__{row['Side']}"
        row["CryoDestination"] = f"{row['CryoSleep']}__{row['Destination']}"
        row["LuxurySpend"] = luxury_spend
        row["BasicSpend"] = basic_spend
        row["LuxuryRatio"] = safe_divide(luxury_spend, total_spend)
        row["BasicRatio"] = safe_divide(basic_spend, total_spend)
        row["CryoAndNoSpend"] = 1 if row["CryoSleep"] == "True" and int(row["NoSpending"]) == 1 else 0
        row["AwakeAndNoSpend"] = 1 if row["CryoSleep"] == "False" and int(row["NoSpending"]) == 1 else 0


def median(values: Iterable[Optional[float]]) -> float:
    valid = [value for value in values if value is not None]
    if not valid:
        return 0.0
    return float(statistics.median(valid))


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def std(values: Iterable[float], average: float) -> float:
    values_list = list(values)
    if not values_list:
        return 1.0
    variance = sum((value - average) ** 2 for value in values_list) / len(values_list)
    result = math.sqrt(variance)
    return result if result > 0 else 1.0


def compute_numeric_stats(train_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    medians = {column: median(row[column] for row in train_rows) for column in CONTINUOUS_COLUMNS}
    filled_rows: List[Dict[str, float]] = []
    for row in train_rows:
        filled_row = {}
        for column in CONTINUOUS_COLUMNS:
            value = row[column]
            filled_row[column] = medians[column] if value is None else float(value)
        filled_rows.append(filled_row)

    means = {column: mean(row[column] for row in filled_rows) for column in CONTINUOUS_COLUMNS}
    stds = {column: std((row[column] for row in filled_rows), means[column]) for column in CONTINUOUS_COLUMNS}
    return {"medians": medians, "means": means, "stds": stds}


def compute_category_maps(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    category_maps: Dict[str, Dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        categories = sorted({str(row[column]) for row in rows})
        category_maps[column] = {category: idx for idx, category in enumerate(categories)}
    return category_maps


def encode_row(
    row: Dict[str, object],
    numeric_stats: Dict[str, Dict[str, float]],
    category_maps: Dict[str, Dict[str, int]],
) -> Dict[str, object]:
    encoded: Dict[str, object] = {PASSENGER_ID: row[PASSENGER_ID]}

    for column in CATEGORICAL_COLUMNS:
        encoded[column] = category_maps[column][str(row[column])]

    for column in CONTINUOUS_COLUMNS:
        raw_value = row[column]
        filled = numeric_stats["medians"][column] if raw_value is None else float(raw_value)
        encoded[column] = round((filled - numeric_stats["means"][column]) / numeric_stats["stds"][column], 6)

    for column in BINARY_NUMERIC_COLUMNS:
        encoded[column] = int(row[column])

    if row["split"] == "train":
        encoded[TRANSPORT_TARGET] = 1 if str(row["Target"]) == "True" else 0

    return encoded


def main() -> None:
    train_rows_raw = read_csv(TRAIN_PATH)
    test_rows_raw = read_csv(TEST_PATH)

    train_rows = derive_base_rows(train_rows_raw, "train")
    test_rows = derive_base_rows(test_rows_raw, "test")
    combined_rows = train_rows + test_rows

    from prepare_tabnet_data import add_group_features

    add_group_features(combined_rows)
    add_signal_features(combined_rows)

    numeric_stats = compute_numeric_stats(train_rows)
    category_maps = compute_category_maps(combined_rows)

    encoded_train = [encode_row(row, numeric_stats, category_maps) for row in train_rows]
    encoded_test = [encode_row(row, numeric_stats, category_maps) for row in test_rows]

    split_map = stratified_split(encoded_train, VALID_RATIO, SEED)
    train_split_rows = []
    valid_split_rows = []
    full_train_rows = []
    for row in encoded_train:
        split_name = split_map[row[PASSENGER_ID]]
        output_row = dict(row)
        output_row["split"] = split_name
        full_train_rows.append(output_row)
        if split_name == "train":
            train_split_rows.append(output_row)
        else:
            valid_split_rows.append(output_row)

    feature_with_id = [PASSENGER_ID] + FEATURE_COLUMNS
    train_fieldnames = feature_with_id + [TRANSPORT_TARGET]
    split_fieldnames = feature_with_id + [TRANSPORT_TARGET, "split"]

    write_csv(OUTPUT_DIR / "tabnet_full_train.csv", full_train_rows, split_fieldnames)
    write_csv(OUTPUT_DIR / "tabnet_train_split.csv", train_split_rows, split_fieldnames)
    write_csv(OUTPUT_DIR / "tabnet_valid_split.csv", valid_split_rows, split_fieldnames)
    write_csv(OUTPUT_DIR / "tabnet_test.csv", encoded_test, feature_with_id)

    metadata = {
        "seed": SEED,
        "valid_ratio": VALID_RATIO,
        "train_rows": len(encoded_train),
        "test_rows": len(encoded_test),
        "train_split_rows": len(train_split_rows),
        "valid_split_rows": len(valid_split_rows),
        "id_column": PASSENGER_ID,
        "target_column": TRANSPORT_TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "continuous_columns": CONTINUOUS_COLUMNS,
        "binary_numeric_columns": BINARY_NUMERIC_COLUMNS,
        "categorical_indices": [FEATURE_COLUMNS.index(column) for column in CATEGORICAL_COLUMNS],
        "categorical_dims": [len(category_maps[column]) for column in CATEGORICAL_COLUMNS],
        "suggested_cat_emb_dims": [suggest_embedding_dim(len(category_maps[column])) for column in CATEGORICAL_COLUMNS],
        "category_maps": category_maps,
        "numeric_stats": numeric_stats,
        "notes": [
            "Compact feature-engineered dataset built from raw train/test columns.",
            "Use GroupCV OOF to decide whether these raw-data features improve TabNet ability.",
        ],
    }
    with (OUTPUT_DIR / "tabnet_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with (OUTPUT_DIR / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# TabNet FE Signal v1\n\n"
            "Compact pure-feature dataset for TabNet. It adds "
            "raw-data transformations such as age/spend/group bins, deck-side interactions, "
            "cryo-destination interaction, and spend-ratio/group-spend features.\n"
        )

    print("TabNet FE signal v1 preprocessing complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train rows: {len(encoded_train)}")
    print(f"Validation rows: {len(valid_split_rows)}")
    print(f"Test rows: {len(encoded_test)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Categorical columns: {len(CATEGORICAL_COLUMNS)}")
    print(f"Continuous columns: {len(CONTINUOUS_COLUMNS)}")
    print(f"Binary numeric columns: {len(BINARY_NUMERIC_COLUMNS)}")


if __name__ == "__main__":
    main()
