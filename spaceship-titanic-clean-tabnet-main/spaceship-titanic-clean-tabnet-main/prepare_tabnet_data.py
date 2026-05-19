#!/usr/bin/env python3
"""Prepare Spaceship Titanic data for TabNet training.

This script avoids external dependencies so it can run in the current
environment. It creates TabNet-friendly CSV files plus metadata describing
categorical columns and their cardinalities.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
OUTPUT_DIR = ROOT / "tabnet_preprocessed"
SEED = 42
VALID_RATIO = 0.2

TRANSPORT_TARGET = "Transported"
PASSENGER_ID = "PassengerId"

EXPENSE_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

CATEGORICAL_COLUMNS = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
]

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
]

BINARY_NUMERIC_COLUMNS = [
    "AgeMissing",
    "CabinMissing",
    "NoSpending",
    "CryoSleepMissing",
    "VIPMissing",
    "IsChild",
    "IsSenior",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + BINARY_NUMERIC_COLUMNS


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_bool_string(value: str) -> str:
    if value in {"True", "False"}:
        return value
    return "Unknown"


def derive_base_rows(rows: List[Dict[str, str]], split_name: str) -> List[Dict[str, object]]:
    derived_rows: List[Dict[str, object]] = []
    for row in rows:
        passenger_id = row[PASSENGER_ID]
        if "_" in passenger_id:
            group_id, group_member = passenger_id.split("_", 1)
        else:
            group_id, group_member = passenger_id, "0"

        cabin_missing = 1 if row.get("Cabin", "") == "" else 0
        deck = "Unknown"
        side = "Unknown"
        cabin_num = None
        if row.get("Cabin"):
            cabin_parts = row["Cabin"].split("/")
            if len(cabin_parts) == 3:
                deck = cabin_parts[0] or "Unknown"
                side = cabin_parts[2] or "Unknown"
                cabin_num = parse_float(cabin_parts[1])

        surname = "Unknown"
        name = row.get("Name", "")
        if name:
            surname = name.strip().split(" ")[-1] or "Unknown"

        spend_values = {}
        spend_missing_count = 0
        total_spend = 0.0
        service_count = 0
        for column in EXPENSE_COLUMNS:
            value = parse_float(row.get(column, ""))
            spend_values[column] = value
            if value is None:
                spend_missing_count += 1
            else:
                total_spend += value
                if value > 0:
                    service_count += 1

        age = parse_float(row.get("Age", ""))
        age_missing = 1 if age is None else 0

        derived_rows.append(
            {
                "split": split_name,
                PASSENGER_ID: passenger_id,
                "Target": row.get(TRANSPORT_TARGET, ""),
                "GroupId": group_id,
                "GroupMember": int(group_member),
                "Surname": surname,
                "HomePlanet": row.get("HomePlanet", "") or "Unknown",
                "CryoSleep": normalize_bool_string(row.get("CryoSleep", "")),
                "Destination": row.get("Destination", "") or "Unknown",
                "VIP": normalize_bool_string(row.get("VIP", "")),
                "Deck": deck,
                "Side": side,
                "Age": age,
                "AgeMissing": age_missing,
                "CabinNum": cabin_num,
                "CabinMissing": cabin_missing,
                "CryoSleepMissing": 1 if row.get("CryoSleep", "") == "" else 0,
                "VIPMissing": 1 if row.get("VIP", "") == "" else 0,
                "NoSpending": 1 if total_spend == 0 else 0,
                "SpendMissingCount": spend_missing_count,
                "ServiceCount": service_count,
                "TotalSpend": total_spend,
                "IsChild": 1 if age is not None and age < 18 else 0,
                "IsSenior": 1 if age is not None and age >= 60 else 0,
                **spend_values,
            }
        )
    return derived_rows


def compute_counts(rows: Iterable[Dict[str, object]], key: str, ignore_value: Optional[str] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        if ignore_value is not None and value == ignore_value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


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


def add_group_features(rows: List[Dict[str, object]]) -> None:
    group_counts = compute_counts(rows, "GroupId")
    surname_counts = compute_counts(rows, "Surname", ignore_value="Unknown")
    for row in rows:
        group_size = float(group_counts.get(str(row["GroupId"]), 1))
        family_size = float(surname_counts.get(str(row["Surname"]), 1))
        row["GroupSize"] = group_size
        row["FamilySize"] = family_size
        row["SpendPerGroup"] = safe_divide(float(row["TotalSpend"]), group_size)


def compute_numeric_stats(train_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    medians = {
        column: median(row[column] for row in train_rows)
        for column in CONTINUOUS_COLUMNS
    }
    filled_rows: List[Dict[str, float]] = []
    for row in train_rows:
        filled_row = {}
        for column in CONTINUOUS_COLUMNS:
            value = row[column]
            filled_row[column] = medians[column] if value is None else float(value)
        filled_rows.append(filled_row)

    means = {
        column: mean(filled_row[column] for filled_row in filled_rows)
        for column in CONTINUOUS_COLUMNS
    }
    stds = {
        column: std((filled_row[column] for filled_row in filled_rows), means[column])
        for column in CONTINUOUS_COLUMNS
    }
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
    encoded: Dict[str, object] = {
        PASSENGER_ID: row[PASSENGER_ID],
    }

    for column in CATEGORICAL_COLUMNS:
        encoded[column] = category_maps[column][str(row[column])]

    for column in CONTINUOUS_COLUMNS:
        raw_value = row[column]
        filled = numeric_stats["medians"][column] if raw_value is None else float(raw_value)
        standardized = (filled - numeric_stats["means"][column]) / numeric_stats["stds"][column]
        encoded[column] = round(standardized, 6)

    for column in BINARY_NUMERIC_COLUMNS:
        encoded[column] = int(row[column])

    if row["split"] == "train":
        encoded[TRANSPORT_TARGET] = 1 if str(row["Target"]) == "True" else 0

    return encoded


def stratified_split(rows: List[Dict[str, object]], ratio: float, seed: int) -> Dict[str, str]:
    positives = [row[PASSENGER_ID] for row in rows if row[TRANSPORT_TARGET] == 1]
    negatives = [row[PASSENGER_ID] for row in rows if row[TRANSPORT_TARGET] == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    valid_positive_count = int(round(len(positives) * ratio))
    valid_negative_count = int(round(len(negatives) * ratio))

    valid_ids = set(positives[:valid_positive_count] + negatives[:valid_negative_count])
    return {
        row[PASSENGER_ID]: ("valid" if row[PASSENGER_ID] in valid_ids else "train")
        for row in rows
    }


def suggest_embedding_dim(cardinality: int) -> int:
    return min(32, max(2, (cardinality + 1) // 2))


def main() -> None:
    train_rows_raw = read_csv(TRAIN_PATH)
    test_rows_raw = read_csv(TEST_PATH)

    train_rows = derive_base_rows(train_rows_raw, "train")
    test_rows = derive_base_rows(test_rows_raw, "test")
    combined_rows = train_rows + test_rows

    add_group_features(combined_rows)

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
    }
    with (OUTPUT_DIR / "tabnet_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("TabNet preprocessing complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train rows: {len(encoded_train)}")
    print(f"Validation rows: {len(valid_split_rows)}")
    print(f"Test rows: {len(encoded_test)}")
    print("Categorical columns:", ", ".join(CATEGORICAL_COLUMNS))
    print("Continuous columns:", ", ".join(CONTINUOUS_COLUMNS))


if __name__ == "__main__":
    main()
