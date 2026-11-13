
"""Evaluation utilities for adaptive data cleaning."""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


def compare_quality(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Compare basic quality indicators before and after cleaning."""

    def metrics(df):
        cells = max(int(df.shape[0] * df.shape[1]), 1)
        missing = int(df.isna().sum().sum())
        for column in df.select_dtypes(include=["object", "string"]).columns:
            tokens = df[column].astype("string").str.strip().str.casefold()
            missing += int(tokens.isin({"", "?", "na", "n/a", "nan", "null", "none", "-", "--", "missing"}).sum())
        rows = max(len(df), 1)

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_cells": missing,
            "missing_pct": round(missing / cells * 100, 3),
            "duplicate_rows": int(df.duplicated().sum()),
            "completeness_pct": round(
                100 - (missing / cells * 100), 2
            ),
            "uniqueness_pct": round(
                100 - (
                    int(df.duplicated().sum()) / rows * 100
                ), 2
            )
        }

    before_metrics = metrics(before)
    after_metrics = metrics(after)

    return {
        "before": before_metrics,
        "after": after_metrics,
        "missing_cells_reduced": max(
            0,
            before_metrics["missing_cells"]
            - after_metrics["missing_cells"]
        ),
        "duplicate_rows_reduced": max(
            0,
            before_metrics["duplicate_rows"]
            - after_metrics["duplicate_rows"]
        ),
        "row_retention_pct": round(
            len(after) / max(len(before), 1) * 100,
            2
        )
    }


def evaluate_injected_numeric_noise(cleaner, seed=42) -> dict:
    """Benchmark adaptive imputation against mean and median baselines."""

    rng = np.random.default_rng(seed)

    original = pd.DataFrame({
        "feature_a": rng.normal(50, 10, 250),
        "feature_b": rng.normal(100, 20, 250),
        "category": rng.choice(["A", "B", "C"], 250)
    })

    corrupted = original.copy()
    positions = rng.choice(len(original), size=40, replace=False)
    expected = original.loc[positions, "feature_a"].to_numpy()
    corrupted.loc[positions, "feature_a"] = np.nan

    cleaned, summary = cleaner(corrupted)
    predicted = pd.to_numeric(
        cleaned.loc[positions, "feature_a"], errors="coerce"
    ).to_numpy()

    train_values = original.drop(index=positions)["feature_a"]
    mean_prediction = np.full(len(positions), train_values.mean())
    median_prediction = np.full(len(positions), train_values.median())

    def metrics(values):
        return {
            "mae": round(float(mean_absolute_error(expected, values)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(expected, values))), 4),
        }

    adaptive_metrics = metrics(predicted)
    mean_metrics = metrics(mean_prediction)
    median_metrics = metrics(median_prediction)

    best_baseline_name, best_baseline = min(
        (("mean", mean_metrics), ("median", median_metrics)),
        key=lambda item: (item[1]["mae"], item[1]["rmse"])
    )

    return {
        "test": "synthetic numeric missingness",
        "masked_cells": int(len(positions)),
        "adaptive_method": adaptive_metrics,
        "mean_baseline": mean_metrics,
        "median_baseline": median_metrics,
        "best_simple_baseline": best_baseline_name,
        "better_than_best_baseline": bool(
            (adaptive_metrics["mae"], adaptive_metrics["rmse"])
            < (best_baseline["mae"], best_baseline["rmse"])
        ),
        # Retain the old keys for compatibility with existing templates/reports.
        "imputation_mae": adaptive_metrics["mae"],
        "imputation_rmse": adaptive_metrics["rmse"],
        "median_baseline_mae": median_metrics["mae"],
        "median_baseline_rmse": median_metrics["rmse"],
        "better_than_median": bool(
            adaptive_metrics["mae"] < median_metrics["mae"]
        ),
        "note": (
            "Synthetic benchmark only. It compares adaptive imputation with mean "
            "and median baselines on artificially masked values. Results do not "
            "prove factual correctness or guarantee improved predictive-model performance."
        ),
    }
