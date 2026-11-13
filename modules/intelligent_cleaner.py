
"""Adaptive, explainable cleaning for general-purpose tabular datasets.

Features:
- Profiles numeric, categorical, Boolean and datetime columns.
- Normalises common missing-value tokens.
- Removes exact duplicate rows.
- Standardises categorical values conservatively.
- Selects suitable missing-value imputation methods.
- Flags potential anomalies without deleting records.
- Produces a JSON-safe cleaning summary.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler


MISSING_TOKENS = {
    "", "?", "na", "n/a", "nan", "null",
    "none", "-", "--", "missing"
}


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _safe(value: Any):
    """Convert values into JSON-safe Python objects."""

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            str(key): _safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]

    return str(value)


def _is_numeric(column: Any, series: pd.Series) -> bool:
    """Check whether a column contains numeric data."""
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def _is_identifier(column: Any, series: pd.Series) -> bool:
    """Identify likely ID columns that should not influence imputation or anomaly detection."""

    name = str(column).strip().lower()

    compact_name = name.replace(" ", "").replace("-", "").replace("_", "")
    common_ids = {
        "id", "index", "rowid", "recordid", "customerid", "userid",
        "employeeid", "transactionid", "passengerid", "patientid",
        "accountid", "ticketid", "bookingid", "orderid", "memberid",
        "serialnumber", "serialno", "registrationnumber", "uuid",
    }

    if compact_name in common_ids or name.endswith("_id") or name.endswith(" id"):
        return True

    # High cardinality alone does not make a numeric feature an identifier:
    # continuous measurements (e.g. age, price, temperature) may be nearly unique.
    # Use explicit identifier-like column names instead of dropping such features.
    return False


def _is_target_like(column: Any) -> bool:
    """Recognise common outcome/label names that must not leak into cleaning features."""
    name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
    compact = name.replace("_", "")
    target_names = {
        "target", "label", "classlabel", "outcome", "response", "prediction_target",
        "survived", "defaulted", "churn", "is_fraud", "fraudulent", "diagnosis",
        "loan_status", "readmitted", "purchase_made", "converted",
    }
    return name in target_names or compact in {item.replace("_", "") for item in target_names}


def _is_continuous_feature(column: Any, series: pd.Series) -> bool:
    """Return True only for numeric measurements suitable for distance/outlier analysis."""
    if not _is_numeric(column, series) or _is_identifier(column, series) or _is_target_like(column):
        return False
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.nunique() < 10:
        return False
    # Exclude integer-coded categorical/count fields; keep genuinely continuous measures.
    if pd.api.types.is_integer_dtype(series) and values.nunique() <= 20:
        return False
    return True


# ---------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a dataset profile without modifying the original data."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    columns = []

    for col in df.columns:

        series = df[col]
        non_null = series.dropna()

        if _is_numeric(col, series):
            kind = "numeric"
        elif pd.api.types.is_bool_dtype(series):
            kind = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "datetime"
        else:
            kind = "categorical"

        missing_mask = series.isna().copy()
        if (pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)):
            text_values = series.astype("string").str.strip().str.casefold()
            missing_mask = missing_mask | text_values.isin(MISSING_TOKENS)
        missing_count = int(missing_mask.sum())

        columns.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "kind": kind,
            "missing": missing_count,
            "missing_pct": round(
                float(missing_count / max(len(series), 1) * 100), 2
            ),
            "unique": int(series.nunique(dropna=True)),
            "is_identifier": _is_identifier(col, series),
            "mean": (
                float(non_null.mean())
                if kind == "numeric" and len(non_null)
                else None
            ),
            "median": (
                float(non_null.median())
                if kind == "numeric" and len(non_null)
                else None
            ),
            "skew": (
                float(non_null.skew())
                if kind == "numeric" and len(non_null) > 2
                else None
            ),
        })

    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing_cells = int(sum(
        item["missing"] for item in columns
    ))

    return {
        "shape": [
            int(df.shape[0]),
            int(df.shape[1])
        ],
        "missing_cells": missing_cells,
        "missing_pct": round(
            missing_cells / total_cells * 100, 2
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


# ---------------------------------------------------------
# Missing-value token normalisation
# ---------------------------------------------------------

def _normalise_missing_tokens(
    df: pd.DataFrame,
    actions: list
) -> pd.DataFrame:

    result = df.copy()

    text_columns = result.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in text_columns:

        values = result[col].astype("string").str.strip()

        mask = values.str.casefold().isin(MISSING_TOKENS)

        changed = int(mask.sum())

        if changed > 0:

            result.loc[mask, col] = np.nan

            actions.append({
                "column": str(col),
                "method": "missing-token normalisation",
                "changed": changed,
                "reason": (
                    "Common text placeholders were converted "
                    "into missing values."
                ),
            })

    return result


# ---------------------------------------------------------
# Categorical standardisation
# ---------------------------------------------------------

def _standardise_categories(series: pd.Series) -> pd.Series:
    """Standardise whitespace and case without risky fuzzy matching."""

    result = series.astype("string").str.strip()

    groups = {}

    for value in result.dropna().unique():

        groups.setdefault(
            str(value).casefold(), []
        ).append(str(value))

    for variants in groups.values():

        if len(variants) > 1:

            preferred = sorted(
                variants,
                key=lambda value: (len(value), value)
            )[0]

            result = result.replace({
                value: preferred
                for value in variants
            })

    return result


# ---------------------------------------------------------
# Numeric imputation benchmarking and strategy selection
# ---------------------------------------------------------

def _knn_predictions_for_masked_values(
    numeric_frame: pd.DataFrame,
    target_column,
    masked_positions: np.ndarray,
) -> np.ndarray:
    """Estimate masked target values using scaled numeric neighbours."""

    # Mask positions are positional offsets; reset indexes because duplicate
    # removal can leave gaps in the source DataFrame index.
    features = numeric_frame.reset_index(drop=True).copy()
    features.loc[masked_positions, target_column] = np.nan

    # Remove columns that cannot contribute to neighbour distances.
    usable = [
        col for col in features.columns
        if features[col].notna().any()
        and features[col].nunique(dropna=True) > 1
    ]
    if target_column not in usable or len(usable) < 2:
        raise ValueError("KNN needs at least two informative numeric columns.")

    features = features[usable].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    imputer = KNNImputer(n_neighbors=min(5, max(1, len(features) - 1)))
    transformed = imputer.fit_transform(scaled)

    # KNNImputer can drop all-empty columns; the target has observed values,
    # so its column remains present.
    target_position = usable.index(target_column)
    return scaler.inverse_transform(transformed)[:, target_position][masked_positions]


def _iterative_predictions_for_masked_values(
    numeric_frame: pd.DataFrame,
    target_column,
    masked_positions: np.ndarray,
    method: str,
    seed: int = 42,
) -> np.ndarray:
    """Predict masked numeric values with MICE or a MissForest-style imputer."""
    # Mask positions are positional offsets; reset indexes because duplicate
    # removal can leave gaps in the source DataFrame index.
    features = numeric_frame.reset_index(drop=True).copy().astype(float)
    features.loc[masked_positions, target_column] = np.nan
    usable = [
        col for col in features.columns
        if features[col].notna().any() and features[col].nunique(dropna=True) > 1
    ]
    if target_column not in usable or len(usable) < 2:
        raise ValueError(f"{method} needs at least two informative numeric columns.")
    features = features[usable]

    if method == "mice":
        estimator = BayesianRidge()
        max_iter = 10
    elif method == "missforest":
        estimator = RandomForestRegressor(
            n_estimators=40,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        )
        max_iter = 5
    else:
        raise ValueError(f"Unsupported iterative imputation method: {method}")

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        initial_strategy="median",
        skip_complete=True,
        random_state=seed,
        tol=1e-3,
    )
    transformed = imputer.fit_transform(features)
    target_position = usable.index(target_column)
    predictions = transformed[masked_positions, target_position]
    if not np.isfinite(predictions).all():
        raise ValueError(f"{method} returned non-finite predictions.")
    return predictions


def _benchmark_numeric_column(
    df: pd.DataFrame,
    target_column,
    seed: int = 42,
    repeats: int = 5,
    excluded_columns: set | None = None,
) -> dict:
    """Compare five numeric imputers using repeated masked-value validation."""
    target_values = df[target_column].reset_index(drop=True)
    observed_positions = np.flatnonzero(target_values.notna().to_numpy())
    observed = target_values.iloc[observed_positions]
    if len(observed) < 30:
        return {
            "status": "insufficient_data", "sample_size": 0, "repeats": 0,
            "reason": "At least 30 observed values are required for a useful masking test.",
        }

    excluded_columns = {str(col) for col in (excluded_columns or set())}
    numeric_columns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if str(col) not in excluded_columns
        and _is_continuous_feature(col, df[col]) and df[col].notna().any()
    ]
    if target_column not in numeric_columns:
        numeric_columns.append(target_column)
    if len(numeric_columns) < 2:
        return {
            "status": "insufficient_features", "sample_size": 0, "repeats": 0,
            "reason": "At least two informative numeric measurements are required to compare all imputers.",
        }

    sample_size = min(max(10, int(round(len(observed) * 0.15))), 200, len(observed) - 5)
    rng = np.random.default_rng(seed)
    method_names = ("mean", "median", "knn", "missforest", "mice")
    collected = {name: [] for name in method_names}
    errors = {name: [] for name in method_names}
    numeric_frame = df[numeric_columns].reset_index(drop=True).copy()

    for repeat in range(max(1, int(repeats))):
        masked_positions = rng.choice(observed_positions, size=sample_size, replace=False)
        expected = target_values.iloc[masked_positions].astype(float).to_numpy()
        training_values = target_values.drop(index=masked_positions)
        predictions_by_method = {
            "mean": np.full(sample_size, float(training_values.mean())),
            "median": np.full(sample_size, float(training_values.median())),
        }
        for method in ("knn", "missforest", "mice"):
            try:
                if method == "knn":
                    predictions_by_method[method] = _knn_predictions_for_masked_values(
                        numeric_frame, target_column, masked_positions
                    )
                else:
                    predictions_by_method[method] = _iterative_predictions_for_masked_values(
                        numeric_frame, target_column, masked_positions, method, seed + repeat
                    )
            except Exception as exc:
                errors[method].append(str(exc))

        for method, predictions in predictions_by_method.items():
            collected[method].append({
                "mae": float(mean_absolute_error(expected, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(expected, predictions))),
            })

    metrics = {}
    for method, results in collected.items():
        if results:
            metrics[method] = {
                "mae": float(np.mean([item["mae"] for item in results])),
                "rmse": float(np.mean([item["rmse"] for item in results])),
                "mae_std": float(np.std([item["mae"] for item in results])),
                "rmse_std": float(np.std([item["rmse"] for item in results])),
            }
        else:
            metrics[method] = {
                "mae": None, "rmse": None, "mae_std": None, "rmse_std": None,
                "unavailable_reason": errors[method][0] if errors[method]
                else "No valid benchmark runs completed.",
            }

    available = [(name, values) for name, values in metrics.items() if values.get("mae") is not None]
    if not available:
        return {"status": "failed", "sample_size": int(sample_size), "repeats": int(repeats),
                "metrics": metrics, "reason": "No imputation method completed validation."}
    best_method, _ = min(available, key=lambda item: (item[1]["mae"], item[1]["rmse"]))
    return {
        "status": "completed",
        "sample_size": int(sample_size),
        "repeats": int(max(1, int(repeats))),
        "total_masked_predictions": int(sample_size * max(1, int(repeats))),
        "metrics": {
            name: {
                "mae": round(values["mae"], 4) if values.get("mae") is not None else None,
                "rmse": round(values["rmse"], 4) if values.get("rmse") is not None else None,
                "mae_std": round(values["mae_std"], 4) if values.get("mae_std") is not None else None,
                "rmse_std": round(values["rmse_std"], 4) if values.get("rmse_std") is not None else None,
                **({"unavailable_reason": values["unavailable_reason"]}
                   if values.get("unavailable_reason") else {}),
            } for name, values in metrics.items()
        },
        "selected_method": best_method,
        "selection_metric": "lowest mean MAE across repeated masks, with mean RMSE as tie-breaker",
        "reason": (
            f"{best_method.title()} achieved the lowest validation error across "
            f"{max(1, int(repeats))} reproducible masking runs."
        ),
        "predictor_features": [str(col) for col in numeric_columns if col != target_column],
        "excluded_predictor_columns": sorted(set(
            [str(col) for col in df.select_dtypes(include=[np.number]).columns
             if col != target_column and (str(col) in excluded_columns or not _is_continuous_feature(col, df[col]))]
        )),
    }


def _choose_numeric_strategy(
    series: pd.Series,
    rows: int,
    benchmark: dict | None = None,
) -> tuple[str, str]:
    """Choose a strategy based on observed-value masking, not dataset size alone."""

    if benchmark and benchmark.get("status") == "completed":
        method = benchmark.get("selected_method", "median")
        return (
            method,
            f"{method.title()} was selected because it achieved the lowest "
            "MAE on a controlled masking test of known values."
        )

    observed = series.dropna()
    if len(observed) > 2 and abs(float(observed.skew())) > 1:
        return (
            "median",
            "Median was selected as a robust fallback because the distribution is skewed "
            "and a reliable masking benchmark was unavailable."
        )

    return (
        "mean",
        "Mean was selected as a simple fallback because a reliable masking benchmark "
        "was unavailable."
    )


# ---------------------------------------------------------
# Numeric imputation
# ---------------------------------------------------------

def _impute_numeric_columns(
    df: pd.DataFrame,
    actions: list,
    decisions: list,
    warnings_out: list,
    excluded_columns: set | None = None,
) -> pd.DataFrame:
    """Benchmark five numeric imputers per column, then apply the selected method."""

    result = df.copy()
    excluded_columns = {str(col) for col in (excluded_columns or set())}
    numeric_columns = [
        col for col in result.select_dtypes(include=[np.number]).columns
        if result[col].isna().any()
    ]

    for col in numeric_columns:
        missing_count = int(result[col].isna().sum())

        # Numeric binary/low-cardinality columns are categorical codes, not
        # continuous measurements. Use their mode rather than averaging 0/1 labels.
        observed_unique = int(result[col].nunique(dropna=True))
        if observed_unique <= 2:
            mode = result[col].mode(dropna=True)
            if len(mode):
                fill_value = mode.iloc[0]
                result[col] = result[col].fillna(fill_value)
                reason = "Mode was used because this numeric column has at most two observed categories; averaging category codes would be misleading."
                decisions.append({
                    "column": str(col),
                    "selected_method": "mode",
                    "reason": reason,
                    "benchmark": {"status": "not_applicable", "sample_size": 0,
                                   "reason": "Binary numeric columns are treated as categorical labels."},
                })
                actions.append({"column": str(col), "method": "numeric-category mode imputation",
                                "changed": missing_count, "reason": reason})
                continue

        benchmark = _benchmark_numeric_column(
            result, col, excluded_columns=excluded_columns
        )
        strategy, reason = _choose_numeric_strategy(
            result[col], len(result), benchmark
        )

        decision = {
            "column": str(col),
            "selected_method": strategy,
            "reason": reason,
            "benchmark": benchmark,
        }
        decisions.append(decision)

        if strategy in {"mice", "missforest"}:
            numeric_columns_for_iterative = [
                feature for feature in result.select_dtypes(include=[np.number]).columns
                if str(feature) not in excluded_columns
                and _is_continuous_feature(feature, result[feature])
                and result[feature].notna().any()
            ]
            if col not in numeric_columns_for_iterative:
                numeric_columns_for_iterative.append(col)
            try:
                numeric_frame = result[numeric_columns_for_iterative].astype(float)
                missing_positions = np.flatnonzero(numeric_frame[col].isna().to_numpy())
                estimates = _iterative_predictions_for_masked_values(
                    numeric_frame, col, missing_positions, strategy
                )
                result.loc[result[col].isna(), col] = estimates
                actions.append({
                    "column": str(col),
                    "method": f"{strategy.title()} iterative imputation",
                    "changed": missing_count,
                    "reason": reason,
                })
                continue
            except Exception as exc:
                warnings_out.append(
                    f"{strategy.title()} imputation for {col} was unavailable; median fallback was used: {exc}"
                )
                failed_method = strategy.title()
                strategy = "median"
                decision["selected_method"] = strategy
                decision["reason"] = f"{failed_method} could not be applied; median fallback was used."

        if strategy == "knn":
            numeric_columns_for_knn = [
                feature for feature in result.select_dtypes(include=[np.number]).columns
                if str(feature) not in excluded_columns
                and _is_continuous_feature(feature, result[feature])
                and result[feature].notna().any()
            ]
            if col not in numeric_columns_for_knn:
                numeric_columns_for_knn.append(col)

            try:
                numeric_frame = result[numeric_columns_for_knn].astype(float)
                original_missing = numeric_frame[col].isna()
                usable = [
                    feature for feature in numeric_frame.columns
                    if numeric_frame[feature].notna().any()
                    and numeric_frame[feature].nunique(dropna=True) > 1
                ]
                if col not in usable or len(usable) < 2:
                    raise ValueError("Insufficient numeric features for KNN.")

                scaler = StandardScaler()
                scaled = scaler.fit_transform(numeric_frame[usable])
                imputer = KNNImputer(
                    n_neighbors=min(5, max(1, len(result) - 1))
                )
                transformed = imputer.fit_transform(scaled)
                restored = scaler.inverse_transform(transformed)
                target_position = usable.index(col)
                estimates = pd.Series(
                    restored[:, target_position], index=result.index
                )
                result[col] = result[col].fillna(estimates)

                actions.append({
                    "column": str(col),
                    "method": "KNN imputation",
                    "changed": missing_count,
                    "reason": reason,
                })
                continue
            except Exception as exc:
                warnings_out.append(
                    f"KNN imputation for {col} was unavailable; median fallback was used: {exc}"
                )
                strategy = "median"
                decision["selected_method"] = strategy
                decision["reason"] = "KNN could not be applied; median fallback was used."

        if strategy == "mean":
            fill_value = result[col].mean()
            method = "mean imputation"
        else:
            fill_value = result[col].median()
            method = "median imputation"

        if pd.isna(fill_value):
            warnings_out.append(
                f"Column {col} contained no usable numeric values and could not be imputed."
            )
            continue

        result[col] = result[col].fillna(fill_value)
        actions.append({
            "column": str(col),
            "method": method,
            "changed": missing_count,
            "reason": reason if strategy != "median" or benchmark.get("status") != "completed"
                      else f"Median achieved the best masking-test result for this column.",
        })

    return result


# ---------------------------------------------------------
# Categorical, Boolean and datetime imputation
# ---------------------------------------------------------

def _impute_remaining_columns(
    df: pd.DataFrame,
    actions: list
) -> pd.DataFrame:

    result = df.copy()

    for col in result.columns:

        series = result[col]

        if not series.isna().any():
            continue

        count = int(series.isna().sum())

        if pd.api.types.is_bool_dtype(series):

            mode = series.mode(dropna=True)

            fill_value = (
                bool(mode.iloc[0])
                if len(mode)
                else False
            )

            method = "boolean mode imputation"

        elif pd.api.types.is_datetime64_any_dtype(series):

            mode = series.mode(dropna=True)

            fill_value = (
                mode.iloc[0]
                if len(mode)
                else pd.NaT
            )

            method = "datetime mode imputation"

        else:

            missing_fraction = float(series.isna().mean())

            mode = series.mode(dropna=True)

            if missing_fraction >= 0.40:

                fill_value = "Unknown"

                method = "explicit missing-category label"

            elif len(mode):

                fill_value = mode.iloc[0]

                method = "categorical mode imputation"

            else:

                fill_value = "Unknown"

                method = "missing-category fallback"

        result[col] = series.fillna(fill_value)

        actions.append({
            "column": str(col),
            "method": method,
            "changed": count,
            "reason": (
                "A suitable non-numeric missing-value "
                "strategy was applied."
            ),
        })

    return result


# ---------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------

def _detect_group_columns(
    df: pd.DataFrame,
    excluded_columns: set[str] | None = None,
    requested_group_columns: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[list[str], str]:
    """Choose useful low-cardinality context columns for group-wise anomaly checks."""
    excluded = {str(value) for value in (excluded_columns or set())}
    available = [str(col) for col in df.columns]
    requested = [str(col) for col in (requested_group_columns or [])]
    if requested:
        valid = [col for col in requested if col in available and col not in excluded
                 and not _is_identifier(col, df[col]) and not _is_target_like(col)]
        return valid, "user-selected" if valid else "automatic (no valid user-selected grouping column)"

    candidates = []
    preferred_words = ("class", "category", "segment", "region", "department", "product", "type", "group")
    for col in df.columns:
        name = str(col)
        if name in excluded or _is_identifier(name, df[name]) or _is_target_like(name):
            continue
        series = df[name]
        unique = int(series.nunique(dropna=True))
        if 2 <= unique <= min(20, max(2, int(len(df) * 0.10))):
            # Low-cardinality text/categorical and integer-coded groups are valid contexts.
            if (not _is_numeric(name, series)) or pd.api.types.is_integer_dtype(series):
                preferred = any(word in name.lower() for word in preferred_words)
                candidates.append((0 if preferred else 1, unique, name))
    if not candidates:
        return [], "automatic (no suitable low-cardinality grouping column found)"
    candidates.sort()
    # Use the strongest single grouping dimension by default. Users can select
    # multiple columns explicitly when intersections are meaningful.
    return [candidates[0][2]], "automatic"


def _fit_anomaly_candidates(matrix: pd.DataFrame, contamination: float) -> tuple[dict, dict]:
    """Fit several detectors and score their agreement (not true accuracy).

    For large groups, models are fitted on a deterministic sample and then
    applied to the full group where the estimator supports out-of-sample scoring.
    DBSCAN uses distance to fitted core samples for the same purpose.
    """
    n_rows = len(matrix)
    contamination = min(max(float(contamination), 0.005), 0.10)
    scaled = RobustScaler().fit_transform(matrix.astype(float))
    sample_limit = 3000
    if n_rows > sample_limit:
        rng = np.random.default_rng(42)
        sample_positions = np.sort(rng.choice(n_rows, size=sample_limit, replace=False))
    else:
        sample_positions = np.arange(n_rows)
    training = scaled[sample_positions]
    sampled = len(sample_positions) < n_rows
    predictions = {}
    errors = {}

    try:
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
        model.fit(training)
        predictions["isolation_forest"] = model.predict(scaled) == -1
    except Exception as exc:
        errors["isolation_forest"] = str(exc)

    try:
        model = LocalOutlierFactor(
            n_neighbors=min(20, max(2, len(training) - 1)),
            contamination=contamination,
            novelty=True,
        )
        model.fit(training)
        predictions["local_outlier_factor"] = model.predict(scaled) == -1
    except Exception as exc:
        errors["local_outlier_factor"] = str(exc)

    try:
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
        model.fit(training)
        predictions["one_class_svm"] = model.predict(scaled) == -1
    except Exception as exc:
        errors["one_class_svm"] = str(exc)

    try:
        min_samples = min(6, max(3, len(training) - 1))
        distances = NearestNeighbors(n_neighbors=min_samples).fit(training).kneighbors(training)[0]
        eps = max(float(np.quantile(distances[:, -1], 0.90)), 0.25)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(training)
        core_positions = dbscan.core_sample_indices_
        if len(core_positions):
            core_points = training[core_positions]
            nearest = NearestNeighbors(n_neighbors=1).fit(core_points)
            new_distances, _ = nearest.kneighbors(scaled)
            predictions["dbscan"] = new_distances[:, 0] > eps
        else:
            predictions["dbscan"] = np.ones(n_rows, dtype=bool)
    except Exception as exc:
        errors["dbscan"] = str(exc)

    names = list(predictions)
    metrics = {}
    for name in names:
        own = predictions[name]
        agreements = []
        for other_name in names:
            if other_name == name:
                continue
            other = predictions[other_name]
            tp = int(np.sum(own & other))
            tn = int(np.sum(~own & ~other))
            fp = int(np.sum(own & ~other))
            fn = int(np.sum(~own & other))
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            agreements.append((sensitivity + specificity) / 2)
        agreement = float(np.mean(agreements)) if agreements else 0.0
        detection_rate = float(own.mean())
        rate_fit = max(0.0, 1.0 - abs(detection_rate - contamination) / max(contamination, 0.05))
        selection_score = 0.80 * agreement + 0.20 * rate_fit
        metrics[name] = {
            "flagged_rows": int(own.sum()),
            "flagged_pct": round(detection_rate * 100, 2),
            "mean_balanced_agreement": round(agreement, 4),
            "configured_rate_fit": round(rate_fit, 4),
            "selection_score": round(selection_score, 4),
            "status": "evaluated",
            "fit_sample_size": int(len(training)),
            "sampled_for_scalability": bool(sampled),
        }

    for name, error in errors.items():
        metrics[name] = {"flagged_rows": None, "flagged_pct": None,
                         "mean_balanced_agreement": None, "configured_rate_fit": None,
                         "selection_score": None, "status": "unavailable",
                         "unavailable_reason": error}

    if not predictions:
        raise RuntimeError("All anomaly detectors failed: " + "; ".join(errors.values()))
    selected = max(
        predictions,
        key=lambda name: (metrics[name]["selection_score"], metrics[name]["mean_balanced_agreement"])
    )
    return predictions, {"metrics": metrics, "selected_method": selected,
                          "selection_basis": "highest consensus agreement with other available detectors, with a soft configured-rate fit",
                          "validation_limitation": "Unsupervised consensus only. Without labelled anomalies, these scores do not measure true precision or recall."}


def _explain_anomaly_rows(
    frame: pd.DataFrame,
    anomaly_mask: pd.Series,
    feature_columns: list,
    limit: int = 100,
) -> list[dict]:
    """Explain flags using largest robust within-group feature deviations."""
    explanations = []
    if not anomaly_mask.any():
        return explanations
    values = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = values.median()
    deviations = (values - medians).abs()
    mad = deviations.median().replace(0, np.nan)
    robust_z = 0.6745 * deviations.div(mad)
    fallback_scale = values.std().replace(0, np.nan)
    fallback_z = deviations.div(fallback_scale)
    robust_z = robust_z.replace([np.inf, -np.inf], np.nan).fillna(fallback_z).fillna(0)
    for row_index in frame.index[anomaly_mask][:limit]:
        row_scores = robust_z.loc[row_index].sort_values(ascending=False)
        top = []
        for feature, score in row_scores.head(3).items():
            if float(score) <= 0:
                continue
            top.append({"feature": str(feature), "deviation_score": round(float(score), 2)})
        explanations.append({
            "row_index": _safe(row_index),
            "reason": ", ".join(
                f"{item['feature']} (robust deviation {item['deviation_score']})" for item in top
            ) or "The combined feature pattern was unusual; no single feature dominated.",
            "contributing_features": top,
        })
    return explanations


def _flag_anomalies(
    df: pd.DataFrame,
    contamination: float,
    actions: list,
    warnings_out: list,
    excluded_columns: set | None = None,
    requested_group_columns: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Compare four anomaly detectors, apply the consensus-selected model, and retain all rows."""
    result = df.copy()
    excluded_columns = {str(col) for col in (excluded_columns or set())}
    result["_auto_clean_anomaly"] = False
    result["_auto_clean_anomaly_reason"] = ""

    anomaly_columns = [
        col for col in result.select_dtypes(include=[np.number]).columns
        if str(col) not in excluded_columns and _is_continuous_feature(col, result[col])
    ]
    if len(result) < 30 or not anomaly_columns:
        warnings_out.append(
            "Anomaly detection was skipped because the dataset does not contain enough rows or continuous numeric measurements."
        )
        return result, 0

    group_columns, grouping_mode = _detect_group_columns(
        result.drop(columns=["_auto_clean_anomaly", "_auto_clean_anomaly_reason"], errors="ignore"),
        excluded_columns,
        requested_group_columns,
    )
    groups = []
    skipped_groups = 0

    def prepare_group(index):
        nonlocal skipped_groups
        if len(index) < 30:
            skipped_groups += 1
            return
        matrix = result.loc[index, anomaly_columns].replace([np.inf, -np.inf], np.nan)
        matrix = matrix.fillna(matrix.median()).fillna(0)
        matrix = matrix.loc[:, matrix.nunique(dropna=True) > 1]
        if matrix.shape[1] < 1:
            skipped_groups += 1
            return
        try:
            predictions, comparison = _fit_anomaly_candidates(matrix, contamination)
            groups.append({"index": list(index), "matrix": matrix,
                           "predictions": predictions, "comparison": comparison})
        except Exception as exc:
            warnings_out.append(f"Anomaly detector comparison failed for one group: {exc}")

    try:
        if group_columns:
            grouped = result.groupby(group_columns, dropna=False, sort=False).groups
            for _, index in grouped.items():
                prepare_group(list(index))
            if not groups:
                warnings_out.append(
                    "No comparison group had at least 30 usable records; global anomaly detection was used instead."
                )
                prepare_group(list(result.index))
                grouping_mode = "global fallback"
                group_columns = []
        else:
            prepare_group(list(result.index))
            grouping_mode = "global (no suitable context column)"

        if not groups:
            raise RuntimeError("No group had enough usable records for anomaly comparison.")

        method_names = sorted({name for group in groups for name in group["predictions"]})
        aggregate = {}
        for name in method_names:
            available_groups = [group for group in groups if name in group["predictions"]]
            weight_total = sum(len(group["index"]) for group in available_groups) or 1
            score = sum(group["comparison"]["metrics"][name]["selection_score"] * len(group["index"])
                        for group in available_groups) / weight_total
            agreement = sum(group["comparison"]["metrics"][name]["mean_balanced_agreement"] * len(group["index"])
                            for group in available_groups) / weight_total
            flagged = sum(int(group["predictions"][name].sum()) for group in available_groups)
            aggregate[name] = {"selection_score": round(float(score), 4),
                               "mean_balanced_agreement": round(float(agreement), 4),
                               "flagged_rows": flagged,
                               "flagged_pct": round(flagged / max(sum(len(g["index"]) for g in available_groups), 1) * 100, 2),
                               "groups_evaluated": len(available_groups)}
        selected_method = max(aggregate, key=lambda name: (aggregate[name]["selection_score"], aggregate[name]["mean_balanced_agreement"]))

        all_explanations = []
        group_sizes = []
        anomaly_mask = pd.Series(False, index=result.index)
        for group in groups:
            if selected_method not in group["predictions"]:
                continue
            group_mask = pd.Series(group["predictions"][selected_method], index=group["index"])
            anomaly_mask.loc[group_mask.index] = group_mask
            group_sizes.append(len(group["index"]))
            all_explanations.extend(_explain_anomaly_rows(
                group["matrix"], group_mask, list(group["matrix"].columns), limit=len(group["index"])
            ))

        result["_auto_clean_anomaly"] = anomaly_mask.astype(bool)
        explanation_map = {str(item["row_index"]): item["reason"] for item in all_explanations}
        result["_auto_clean_anomaly_reason"] = [
            explanation_map.get(str(index), "") if bool(anomaly_mask.loc[index]) else ""
            for index in result.index
        ]
        anomaly_count = int(anomaly_mask.sum())
        actions.append({
            "column": "*",
            "method": f"Automatic anomaly detector selection ({selected_method.replace('_', ' ').title()})",
            "changed": anomaly_count,
            "reason": "Four detectors were compared where available. The selected detector had the strongest agreement with the other detectors; all flagged records were retained for human review.",
        })
        if skipped_groups:
            warnings_out.append(f"{skipped_groups} groups were too small or uninformative for separate anomaly comparison.")
        if anomaly_count:
            warnings_out.append(f"{anomaly_count} unusual rows were flagged and retained for review.")
        warnings_out.append(
            "Anomaly algorithm selection uses unsupervised consensus, not verified anomaly accuracy. Review flagged records before changing or deleting them."
        )

        result.attrs["_anomaly_group_columns"] = group_columns
        result.attrs["_anomaly_grouping_mode"] = grouping_mode
        result.attrs["_anomaly_group_sizes"] = group_sizes
        result.attrs["_anomaly_comparison"] = aggregate
        result.attrs["_anomaly_selected_method"] = selected_method
        result.attrs["_anomaly_explanations"] = all_explanations
        return result, anomaly_count
    except Exception as exc:
        warnings_out.append(f"Anomaly detection comparison was skipped: {exc}")
        result.attrs["_anomaly_group_columns"] = group_columns
        result.attrs["_anomaly_grouping_mode"] = grouping_mode
        result.attrs["_anomaly_group_sizes"] = []
        result.attrs["_anomaly_comparison"] = {}
        result.attrs["_anomaly_selected_method"] = None
        result.attrs["_anomaly_explanations"] = []
        return result, 0


# ---------------------------------------------------------
# Quality score
# ---------------------------------------------------------

def _quality_score(profile: dict, row_count: int) -> dict:
    """Return transparent quality components and an indicative weighted score."""

    cells = max(int(profile["shape"][0] * profile["shape"][1]), 1)
    completeness = max(
        0.0,
        100.0 - (float(profile["missing_cells"]) / cells * 100.0)
    )
    duplicate_free = max(
        0.0,
        100.0 - (
            int(profile["duplicate_rows"]) / max(row_count, 1) * 100.0
        )
    )

    # Preserve the original score formula for comparability:
    # missingness contributes 55% of its penalty and duplicates 25%.
    score = 100.0 - ((100.0 - completeness) * 0.55) - (
        (100.0 - duplicate_free) * 0.25
    )

    return {
        "overall": round(max(0.0, min(100.0, score)), 2),
        "completeness_pct": round(completeness, 2),
        "duplicate_free_pct": round(duplicate_free, 2),
        "components_note": (
            "Completeness and duplicate-free rates are shown separately. "
            "The overall indicative score applies the documented 0.55 and 0.25 "
            "penalty weights. Validity and factual accuracy are not measured."
        ),
    }


# ---------------------------------------------------------
# Main adaptive cleaning function
# ---------------------------------------------------------

def adaptive_clean(
    df: pd.DataFrame,
    contamination: float = 0.03,
    exclude_columns: list[str] | tuple[str, ...] | set[str] | None = None,
    group_columns: list[str] | tuple[str, ...] | set[str] | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a copy of a tabular dataset.

    Returns:
        cleaned_df: Cleaned pandas DataFrame.
        summary: JSON-safe cleaning audit summary.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "The uploaded dataset must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    def report_progress(percent, message):
        if callable(progress_callback):
            progress_callback(int(percent), str(message))

    report_progress(15, 'Dataset loaded; profiling columns...')
    original = df.copy(deep=True)

    actions = []
    warnings_out = []
    decisions = []
    requested_exclusions = {str(column) for column in (exclude_columns or [])}
    unknown_exclusions = sorted(requested_exclusions - {str(column) for column in df.columns})
    if unknown_exclusions:
        warnings_out.append(
            "Requested target/exclusion columns were not found and were ignored: "
            + ", ".join(unknown_exclusions)
        )
    requested_exclusions &= {str(column) for column in df.columns}

    before = profile_dataset(original)

    work = original.copy(deep=True)

    # 1. Normalise common missing-value tokens.
    report_progress(25, 'Normalising missing-value markers...')
    work = _normalise_missing_tokens(
        work,
        actions
    )

    # 2. Convert infinite numeric values into missing values.
    for col in work.columns:

        if _is_numeric(col, work[col]):

            infinity_mask = work[col].isin([
                np.inf,
                -np.inf
            ])

            changed = int(infinity_mask.sum())

            if changed:

                work.loc[
                    infinity_mask,
                    col
                ] = np.nan

                actions.append({
                    "column": str(col),
                    "method": "infinity-to-missing",
                    "changed": changed,
                    "reason": (
                        "Infinite numeric values cannot be "
                        "used reliably in analysis."
                    ),
                })

    # 3. Remove exact duplicate records.
    report_progress(35, 'Checking and removing exact duplicate rows...')
    duplicate_count = int(work.duplicated().sum())

    if duplicate_count:

        work = work.drop_duplicates().copy()

        actions.append({
            "column": "*",
            "method": "exact duplicate removal",
            "changed": duplicate_count,
            "reason": (
                "Rows were exact duplicates across all columns."
            ),
        })

    # 4. Standardise categorical columns.
    report_progress(45, 'Standardising categorical values...')
    for col in work.columns:

        series = work[col]

        is_text = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

        if (
            is_text
            and series.nunique(dropna=True)
            <= max(100, int(len(series) * 0.5))
        ):

            normalised = _standardise_categories(series)

            changed = int(
                (
                    series.fillna("<NA>").astype(str)
                    != normalised.fillna("<NA>").astype(str)
                ).sum()
            )

            if changed:

                work[col] = normalised

                actions.append({
                    "column": str(col),
                    "method": "categorical standardisation",
                    "changed": changed,
                    "reason": (
                        "Whitespace and case variations were "
                        "standardised without ambiguous fuzzy matching."
                    ),
                })

    # 5. Drop columns containing no observed values.
    empty_columns = [
        col
        for col in work.columns
        if work[col].isna().all()
    ]

    if empty_columns:

        work = work.drop(
            columns=empty_columns
        )

        for col in empty_columns:

            actions.append({
                "column": str(col),
                "method": "drop all-missing column",
                "changed": 1,
                "reason": (
                    "The column contained no observed values "
                    "and could not be inferred."
                ),
            })

    # 6. Impute missing numeric values.
    report_progress(55, 'Benchmarking and imputing numeric columns...')
    work = _impute_numeric_columns(
        work,
        actions,
        decisions,
        warnings_out,
        excluded_columns=requested_exclusions,
    )

    # 7. Impute remaining non-numeric values.
    report_progress(70, 'Handling categorical and other missing values...')
    work = _impute_remaining_columns(
        work,
        actions
    )

    # 8. Flag anomalies without deleting rows.
    report_progress(82, 'Detecting contextual anomalies; records will be retained...')
    work, anomaly_count = _flag_anomalies(
        work,
        contamination,
        actions,
        warnings_out,
        excluded_columns=requested_exclusions,
        requested_group_columns=group_columns,
    )

    anomaly_group_columns = list(work.attrs.get("_anomaly_group_columns", []))
    anomaly_grouping_mode = work.attrs.get("_anomaly_grouping_mode", "automatic")
    anomaly_group_sizes = work.attrs.get("_anomaly_group_sizes", [])
    anomaly_comparison = work.attrs.get("_anomaly_comparison", {})
    anomaly_selected_method = work.attrs.get("_anomaly_selected_method")
    anomaly_explanations = work.attrs.get("_anomaly_explanations", [])

    # 9. Validate final result.
    report_progress(94, 'Validating cleaned dataset and preparing audit summary...')
    if len(work) == 0:

        raise ValueError(
            "Automatic cleaning produced no rows. "
            "The original dataset was not modified."
        )

    # Profile excludes the audit flag because it is metadata,
    # not an original dataset feature.
    profile_for_quality = work.drop(
        columns=["_auto_clean_anomaly", "_auto_clean_anomaly_reason"],
        errors="ignore"
    )

    after = profile_dataset(profile_for_quality)

    # Keep the profile dimensions limited to actual dataset columns;
    # the anomaly flag is audit metadata and is excluded from scoring.

    quality_before = _quality_score(before, len(original))
    quality_after = _quality_score(after, len(profile_for_quality))

    flagged_rows = work.index[
        work["_auto_clean_anomaly"].astype(bool)
    ].tolist()
    anomaly_preview_columns = [
        col for col in work.columns
        if col not in {"_auto_clean_anomaly", "_auto_clean_anomaly_reason"}
    ][:8]
    if "_auto_clean_anomaly_reason" in work.columns:
        anomaly_preview_columns.append("_auto_clean_anomaly_reason")
    anomaly_preview = []
    if flagged_rows:
        preview_frame = work.loc[flagged_rows[:100], anomaly_preview_columns]
        for row_index, row in preview_frame.iterrows():
            record = {"row_index": _safe(row_index)}
            record.update({str(key): _safe(value) for key, value in row.items()})
            anomaly_preview.append(record)

    anomaly_pct = round(
        anomaly_count / max(len(work), 1) * 100,
        2
    )

    # 10. Generate audit summary.
    summary = {
        "engine": "Adaptive Intelligent Cleaner",
        "version": "1.3",

        "initial_shape": [
            int(original.shape[0]),
            int(original.shape[1])
        ],

        "final_shape": [
            int(work.shape[0]),
            int(work.shape[1])
        ],

        "before_profile": before,
        "after_profile": after,

        "quality_score_before": quality_before["overall"],
        "quality_score_after": quality_after["overall"],
        "quality_components_before": quality_before,
        "quality_components_after": quality_after,

        "quality_score_note": (
            "Structural Data Quality Score: an indicative score based on completeness "
            "and duplicate-free rate using the documented 0.55 and 0.25 penalty weights. "
            "It is not a complete quality measure and does not assess validity, factual accuracy, "
            "or guarantee that imputed values are correct."
        ),

        "duplicate_rows_removed": duplicate_count,
        "anomaly_rows_flagged": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "anomaly_method": (anomaly_selected_method or "No detector selected").replace("_", " ").title(),
        "anomaly_selected_method": anomaly_selected_method,
        "anomaly_algorithm_comparison": anomaly_comparison,
        "anomaly_algorithm_selection_note": (
            "The selected detector maximised weighted agreement with other available detectors, with a soft preference for the configured detection rate. This is unsupervised consensus, not verified accuracy; labelled anomaly examples are needed to estimate precision and recall."
        ),
        "anomaly_explanations": anomaly_explanations[:500],
        "anomaly_contamination_setting": float(contamination),
        "selected_exclusion_columns": sorted(requested_exclusions),
        "anomaly_method_detail": (f"Contextual {anomaly_selected_method.replace('_', ' ').title()} on continuous numeric measurements only" if anomaly_selected_method and anomaly_group_columns else (f"{anomaly_selected_method.replace('_', ' ').title()} on continuous numeric measurements only" if anomaly_selected_method else "Anomaly detection unavailable")),
        "anomaly_grouping_mode": anomaly_grouping_mode,
        "anomaly_group_columns": anomaly_group_columns,
        "anomaly_group_sizes": anomaly_group_sizes,
        "anomaly_features_used": [
            str(col) for col in work.select_dtypes(include=[np.number]).columns
            if col not in {"_auto_clean_anomaly", "_auto_clean_anomaly_reason"} and str(col) not in requested_exclusions
            and _is_continuous_feature(col, work[col])
        ],
        "anomaly_features_excluded": [
            str(col) for col in work.columns
            if _is_identifier(col, work[col]) or _is_target_like(col)
            or str(col) in requested_exclusions
            or ( _is_numeric(col, work[col]) and not _is_continuous_feature(col, work[col]) )
        ],
        "anomaly_policy_detail": (
            "Identifiers, common target/label columns, binary/ordinal codes and discrete count fields are excluded. "
            "Only continuous numeric measurements are considered; selected target columns are excluded; when suitable grouping columns are available, records are compared within those groups; flagged records remain in the dataset for human review."
        ),
        "anomaly_review_preview": anomaly_preview,
        "anomaly_preview_truncated": anomaly_count > len(anomaly_preview),

        "rows_removed": duplicate_count,

        "missing_category_details": [
            {
                "column": action["column"],
                "values_replaced": action["changed"],
                "replacement": "Unknown",
                "method": action["method"],
                "explanation": "The original value was unavailable. 'Unknown' preserves the row without claiming that the category was absent."
            }
            for action in actions
            if action.get("method") == "explicit missing-category label"
        ],
        "actions": actions,
        "decisions": decisions,
        "warnings": warnings_out,

        "automated": True,

        "valid_rows_preserved_except_exact_duplicates": True,

        "anomaly_policy": (
            "Potential anomalies are flagged and retained. "
            "They are not automatically deleted."
        ),
    }

    report_progress(98, 'Cleaning summary completed; saving results...')
    return work, _safe(summary)
