# modules/cleaner.py
"""
Data cleaning for the Data Platform.

What changed compared with the previous version
-----------------------------------------------
* Outlier removal no longer wipes out data:
    - bounds are computed once on the original rows (no cascading, where each
      column was measured on rows already thinned by the previous column);
    - only continuous-looking columns are checked (more than
      OUTLIER_MIN_UNIQUE distinct values), so 0/1 flags, ratings and encoded
      categories are never treated as outliers;
    - columns with IQR == 0 or std == 0 are skipped (a constant column used to
      delete every row under the z-score method);
    - missing values are never flagged;
    - if the removal would delete more than MAX_OUTLIER_ROW_FRACTION of the
      rows it is skipped and a warning is returned instead.
* Outliers are removed BEFORE missing values are filled, so imputed values
  don't distort the quartiles.
* The pipeline raises a clear ValueError if cleaning leaves no rows, instead of
  passing an empty frame to the next step. Completely empty columns are dropped.
* Categorical encoding keeps the column (integer codes) but now records the
  code -> label mapping in the summary so class names are not lost, skips
  free-text/identifier-like columns, and leaves date columns alone. The
  training step (ml_pipeline.prepare_features) one-hot encodes these columns
  when `categorical_columns` is passed, so nominal categories are not treated
  as ordered numbers.
* fix_dtypes works with pandas string dtypes, turns placeholders such as "?",
  "N/A", "null" and "-" into real missing values, understands "$1,200"-style
  numbers, and refuses to convert values with leading zeros (ZIP codes,
  phone numbers, codes) to numbers.
* Normalisation skips encoded categorical and protected columns.
* Summary figures are accurate (`duplicate_rows_removed` is the real count,
  `cols_cleaned` counts columns that actually changed) and include warnings.
* `options['protect_columns']` (optional) lists columns, such as the target,
  that must not be encoded, scaled or have outliers removed.

Note on leakage: imputing and scaling here uses statistics from the whole
dataset. ml_pipeline.prepare_features imputes and standardises again using
training rows only, so model evaluation is not affected by the "normalize"
option; it mainly changes the downloaded CSV.
"""
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

VALID_MISSING_STRATEGIES = ('mean', 'median', 'mode', 'drop', 'none')
VALID_OUTLIER_METHODS = ('iqr', 'zscore')

# Outliers
OUTLIER_MIN_UNIQUE = 10          # columns with fewer distinct values are not checked
MAX_OUTLIER_ROW_FRACTION = 0.30  # skip removal if it would delete more than this

# Scaling
SCALE_MIN_UNIQUE = 20            # low-cardinality numerics are usually codes/labels

# Encoding
TEXT_LIKE_MIN_UNIQUE = 50        # more unique values AND ...
TEXT_LIKE_MIN_RATIO = 0.5        # ... a unique/rows ratio above this => free text / IDs
MAX_LABEL_COLUMNS = 20           # summary size limits (the summary is stored in a cookie)
MAX_LABELS_PER_COLUMN = 20
MAX_LABEL_LENGTH = 40
MAX_SUMMARY_ITEMS = 20

# Placeholder strings that really mean "missing"
MISSING_TOKENS = {'', '?', 'na', 'n/a', '#n/a', 'nan', 'null', 'none', '-', '--'}
_LEADING_ZERO = r'^[+-]?0\d'
_FORMATTED_NUMBER = r'^[+-]?[$€£¥]?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dataset(path):
    if str(path).lower().endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _is_text_dtype(series):
    """Plain text columns (object / pandas string), not categorical or numeric."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return False
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _categorical_columns(df):
    """Columns that hold categories: text, pandas category and booleans (not dates)."""
    columns = []
    for column in df.columns:
        series = df[column]
        if (_is_text_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series)):
            columns.append(column)
    return columns


def _cap(values, limit=MAX_SUMMARY_ITEMS):
    return [str(value) for value in list(values)[:limit]]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
def clean_missing(df, strategy='mean'):
    if strategy not in VALID_MISSING_STRATEGIES:
        raise ValueError(
            f"Unknown missing-value strategy '{strategy}'. "
            f"Choose one of: {', '.join(VALID_MISSING_STRATEGIES)}."
        )

    df_cleaned = df.copy()

    # 'none' preserves missing values exactly as they are. The ML pipeline
    # can then impute them inside the training split / CV folds.
    if strategy == 'none':
        return df_cleaned

    if strategy == 'drop':
        # Drop rows where ANY column has missing values
        return df_cleaned.dropna()

    for column in df_cleaned.columns:
        series = df_cleaned[column]
        if not series.isna().any():
            continue

        if _is_numeric(series):
            if strategy == 'mean':
                fill_value = series.mean()
            elif strategy == 'median':
                fill_value = series.median()
            else:  # mode
                modes = series.mode()
                fill_value = modes.iloc[0] if len(modes) else np.nan
            if pd.isna(fill_value):
                continue
            # A nullable integer column can't hold a fractional fill value.
            if pd.api.types.is_integer_dtype(series.dtype) and not float(fill_value).is_integer():
                series = series.astype('float64')
            df_cleaned[column] = series.fillna(fill_value)
        else:
            if isinstance(series.dtype, pd.CategoricalDtype):
                series = series.astype(object)
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if len(modes) else 'Unknown'
            df_cleaned[column] = series.fillna(fill_value)

    return df_cleaned


def remove_duplicates(df):
    return df.drop_duplicates()


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def _outlier_mask(series, method='iqr'):
    """Boolean mask of outlier values; NaN is never an outlier; degenerate spread => none."""
    values = pd.to_numeric(series, errors='coerce').astype('float64')
    none = pd.Series(False, index=series.index)

    if method == 'iqr':
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return none
        return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)

    std = values.std()
    if not np.isfinite(std) or std == 0:
        return none
    return ((values - values.mean()) / std).abs() > 3


def _outlier_columns(df, protect=None):
    """Continuous-looking numeric columns only (not flags, ratings or codes)."""
    protect = set(protect or [])
    return [
        column for column in df.columns
        if column not in protect
        and _is_numeric(df[column])
        and df[column].nunique(dropna=True) > OUTLIER_MIN_UNIQUE
    ]


def detect_outliers_iqr(df):
    """Number of flagged values across continuous numeric columns (IQR rule)."""
    return int(sum(_outlier_mask(df[column], 'iqr').sum() for column in _outlier_columns(df)))


def detect_outliers_zscore(df):
    """Number of flagged values across continuous numeric columns (|z| > 3)."""
    return int(sum(_outlier_mask(df[column], 'zscore').sum() for column in _outlier_columns(df)))


def _remove_outliers_detailed(df, method='iqr', protect=None):
    """
    Returns (frame, info). All columns are judged on the ORIGINAL rows and the
    rows are dropped once, so the result doesn't depend on column order.
    """
    if method not in VALID_OUTLIER_METHODS:
        raise ValueError(
            f"Unknown outlier method '{method}'. Choose one of: {', '.join(VALID_OUTLIER_METHODS)}."
        )

    columns = _outlier_columns(df, protect)
    flagged_columns = []
    combined = pd.Series(False, index=df.index)
    for column in columns:
        mask = _outlier_mask(df[column], method)
        if mask.any():
            flagged_columns.append(column)
            combined |= mask

    removed = int(combined.sum())
    info = {'rows_removed': 0, 'columns': flagged_columns, 'skipped': False, 'reason': None}
    if removed == 0:
        return df, info

    if removed / max(len(df), 1) > MAX_OUTLIER_ROW_FRACTION:
        info['skipped'] = True
        info['reason'] = (
            f'Outlier removal would delete {removed} of {len(df)} rows '
            f'({removed / len(df):.0%}; the limit is {MAX_OUTLIER_ROW_FRACTION:.0%}), '
            'so it was skipped. Try the other method or review the data.'
        )
        return df, info

    info['rows_removed'] = removed
    return df.loc[~combined], info


def remove_outliers(df, method='iqr'):
    """Drop outlier rows (see _remove_outliers_detailed for the rules)."""
    cleaned, _ = _remove_outliers_detailed(df, method=method)
    return cleaned


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
def _convert_text_column(series):
    """
    Returns (new_series, placeholder_count, converted_to_numeric).
    Placeholders become NaN; numbers (including "$1,200") become numeric unless
    the values have leading zeros (ZIP codes, phone numbers, codes).
    """
    text = series.astype('string').str.strip()
    placeholder = text.str.lower().isin(MISSING_TOKENS)
    placeholder_count = int(placeholder.sum())
    cleaned = text.mask(placeholder)
    as_object = cleaned.astype(object).where(cleaned.notna(), np.nan)

    non_null = cleaned.dropna()
    if non_null.empty:
        return as_object, placeholder_count, False

    if non_null.str.match(_LEADING_ZERO).any():
        return as_object, placeholder_count, False

    try:
        return pd.to_numeric(as_object), placeholder_count, True
    except (ValueError, TypeError):
        pass

    if non_null.str.fullmatch(_FORMATTED_NUMBER).all():
        stripped = as_object.str.replace(r'[$€£¥,\s]', '', regex=True)
        try:
            return pd.to_numeric(stripped), placeholder_count, True
        except (ValueError, TypeError):
            pass

    return as_object, placeholder_count, False


def _fix_dtypes_detailed(df):
    df_cleaned = df.copy()
    converted, placeholder_columns, placeholder_cells = [], [], 0
    for column in df_cleaned.columns:
        series = df_cleaned[column]
        if not _is_text_dtype(series):
            continue
        new_series, placeholder_count, was_converted = _convert_text_column(series)
        if not was_converted and placeholder_count == 0:
            continue  # nothing to change; keep the original dtype untouched
        df_cleaned[column] = new_series
        placeholder_cells += placeholder_count
        if placeholder_count:
            placeholder_columns.append(column)
        if was_converted:
            converted.append(column)
    info = {
        'converted': converted,
        'placeholder_columns': placeholder_columns,
        'placeholder_cells': placeholder_cells,
    }
    return df_cleaned, info


def fix_dtypes(df):
    cleaned, _ = _fix_dtypes_detailed(df)
    return cleaned


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_categorical(df, protect=None):
    """
    Replace categorical columns with integer codes (same column names).

    Returns (frame, info) where info has:
      columns - every encoded column
      labels  - {column: [label for code 0, label for code 1, ...]} (size-limited)
      skipped - free-text / identifier-like columns that were left unchanged
    The codes are arbitrary (alphabetical), NOT an ordering. Downstream training
    should one-hot encode them (ml_pipeline.prepare_features(categorical_columns=...)).
    """
    df_encoded = df.copy()
    protect = set(protect or [])
    row_count = max(len(df_encoded), 1)
    encoded, labels, skipped = [], {}, []

    for column in _categorical_columns(df_encoded):
        if column in protect:
            continue
        series = df_encoded[column]
        unique = series.nunique(dropna=True)
        if unique > TEXT_LIKE_MIN_UNIQUE and unique / row_count > TEXT_LIKE_MIN_RATIO:
            skipped.append(column)
            continue

        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(series.astype(str))
        encoded.append(column)
        if (len(labels) < MAX_LABEL_COLUMNS and len(encoder.classes_) <= MAX_LABELS_PER_COLUMN):
            labels[str(column)] = [str(label)[:MAX_LABEL_LENGTH] for label in encoder.classes_]

    return df_encoded, {'columns': encoded, 'labels': labels, 'skipped': skipped}


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def _scale_numeric_detailed(df, exclude=None):
    df_scaled = df.copy()
    exclude = set(exclude or [])
    # Preserve low-cardinality numeric columns because they commonly represent
    # encoded class labels and must not become continuous-looking targets.
    scale_columns = [
        column for column in df_scaled.columns
        if column not in exclude
        and _is_numeric(df_scaled[column])
        and df_scaled[column].nunique(dropna=True) > SCALE_MIN_UNIQUE
    ]
    if scale_columns:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_scaled[scale_columns].astype('float64'))
        df_scaled[scale_columns] = pd.DataFrame(scaled, columns=scale_columns, index=df_scaled.index)
    return df_scaled, scale_columns


def scale_numeric(df, exclude=None):
    scaled, _ = _scale_numeric_detailed(df, exclude)
    return scaled


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def full_clean_pipeline(df, options):
    """
    Run the selected cleaning steps. Returns (cleaned_df, summary).
    Raises ValueError when the input is empty, an option is invalid, or the
    cleaning would leave no rows.

    Order: fix dtypes -> drop empty columns -> duplicates -> outliers ->
    missing values -> encode -> normalise.
    """
    options = options or {}
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError('The dataset is empty.')

    strategy = options.get('missing_strategy', 'mean')
    if strategy not in VALID_MISSING_STRATEGIES:
        raise ValueError(
            f"Unknown missing-value strategy '{strategy}'. "
            f"Choose one of: {', '.join(VALID_MISSING_STRATEGIES)}."
        )
    outlier_method = options.get('outlier_method', 'iqr')
    if options.get('remove_outliers') and outlier_method not in VALID_OUTLIER_METHODS:
        raise ValueError(
            f"Unknown outlier method '{outlier_method}'. Choose one of: {', '.join(VALID_OUTLIER_METHODS)}."
        )
    protect = [column for column in (options.get('protect_columns') or []) if column in df.columns]

    initial_shape = df.shape
    duplicate_rows_detected = int(df.duplicated().sum())
    df_cleaned = df.copy()
    touched = set()
    notes = []
    summary = {}

    # 1. Data types and placeholder values
    if options.get('fix_dtypes'):
        df_cleaned, fix_info = _fix_dtypes_detailed(df_cleaned)
        touched.update(fix_info['converted'])
        touched.update(fix_info['placeholder_columns'])
        summary['dtype_columns_converted'] = _cap(fix_info['converted'])
        summary['placeholders_to_missing'] = fix_info['placeholder_cells']

    # 2. Columns with no data at all can't be cleaned or modelled
    empty_columns = [column for column in df_cleaned.columns if df_cleaned[column].isna().all()]
    if empty_columns:
        if len(empty_columns) == df_cleaned.shape[1]:
            raise ValueError('Every column in the dataset is empty.')
        df_cleaned = df_cleaned.drop(columns=empty_columns)
        notes.append(f"Dropped {len(empty_columns)} completely empty column(s): "
                     f"{', '.join(_cap(empty_columns, 5))}.")
    summary['empty_columns_dropped'] = _cap(empty_columns)

    # 3. Duplicates (after dtype fixing, so "1" and 1 count as equal)
    duplicates_removed = 0
    if options.get('remove_duplicates'):
        before = len(df_cleaned)
        df_cleaned = remove_duplicates(df_cleaned)
        duplicates_removed = before - len(df_cleaned)

    # 4. Outliers (before imputation so filled values don't distort the quartiles)
    outliers_applied = False
    if options.get('remove_outliers', False):
        df_cleaned, outlier_info = _remove_outliers_detailed(
            df_cleaned, method=outlier_method, protect=protect
        )
        outliers_applied = outlier_info['rows_removed'] > 0
        summary['outlier_method'] = outlier_method
        summary['outlier_rows_removed'] = outlier_info['rows_removed']
        summary['outlier_columns'] = _cap(outlier_info['columns'])
        if outlier_info['skipped']:
            notes.append(outlier_info['reason'])
    summary['outliers_removed'] = outliers_applied

    # 5. Missing values
    missing_before = int(df_cleaned.isna().sum().sum())
    columns_with_missing = [column for column in df_cleaned.columns if df_cleaned[column].isna().any()]
    rows_before_missing = len(df_cleaned)
    df_cleaned = clean_missing(df_cleaned, strategy=strategy)
    if strategy == 'drop':
        summary['rows_dropped_for_missing'] = rows_before_missing - len(df_cleaned)
        summary['missing_values_filled'] = 0
        summary['missing_values_left'] = int(df_cleaned.isna().sum().sum())
    elif strategy == 'none':
        summary['rows_dropped_for_missing'] = 0
        summary['missing_values_filled'] = 0
        summary['missing_values_left'] = int(df_cleaned.isna().sum().sum())
        if summary['missing_values_left']:
            notes.append(
                'Missing values were preserved. Model training should impute them '
                'inside the training pipeline; some non-ML exports may still contain blanks.'
            )
    else:
        summary['rows_dropped_for_missing'] = 0
        summary['missing_values_filled'] = missing_before - int(df_cleaned.isna().sum().sum())
        summary['missing_values_left'] = int(df_cleaned.isna().sum().sum())
        touched.update(columns_with_missing)

    if df_cleaned.shape[0] == 0:
        raise ValueError(
            'Cleaning removed every row. Try a different missing-value strategy '
            'or choose a different missing-value strategy or turn off outlier removal.'
        )

    # 6. Encoding
    encoded_columns = []
    encoded_info = {'columns': [], 'labels': {}, 'skipped': []}
    if options.get('encode_categorical'):
        df_cleaned, encoded_info = encode_categorical(df_cleaned, protect=protect)
        encoded_columns = encoded_info['columns']
        touched.update(encoded_columns)
        if encoded_info['skipped']:
            notes.append(
                'Left unencoded because they look like free text or identifiers: '
                + ', '.join(_cap(encoded_info['skipped'], 5)) + '.'
            )

    # 7. Scaling (never touches encoded category codes or protected columns)
    scaled_columns = []
    if options.get('normalize'):
        df_cleaned, scaled_columns = _scale_numeric_detailed(
            df_cleaned, exclude=set(encoded_columns) | set(protect)
        )
        touched.update(scaled_columns)
        notes.append(
            'Normalisation used statistics from the whole dataset. Model training '
            'rescales features using training rows only, so this option mainly '
            'affects the downloaded CSV.'
        )

    df_cleaned = df_cleaned.reset_index(drop=True)

    summary = {
        'initial_shape': initial_shape,
        'final_shape': df_cleaned.shape,
        'rows_removed': initial_shape[0] - df_cleaned.shape[0],
        'columns_removed': initial_shape[1] - df_cleaned.shape[1],
        'duplicate_rows_detected': duplicate_rows_detected,
        'duplicate_rows_removed': duplicates_removed,
        'cols_cleaned': len(touched),
        'missing_strategy': strategy,
        # Every encoded column is listed (the app uses this to exclude them from charts).
        'encoded_columns': [str(column) for column in encoded_columns],
        'encoded_labels': encoded_info['labels'],
        'scaled_columns': _cap(scaled_columns),
        'warnings': notes[:MAX_SUMMARY_ITEMS],
        **summary
    }

    return df_cleaned, summary