# modules/ml_pipeline.py
"""
Machine-learning pipeline for the Data Platform.

What changed compared with the previous version
-----------------------------------------------
* Tie-breaking now returns the actual best model (lowest CV std among models
  whose CV mean is within TIE_TOLERANCE of the top) instead of whichever tied
  model happened to be trained first.
* Cross-validation scores macro-F1 (classification) / R2 (regression), the same
  metrics shown in the UI, runs 5 folds x 3 repeats on small data (1 repeat on
  large data), skips probability calibration while cross-validating SVMs, and
  stops early when a model exceeds CV_TIME_BUDGET_SECONDS.
* Features are imputed (training medians) and standardised (training mean/std)
  AFTER the train/test split, so test data never leaks into preprocessing, and
  distance/gradient based models (SVM, MLP, Logistic Regression) get scaled
  inputs. Coefficient-based importances are now comparable across features.
* SVR and MLP regressors train on a standardised target and their predictions
  are converted back, so their R2/MAE/RMSE are in the original units.
* Unticking every model raises a clear error (it used to train all models), and
  failed/unavailable models are reported through `collect_model_errors`.
* One shared rule decides classification vs regression, with an optional
  override, and it works with pandas string/category dtypes.
* Identifier detection matches modules/eda.py (no more "paid"/"valid" false
  positives) and non-numeric/text-like/datetime columns can't crash training.
* The data quality report ignores rows with a missing target, caps list sizes
  (they are stored in the session cookie), and adds target-leakage, class
  imbalance and small-sample checks.
* Multi-class ROC-AUC (one-vs-rest), confusion matrices that always match the
  class list, probabilities aligned to all classes, permutation importance for
  models without native importances, and JSON-safe hyperparameters.
"""
import logging
import math
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Suppress ConvergenceWarnings
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE_DEFAULT = 0.2

# Cross-validation
CV_FOLDS = 5
CV_REPEATS_SMALL = 3           # repeats when the training set is small
CV_SMALL_ROW_LIMIT = 2000      # above this many training rows, use 1 repeat
CV_TIME_BUDGET_SECONDS = 45    # per model; stop adding folds after this

# Model selection
TIE_TOLERANCE = 0.01

# Feature preparation
MIN_ROWS = 10
MAX_ONE_HOT_LEVELS = 30        # integer-coded categories above this stay numeric
TEXT_LIKE_MIN_UNIQUE = 50      # non-numeric columns with more unique values AND ...
TEXT_LIKE_MIN_RATIO = 0.5      # ... a unique/rows ratio above this are dropped

# Session-cookie friendliness: keep reported lists short
MAX_LIST_ITEMS = 20
MAX_CORRELATION_PAIRS = 10
MAX_WARNING_NAMES = 10
MAX_CORRELATION_COLUMNS = 300

# Permutation importance (used only when a model has no native importances)
PERMUTATION_MAX_ROWS = 1000
PERMUTATION_MAX_FEATURES = 100

# Regressors that need a standardised target to behave well
TARGET_SCALED_MODELS = {'SVM', 'Neural Network'}

CLASSIFICATION_ALIASES = {'Linear Regression': 'Logistic Regression'}
REGRESSION_ALIASES = {'Logistic Regression': 'Linear Regression'}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _names(values, limit=MAX_WARNING_NAMES):
    values = [str(value) for value in values]
    shown = ', '.join(values[:limit])
    return shown + (f' (+{len(values) - limit} more)' if len(values) > limit else '')


def _cap(values, limit=MAX_LIST_ITEMS):
    return [str(value) for value in list(values)[:limit]]


def _take(data, indices):
    return data.iloc[indices] if hasattr(data, 'iloc') else np.asarray(data)[indices]


def _jsonable(value, depth=0):
    """Convert estimator parameters into values Flask's session can store."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (list, tuple, set)) and depth < 2:
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, dict) and depth < 2:
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    return getattr(value, '__name__', type(value).__name__)


def get_hyperparameters(model):
    """
    Return readable, JSON-safe hyperparameters for reporting.

    If ``model`` is a sklearn Pipeline, report the final estimator's actual
    parameters rather than Pipeline wrapper fields such as ``memory``,
    ``steps``, ``transform_input`` and ``verbose``. Include a concise summary
    of the preprocessing steps as context.
    """
    if model is None:
        return {}

    estimator = model
    preprocessing = []

    # A Pipeline exposes named_steps; its final step is the estimator being
    # evaluated. Do not serialize the whole Pipeline's shallow parameters.
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        if steps:
            estimator = steps[-1][1]
            preprocessing = [
                f"{step_name}: {step.__class__.__name__}"
                for step_name, step in steps[:-1]
            ]

    try:
        params = estimator.get_params(deep=False)
    except Exception:
        params = {}

    result = {
        "Estimator": estimator.__class__.__name__,
    }
    if preprocessing:
        result["Preprocessing"] = "; ".join(preprocessing)

    for key, value in params.items():
        result[str(key)] = _jsonable(value)

    return result


def collect_model_errors(results):
    """Return {model name: error message} for every model that did not train."""
    return {name: str(result.get('error', 'Unknown error'))
            for name, result in results.items() if result.get('model') is None}


# ---------------------------------------------------------------------------
# Column classification (kept consistent with modules/eda.py)
# ---------------------------------------------------------------------------
_ID_NAMES = {'id', 'index', 'rowid', 'recordid', 'userid', 'uuid', 'guid', 'key'}
# Matches: id, ID, user_id, user id, customer-id, userId, customerID
# Does NOT match: paid, valid, android, grid, liquid
_ID_PATTERN = re.compile(r'(?:^|[_\s\-.])(?:id|ID|Id)$|[a-z](?:ID|Id)$')


def _identifier_columns(df, target=None):
    """Return columns that are row identifiers rather than predictive features."""
    identifier_columns = []
    row_count = len(df)

    for column in df.columns:
        if target is not None and column == target:
            continue
        name = str(column)
        normalized = ''.join(character for character in name.lower() if character.isalnum())
        series = df[column]

        looks_like_identifier = normalized in _ID_NAMES or bool(_ID_PATTERN.search(name))
        # Names that merely end in "id" only count when the values look unique.
        # Float columns are skipped: continuous measurements are rarely IDs.
        if (not looks_like_identifier and normalized.endswith('id') and row_count > 1
                and not pd.api.types.is_float_dtype(series)):
            looks_like_identifier = series.nunique(dropna=True) / row_count >= 0.9

        is_unique_integer_sequence = (
            row_count > 1
            and pd.api.types.is_integer_dtype(series)
            and series.notna().all()
            and series.nunique() == row_count
            and series.is_monotonic_increasing
        )
        if looks_like_identifier or is_unique_integer_sequence:
            identifier_columns.append(column)

    return identifier_columns


def _unusable_columns(features):
    """Find feature columns that can't (or shouldn't) be used for training."""
    row_count = max(len(features), 1)
    datetime_columns, empty_columns, text_like_columns = [], [], []
    for column in features.columns:
        series = features[column]
        if series.isna().all():
            empty_columns.append(column)
        elif (pd.api.types.is_datetime64_any_dtype(series)
              or pd.api.types.is_timedelta64_dtype(series)):
            datetime_columns.append(column)
        elif not _is_numeric(series) and not pd.api.types.is_bool_dtype(series):
            unique = series.nunique(dropna=True)
            if unique > TEXT_LIKE_MIN_UNIQUE and unique / row_count > TEXT_LIKE_MIN_RATIO:
                text_like_columns.append(column)
    return {
        'empty': empty_columns,
        'datetime': datetime_columns,
        'text_like': text_like_columns,
    }


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------
def data_quality_report(df, target):
    """Find risks before training and identify columns excluded from X."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    df_clean = df.dropna(subset=[target])  # same rows the model will see
    row_count = len(df_clean)
    safe_rows = max(row_count, 1)
    identifier_columns = _identifier_columns(df_clean, target)
    candidates = df_clean.drop(columns=[target, *identifier_columns])
    unusable = _unusable_columns(candidates)
    # Checks below look only at columns the model will actually use.
    excluded = unusable['empty'] + unusable['datetime'] + unusable['text_like']
    candidates = candidates.drop(columns=excluded)

    feature_duplicate_rows = int(candidates.duplicated().sum()) if candidates.shape[1] else 0

    high_cardinality = [
        column for column in candidates.columns
        if not _is_numeric(candidates[column]) and not pd.api.types.is_bool_dtype(candidates[column])
        and candidates[column].nunique(dropna=False) / safe_rows >= 0.9
    ]

    near_constant = []
    for column in candidates.columns:
        shares = candidates[column].value_counts(normalize=True, dropna=False)
        if len(shares) and shares.iloc[0] >= 0.99:
            near_constant.append(column)

    numeric = candidates.select_dtypes(include=[np.number])
    if numeric.shape[1]:
        numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] > MAX_CORRELATION_COLUMNS:
        numeric = numeric.iloc[:, :MAX_CORRELATION_COLUMNS]

    # Highly correlated feature pairs (multicollinearity)
    correlation_pairs, total_pairs = [], 0
    try:
        if numeric.shape[1] >= 2:
            with np.errstate(all='ignore'):
                correlation = numeric.corr().abs()
            upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))
            pairs = upper.stack()
            pairs = pairs[pairs >= 0.9].sort_values(ascending=False)
            total_pairs = len(pairs)
            for (left, right), value in pairs.head(MAX_CORRELATION_PAIRS).items():
                correlation_pairs.append(
                    {'left': str(left), 'right': str(right), 'correlation': float(value)}
                )
    except Exception:
        logger.warning('Correlation check failed', exc_info=True)

    # Possible target leakage: a feature that (almost) equals the target
    leakage_suspects = []
    try:
        target_series = df_clean[target]
        if numeric.shape[1] and row_count >= 3 and (
                _is_numeric(target_series) or target_series.nunique(dropna=True) == 2):
            if _is_numeric(target_series):
                target_numeric = target_series.astype(float)
            else:
                target_numeric = pd.Series(
                    pd.factorize(target_series)[0], index=target_series.index
                ).astype(float)
            if target_numeric.nunique() > 1:
                with np.errstate(all='ignore'):
                    target_corr = numeric.corrwith(target_numeric).abs()
                suspects = target_corr[target_corr >= 0.98].sort_values(ascending=False)
                leakage_suspects = [
                    {'feature': str(name), 'correlation': float(value)}
                    for name, value in suspects.head(MAX_LIST_ITEMS).items()
                ]
    except Exception:
        logger.warning('Leakage check failed', exc_info=True)

    # Class imbalance (classification targets only)
    class_imbalance = None
    singleton_classes = []
    try:
        if row_count and detect_problem_type(df_clean, target) == 'classification':
            counts = df_clean[target].value_counts()
            smallest_label = counts.index[-1]
            smallest_count = int(counts.iloc[-1])
            class_imbalance = {
                'smallest_class': str(smallest_label),
                'smallest_count': smallest_count,
                'smallest_share': float(smallest_count / safe_rows),
            }
            singleton_classes = [str(label) for label, count in counts.items() if count < 2]
    except Exception:
        logger.warning('Class balance check failed', exc_info=True)

    warnings_list = []
    if row_count < 50:
        warnings_list.append(
            f'Only {row_count} rows have a target value; results will be unreliable on so little data.'
        )
    if identifier_columns:
        warnings_list.append(
            f"Identifier/leakage columns excluded from training: {_names(identifier_columns)}"
        )
    if unusable['text_like']:
        warnings_list.append(
            f"Text-like columns excluded from training: {_names(unusable['text_like'])}"
        )
    if unusable['datetime']:
        warnings_list.append(
            f"Date/time columns excluded from training: {_names(unusable['datetime'])}"
        )
    if unusable['empty']:
        warnings_list.append(
            f"Empty columns excluded from training: {_names(unusable['empty'])}"
        )
    if high_cardinality:
        warnings_list.append(f"High-cardinality columns: {_names(high_cardinality)}")
    if near_constant:
        warnings_list.append(f"Near-constant columns: {_names(near_constant)}")
    if total_pairs:
        warnings_list.append(
            f'{total_pairs} highly correlated numeric feature pair(s) (|r| >= 0.90) '
            'may indicate multicollinearity.'
        )
    if leakage_suspects:
        warnings_list.append(
            'Possible target leakage (feature almost identical to the target): '
            + _names([item['feature'] for item in leakage_suspects])
        )
    if class_imbalance and class_imbalance['smallest_share'] < 0.10:
        warnings_list.append(
            f"Class imbalance: the smallest class ('{class_imbalance['smallest_class']}') has "
            f"{class_imbalance['smallest_count']} rows ({class_imbalance['smallest_share']:.1%}). "
            'Accuracy can look high while rare classes are missed; check macro-F1.'
        )
    if singleton_classes:
        warnings_list.append(
            f"Classes with fewer than 2 rows prevent a stratified split: {_names(singleton_classes)}"
        )
    if feature_duplicate_rows:
        warnings_list.append(
            f"Identical feature rows: {feature_duplicate_rows}; duplicates can occur in both train and test splits."
        )
    missing_feature_cells = int(candidates.isna().sum().sum()) if candidates.shape[1] else 0
    if missing_feature_cells:
        warnings_list.append(
            f'{missing_feature_cells} missing feature value(s) will be filled using training-set medians.'
        )

    return {
        'identifier_columns': _cap(identifier_columns),
        'high_cardinality_columns': _cap(high_cardinality),
        'near_constant_columns': _cap(near_constant),
        'high_correlation_pairs': correlation_pairs,
        'feature_duplicate_rows': feature_duplicate_rows,
        'leakage_suspects': leakage_suspects,
        'class_imbalance': class_imbalance,
        'warnings': warnings_list,
    }


# ---------------------------------------------------------------------------
# Problem type
# ---------------------------------------------------------------------------
def _looks_like_classification(series):
    """Single rule used for both detection and preparation."""
    if not _is_numeric(series):
        return True
    n_unique = series.nunique()
    if n_unique == 2:
        return True
    values = series.to_numpy(dtype=float)
    # Fractional values mean a continuous quantity.
    if not bool(np.all(np.mod(values, 1) == 0)):
        return False
    n_rows = len(series)
    return n_unique <= 20 and (n_rows < 200 or (n_unique / n_rows) < 0.05)


def detect_problem_type(df, target, override=None) -> str:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    target_series = df[target].dropna()
    if target_series.empty:
        raise ValueError(f"Target column '{target}' is entirely null.")

    if override is not None:
        if override not in ('classification', 'regression'):
            raise ValueError("problem_type must be 'classification' or 'regression'.")
        if override == 'regression' and not _is_numeric(target_series):
            raise ValueError(f"Target column '{target}' is not numeric, so it can't be used for regression.")
        return override

    if not _is_numeric(target_series):
        n_unique = target_series.nunique()
        if n_unique > TEXT_LIKE_MIN_UNIQUE and n_unique / len(target_series) > TEXT_LIKE_MIN_RATIO:
            raise ValueError(
                f"Target column '{target}' has {n_unique} different text values, which looks like "
                'free text or an identifier. Choose a different target.'
            )
        return "classification"

    return "classification" if _looks_like_classification(target_series) else "regression"


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(df, target, test_size=TEST_SIZE_DEFAULT, random_state=RANDOM_STATE,
                     return_metadata=False, problem_type=None, scale=True,
                     categorical_columns=None):
    """Prepare a raw feature table and split it before learned preprocessing.

    Imputation, scaling and one-hot encoding are deliberately left to the
    estimator pipeline. The pipeline is fitted on each CV training fold and on
    the final training split only. `categorical_columns` identifies columns
    already label-encoded by cleaner.py; those integer codes are converted back
    to categorical strings so their arbitrary numeric ordering is not learned.

    Returns (X_train, X_test, y_train, y_test, encoder, feature_names[, metadata]).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    df_clean = df.dropna(subset=[target]).copy()
    if len(df_clean) < MIN_ROWS:
        raise ValueError(
            f'At least {MIN_ROWS} rows with a target value are required '
            f'(found {len(df_clean)}).'
        )

    problem_type = detect_problem_type(df_clean, target, override=problem_type)
    identifier_columns = _identifier_columns(df_clean, target)
    # Cleaner-generated anomaly flags and explanations are audit metadata,
    # not predictive features. Exclude them to prevent accidental leakage.
    audit_columns = [column for column in ('_auto_clean_anomaly', '_auto_clean_anomaly_reason') if column in df_clean.columns]
    X = df_clean.drop(columns=[target, *identifier_columns, *audit_columns]).copy()
    y = df_clean[target].copy()

    unusable = _unusable_columns(X)
    to_drop = unusable['empty'] + unusable['datetime'] + unusable['text_like']
    if to_drop:
        X = X.drop(columns=to_drop)
    if X.shape[1] == 0:
        raise ValueError(
            'No usable feature columns remain after removing identifiers, empty, '
            'date/time and text-like columns.'
        )

    # Convert cleaner-generated category codes to categorical strings. Also keep
    # native text/category/bool columns categorical. No one-hot fitting occurs
    # here; it happens inside the model pipeline after the split.
    encoded_category_columns = []
    requested_categoricals = set(categorical_columns or [])
    for column in X.columns:
        series = X[column]
        if column in requested_categoricals:
            X[column] = series.astype('string').astype(object).where(series.notna(), np.nan)
            encoded_category_columns.append(str(column))
        elif (pd.api.types.is_object_dtype(series)
              or pd.api.types.is_string_dtype(series)
              or isinstance(series.dtype, pd.CategoricalDtype)
              or pd.api.types.is_bool_dtype(series)):
            X[column] = series.astype('string').astype(object).where(series.notna(), np.nan)
            encoded_category_columns.append(str(column))
        elif _is_numeric(series):
            X[column] = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)

    # Avoid duplicate/non-string column names causing issues in sklearn and reports.
    X.columns = [str(column) for column in X.columns]
    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f'Feature names must be unique after conversion to text: {duplicates[:5]}')

    encoder = None
    if problem_type == 'classification':
        labels = y if _is_numeric(y) else y.astype(str)
        encoder = LabelEncoder()
        y_values = encoder.fit_transform(labels)
        if len(encoder.classes_) < 2:
            raise ValueError('The target has only one class; at least two are required.')
    else:
        y_values = pd.to_numeric(y, errors='coerce').astype('float64').to_numpy()
        valid_target = np.isfinite(y_values)
        if not valid_target.all():
            X = X.loc[valid_target].copy()
            y_values = y_values[valid_target]
        if pd.Series(y_values).nunique() < 2:
            raise ValueError('The target has no variation, so there is nothing to predict.')

    stratify_param = None
    if problem_type == 'classification':
        counts = np.bincount(y_values)
        n_samples = len(y_values)
        n_test = int(math.ceil(test_size * n_samples))
        n_classes = len(counts)
        if counts.min() >= 2 and n_test >= n_classes and (n_samples - n_test) >= n_classes:
            stratify_param = y_values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_values, test_size=test_size, shuffle=True,
        random_state=random_state, stratify=stratify_param
    )

    imputed_cells = int(X_train.isna().sum().sum() + X_test.isna().sum().sum())
    dropped_columns = {
        key: _cap(columns) for key, columns in
        (('empty', unusable['empty']), ('datetime', unusable['datetime']),
         ('text_like', unusable['text_like'])) if columns
    }
    metadata = {
        'identifier_columns': _cap(identifier_columns),
        'dropped_columns': dropped_columns,
        'test_size': test_size,
        'train_size': len(X_train),
        'test_count': len(X_test),
        'shuffle': True,
        'stratified': stratify_param is not None,
        'scaled': bool(scale),
        'imputed_cells': imputed_cells,
        'preprocessing_in_pipeline': True,
        'n_features': int(X.shape[1]),
        'problem_type': problem_type,
        'one_hot_encoded_codes': _cap(encoded_category_columns),
        'categorical_columns': _cap(encoded_category_columns),
    }
    result = (X_train, X_test, y_train, y_test, encoder, X.columns.tolist())
    return (*result, metadata) if return_metadata else result


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def _make_preprocessing_pipeline(model, X_train=None, scale=True):
    """Build a leakage-safe preprocessing + estimator pipeline.

    Numeric imputation/scaling and categorical imputation/one-hot encoding are
    fitted only on the data passed to Pipeline.fit (a CV fold or training split).
    X_train is used only to define column groups; no statistics/categories are
    learned from it at construction time.
    """
    if X_train is None:
        numeric_columns = []
        categorical_columns = []
    else:
        numeric_columns = [
            column for column in X_train.columns
            if _is_numeric(X_train[column]) and not pd.api.types.is_bool_dtype(X_train[column])
        ]
        categorical_columns = [column for column in X_train.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale:
            numeric_steps.append(('scaler', StandardScaler()))
        transformers.append(('numeric', Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        try:
            one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        except TypeError:  # scikit-learn < 1.2
            one_hot = OneHotEncoder(handle_unknown='ignore', sparse=True)
        categorical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', one_hot),
        ])
        transformers.append(('categorical', categorical_pipe, categorical_columns))

    if not transformers:
        raise ValueError('No numeric or categorical feature columns are available for training.')

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def _cv_estimator(model):
    """Clone a model for CV; skip SVC probability calibration when possible."""
    estimator = clone(model)
    # SVC is the final step inside our preprocessing Pipeline.
    if isinstance(estimator, Pipeline) and 'model' in estimator.named_steps:
        final_estimator = estimator.named_steps['model']
        if 'probability' in final_estimator.get_params(deep=False):
            final_estimator.set_params(probability=False)
    elif 'probability' in estimator.get_params(deep=False):
        estimator.set_params(probability=False)
    return estimator


def _cross_validation_metrics(model, X_train, y_train, problem_type, scale_target=False):
    y_values = np.asarray(y_train)
    n_rows = len(y_values)
    is_classification = problem_type == 'classification'
    metric_name = 'macro_f1' if is_classification else 'r2'

    if is_classification:
        minimum_class_count = int(pd.Series(y_values).value_counts().min())
        folds = min(CV_FOLDS, minimum_class_count)
        repeats = CV_REPEATS_SMALL if n_rows <= CV_SMALL_ROW_LIMIT else 1
        if folds < 2:
            return {'cv_folds': folds, 'cv_repeats': repeats, 'cv_mean': None,
                    'cv_std': None, 'cv_metric': metric_name}
        splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                           random_state=RANDOM_STATE)
    else:
        repeats = 1
        folds = min(CV_FOLDS, max(2, n_rows // 2))
        if n_rows < 4:
            return {'cv_folds': 0, 'cv_repeats': repeats, 'cv_mean': None,
                    'cv_std': None, 'cv_metric': metric_name}
        splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        y_values = y_values.astype(float)

    scores, errors = [], []
    truncated = False
    started = time.perf_counter()

    for train_index, validation_index in splitter.split(np.zeros(n_rows), y_values):
        try:
            estimator = _cv_estimator(model)
            X_fit = _take(X_train, train_index)
            X_validation = _take(X_train, validation_index)
            y_fit = y_values[train_index]
            y_validation = y_values[validation_index]

            if scale_target:
                fold_mean = float(y_fit.mean())
                fold_std = float(y_fit.std()) or 1.0
                estimator.fit(X_fit, (y_fit - fold_mean) / fold_std)
                predictions = np.asarray(estimator.predict(X_validation), dtype=float)
                predictions = predictions * fold_std + fold_mean
            else:
                estimator.fit(X_fit, y_fit)
                predictions = estimator.predict(X_validation)

            if is_classification:
                scores.append(float(f1_score(y_validation, predictions,
                                             average='macro', zero_division=0)))
            else:
                scores.append(float(r2_score(y_validation, predictions)))
        except Exception as error:
            errors.append(str(error))

        # Stop early on slow models, but only after one full repeat so the
        # score is still comparable between models.
        if (len(scores) >= folds
                and time.perf_counter() - started > CV_TIME_BUDGET_SECONDS):
            truncated = True
            break

    metrics = {
        'cv_folds': folds,
        'cv_repeats': max(1, len(scores) // folds) if truncated else repeats,
        'cv_mean': float(np.mean(scores)) if scores else None,
        'cv_std': float(np.std(scores)) if scores else None,
        'cv_metric': metric_name,
    }
    if truncated:
        metrics['cv_truncated'] = True
    if errors and not scores:
        metrics['cv_error'] = errors[0]
    return metrics


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def _select_models(all_models, selected_models, aliases):
    """
    Return (models to train, error entries for names that can't be trained).
    selected_models=None means "everything"; an empty selection is an error.
    """
    if selected_models is None:
        return dict(all_models), {}

    selected = []
    for name in selected_models:
        name = aliases.get(name, name)
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError('No models were selected. Choose at least one algorithm.')

    chosen = {name: all_models[name] for name in selected if name in all_models}
    unavailable = {}
    for name in selected:
        if name in all_models:
            continue
        if name == 'XGBoost' and not XGBOOST_AVAILABLE:
            message = 'XGBoost is not installed (pip install xgboost).'
        else:
            message = f"'{name}' is not available for this problem type."
        unavailable[name] = {'model': None, 'error': message}
    return chosen, unavailable


def _aligned_probabilities(model, X_test, all_labels):
    """predict_proba with one column per class (zeros for classes unseen in training)."""
    if not hasattr(model, 'predict_proba'):
        return None
    raw = np.asarray(model.predict_proba(X_test), dtype=float)
    aligned = np.zeros((raw.shape[0], len(all_labels)))
    label_index = {label: position for position, label in enumerate(all_labels)}
    for column, label in enumerate(model.classes_):
        aligned[:, label_index[label]] = raw[:, column]
    return aligned


def train_classification_models(X_train, X_test, y_train, y_test, selected_models=None, progress_callback=None) -> dict:
    y_train_array = np.asarray(y_train)
    y_test_array = np.asarray(y_test)
    all_labels = np.unique(np.concatenate([y_train_array, y_test_array]))
    n_classes = len(all_labels)

    all_models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
        "Neural Network": MLPClassifier(random_state=RANDOM_STATE, max_iter=1000)
    }
    if XGBOOST_AVAILABLE:
        all_models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss' if n_classes == 2 else 'mlogloss'
        )

    # Keep preprocessing inside the estimator so CV folds cannot influence
    # imputation medians or scaling statistics. The same pipeline is refit on
    # the complete training split before evaluation on the held-out test set.
    all_models = {name: _make_preprocessing_pipeline(model, X_train)
                  for name, model in all_models.items()}

    models_to_train, unavailable = _select_models(all_models, selected_models, CLASSIFICATION_ALIASES)

    results = {}
    total_models = max(len(models_to_train), 1)
    for model_index, (name, model) in enumerate(models_to_train.items(), start=1):
        try:
            started = time.perf_counter()
            model.fit(X_train, y_train_array)
            fit_seconds = time.perf_counter() - started

            preds = model.predict(X_test)
            probs = _aligned_probabilities(model, X_test, all_labels)

            metrics = {
                "accuracy": float(accuracy_score(y_test_array, preds)),
                "precision": float(precision_score(y_test_array, preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test_array, preds, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test_array, preds, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(y_test_array, preds, average="macro", zero_division=0)),
                # labels=all_labels keeps the matrix the same size as the class list
                "confusion_matrix": confusion_matrix(y_test_array, preds, labels=all_labels).tolist(),
            }

            # ROC-AUC: binary, or one-vs-rest macro average for multi-class
            roc_auc = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if n_classes == 2:
                        if probs is not None:
                            roc_auc = float(roc_auc_score(y_test_array, probs[:, 1]))
                        elif hasattr(model, "decision_function"):
                            roc_auc = float(roc_auc_score(y_test_array, model.decision_function(X_test)))
                    elif probs is not None:
                        roc_auc = float(roc_auc_score(y_test_array, probs, multi_class='ovr',
                                                      average='macro', labels=all_labels))
                if roc_auc is not None and not math.isfinite(roc_auc):
                    roc_auc = None
            except (ValueError, IndexError):
                roc_auc = None
            metrics["roc_auc"] = roc_auc
            metrics["fit_seconds"] = float(fit_seconds)
            metrics["y_true"] = y_test_array.tolist()
            metrics["y_pred"] = np.asarray(preds).tolist()
            metrics["probabilities"] = probs.tolist() if probs is not None else None
            metrics.update(_cross_validation_metrics(model, X_train, y_train_array, 'classification'))

            results[name] = {"model": model, "metrics": metrics}
        except Exception as error:
            logger.warning("Classification model %s failed: %s", name, error, exc_info=True)
            results[name] = {"model": None, "error": str(error)}
        if callable(progress_callback):
            progress_callback(model_index, total_models, name)

    results.update(unavailable)
    return results


def train_regression_models(X_train, X_test, y_train, y_test, selected_models=None, progress_callback=None) -> dict:
    y_train_array = np.asarray(y_train, dtype=float)
    y_test_array = np.asarray(y_test, dtype=float)
    target_mean = float(y_train_array.mean())
    target_std = float(y_train_array.std()) or 1.0

    all_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "SVM": SVR(),
        "Neural Network": MLPRegressor(random_state=RANDOM_STATE, max_iter=1000),
    }
    if XGBOOST_AVAILABLE:
        all_models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE)

    # Preprocessing is learned independently inside every CV fold and then
    # refit on the full training split for final test-set evaluation.
    all_models = {name: _make_preprocessing_pipeline(model, X_train)
                  for name, model in all_models.items()}

    models_to_train, unavailable = _select_models(all_models, selected_models, REGRESSION_ALIASES)

    results = {}
    total_models = max(len(models_to_train), 1)
    for model_index, (name, model) in enumerate(models_to_train.items(), start=1):
        scale_target = name in TARGET_SCALED_MODELS
        try:
            started = time.perf_counter()
            if scale_target:
                model.fit(X_train, (y_train_array - target_mean) / target_std)
            else:
                model.fit(X_train, y_train_array)
            fit_seconds = time.perf_counter() - started

            preds = np.asarray(model.predict(X_test), dtype=float)
            if scale_target:
                preds = preds * target_std + target_mean

            mse = mean_squared_error(y_test_array, preds)
            metrics = {
                "mae": float(mean_absolute_error(y_test_array, preds)),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test_array, preds)),
                "fit_seconds": float(fit_seconds),
            }
            metrics["y_true"] = y_test_array.tolist()
            metrics["y_pred"] = preds.tolist()
            metrics.update(_cross_validation_metrics(model, X_train, y_train_array,
                                                     'regression', scale_target=scale_target))
            result = {"model": model, "metrics": metrics}
            if scale_target:
                # The fitted model predicts in standardised units; keep the
                # scale so callers can convert predictions back.
                result["target_scale"] = {"mean": target_mean, "std": target_std}
            results[name] = result
        except Exception as error:
            logger.warning("Regression model %s failed: %s", name, error, exc_info=True)
            results[name] = {"model": None, "error": str(error)}
        if callable(progress_callback):
            progress_callback(model_index, total_models, name)

    results.update(unavailable)
    return results


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def _model_primary_score_key(problem_type):
    return 'macro_f1' if problem_type == 'classification' else 'r2'


def get_best_model_selection(results, problem_type):
    """
    Return (winner name, valid results, explanation).

    Models are ranked by CV mean. If several models are within TIE_TOLERANCE of
    the top CV mean, the most stable one (lowest CV std, then highest CV mean,
    then best test score) wins.
    """
    valid_results = {k: v for k, v in results.items() if v.get("model") is not None}
    if not valid_results:
        return None, {}, 'No valid models were trained.'

    score_key = _model_primary_score_key(problem_type)

    def metric(result, key):
        return result['metrics'].get(key)

    def rank_key(item):
        _, result = item
        cv_mean, cv_std, score = metric(result, 'cv_mean'), metric(result, 'cv_std'), metric(result, score_key)
        return (
            cv_mean is not None,
            cv_mean if cv_mean is not None else float('-inf'),
            -(cv_std if cv_std is not None else float('inf')),
            score if score is not None else float('-inf'),
        )

    ranked_models = sorted(valid_results.items(), key=rank_key, reverse=True)
    top_name, top_result = ranked_models[0]
    top_cv_mean = metric(top_result, 'cv_mean')
    if top_cv_mean is None:
        return top_name, valid_results, 'Selected by the available validation metrics.'

    tied = [(name, result) for name, result in ranked_models
            if metric(result, 'cv_mean') is not None
            and top_cv_mean - metric(result, 'cv_mean') <= TIE_TOLERANCE]

    if len(tied) == 1:
        return top_name, valid_results, 'Selected by highest CV mean and lowest CV std.'

    def tie_break(item):
        _, result = item
        cv_std = metric(result, 'cv_std')
        score = metric(result, score_key)
        return (
            cv_std if cv_std is not None else float('inf'),
            -metric(result, 'cv_mean'),
            -(score if score is not None else float('-inf')),
        )

    winner_name = min(tied, key=tie_break)[0]
    tied_names = ' / '.join(name for name, _ in tied)
    note = (f'{winner_name} selected among {len(tied)} models within {TIE_TOLERANCE:.3f} '
            f'CV mean of the top score ({tied_names}); the lowest CV std decided.')
    return winner_name, valid_results, note


def compare_models(results, problem_type):
    """Return (best model name, all results). Failed models stay in `results`."""
    best_model_name, _, _ = get_best_model_selection(results, problem_type)
    return best_model_name, results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def _unwrap_estimator(model):
    steps = getattr(model, 'steps', None)  # sklearn Pipeline
    return steps[-1][1] if steps else model


def _permutation_importances(model, X_test, y_test, problem_type, target_scale=None):
    """Permutation importance on (a sample of) the test set, or None if too expensive."""
    try:
        if X_test.shape[1] > PERMUTATION_MAX_FEATURES:
            return None
        y_eval = np.asarray(y_test)
        X_eval = X_test
        if len(y_eval) > PERMUTATION_MAX_ROWS:
            chosen = np.random.default_rng(RANDOM_STATE).choice(
                len(y_eval), PERMUTATION_MAX_ROWS, replace=False
            )
            X_eval, y_eval = X_test.iloc[chosen], y_eval[chosen]
        if target_scale:
            # The model predicts standardised values; R2 is unchanged by an
            # affine rescale applied to both sides.
            y_eval = (y_eval.astype(float) - target_scale['mean']) / target_scale['std']
        scoring = 'f1_macro' if problem_type == 'classification' else 'r2'
        outcome = permutation_importance(
            model, X_eval, y_eval, scoring=scoring, n_repeats=5,
            random_state=RANDOM_STATE, n_jobs=1
        )
        return outcome.importances_mean
    except Exception:
        logger.warning('Permutation importance failed', exc_info=True)
        return None


def _transformed_feature_names(model, original_columns):
    """Return post-encoding feature names for native model importances."""
    try:
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
            names = preprocessor.get_feature_names_out()
            return [str(name).replace('numeric__', '').replace('categorical__', '')
                    for name in names]
    except Exception:
        logger.debug('Could not retrieve transformed feature names', exc_info=True)
    return [str(name) for name in original_columns]


def feature_importance(best_model, columns, X_test=None, y_test=None,
                       problem_type=None, target_scale=None):
    """Return native importances for transformed features or permutation
    importances for original input columns when the estimator has no native
    feature-importance attribute.
    """
    if best_model is None:
        return None

    estimator = _unwrap_estimator(best_model)
    importances = None
    importance_names = [str(name) for name in columns]
    native_importance = False

    if hasattr(estimator, 'feature_importances_'):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        importance_names = _transformed_feature_names(best_model, columns)
        native_importance = True
    elif hasattr(estimator, 'coef_'):
        coef = np.asarray(estimator.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        importance_names = _transformed_feature_names(best_model, columns)
        native_importance = True

    if importances is None and X_test is not None and y_test is not None and problem_type:
        importances = _permutation_importances(
            best_model, X_test, y_test, problem_type, target_scale
        )
        importance_names = [str(name) for name in columns]

    if importances is None:
        return None

    importances = np.ravel(importances)
    if len(importances) != len(importance_names):
        logger.warning(
            'Importance length (%d) does not match feature-name count (%d); skipping.',
            len(importances), len(importance_names)
        )
        return None

    importance_list = [
        {'feature': str(name), 'importance': float(value)}
        for name, value in zip(importance_names, importances)
    ]
    return sorted(importance_list, key=lambda item: item['importance'], reverse=True)

