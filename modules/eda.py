# modules/eda.py
"""
Charts for the Data Platform.

Design notes
------------
* Figures are built with matplotlib's ``Figure`` class instead of ``pyplot``.
  ``pyplot`` keeps global state and is not thread-safe, and un-closed pyplot
  figures leak memory. ``Figure`` objects are garbage-collected normally.
* Every plot function accepts ``output_dir`` (so each user/session can have its
  own folder) and ``theme`` ('light' or 'dark'). With ``theme=None`` the module
  falls back to the global ``COLORS`` dict, so older code that swaps ``COLORS``
  to render a dark variant keeps working.
* When a chart cannot be drawn (no numeric columns, empty data, ...) any old
  file with the same name is deleted, so a previous dataset's chart is never
  shown by mistake.
* Files are written to a temporary name and then renamed, so a browser never
  receives a half-written PNG.
"""
import logging
import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)

# Chart fonts can't draw every script (e.g. CJK column names); matplotlib warns once
# per character. The chart is still produced, so keep the console quiet.
warnings.filterwarnings('ignore', message=r'Glyph .* missing from font')

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
LIGHT_THEME = {
    'purple': '#6366f1',
    'blue': '#2563eb',
    'green': '#16a34a',
    'orange': '#ea580c',
    'pink': '#db2777',
    'red': '#dc2626',
    'bg': '#ffffff',
    'text': '#1f2937',
    'muted': '#6b7280',
    'grid': '#d1d5db',
}

DARK_THEME = {
    'purple': '#6366f1',
    'blue': '#3b82f6',
    'green': '#22c55e',
    'orange': '#f97316',
    'pink': '#ec4899',
    'red': '#ef4444',
    'bg': '#0d1117',
    'text': '#e5e7eb',
    'muted': '#9ca3af',
    'grid': '#374151',
}

# Legacy global palette. Code that sets theme=None uses this dict, which lets
# callers render a dark variant by temporarily updating it.
COLORS = dict(LIGHT_THEME)

# Kept for backwards compatibility with other modules.
REPORT_COLORS = {
    'purple': '#6366f1',
    'blue': '#2563eb',
    'green': '#16a34a',
    'red': '#dc2626',
    'text': '#1f2937',
    'muted': '#6b7280',
    'grid': '#d1d5db',
}

# Save beside Flask's static folder so generated images are served by url_for('static').
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_DIR, 'static', 'generated')
os.makedirs(SAVE_DIR, exist_ok=True)

# Readability limits
MAX_MISSING_BARS = 30
MAX_HEATMAP_COLUMNS = 25
MAX_CLASSES_SHOWN = 20
MAX_SCATTER_POINTS = 5000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _palette(theme=None):
    if theme == 'dark':
        return dict(DARK_THEME)
    if theme == 'light':
        return dict(LIGHT_THEME)
    palette = dict(LIGHT_THEME)
    palette.update(COLORS)
    return palette


def _resolve_dir(output_dir=None):
    directory = output_dir or SAVE_DIR
    os.makedirs(directory, exist_ok=True)
    return directory


def _chart_name(base, theme=None):
    return f'{base}_dark.png' if theme == 'dark' else f'{base}.png'


def _safe_name(value):
    cleaned = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value)).strip('_')
    return cleaned[:60] or 'chart'


def _remove_chart(directory, filename):
    try:
        os.remove(os.path.join(directory, filename))
    except OSError:
        pass


def _save_figure(fig, directory, filename, transparent=True, facecolor='white', dpi=130):
    """Save atomically: write to a temp file, then rename."""
    path = os.path.join(directory, filename)
    tmp_path = path + '.tmp'
    options = {'format': 'png', 'dpi': dpi, 'bbox_inches': 'tight'}
    if transparent:
        options['transparent'] = True
    else:
        options['facecolor'] = facecolor
    try:
        with warnings.catch_warnings():
            # "Glyph ... missing from font" for names the chart font can't draw
            warnings.simplefilter('ignore', UserWarning)
            fig.savefig(tmp_path, **options)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return filename


def _style_axes(ax, pal):
    ax.figure.set_facecolor(pal['bg'])
    ax.set_facecolor(pal['bg'])
    for spine in ax.spines.values():
        spine.set_color(pal['muted'])
    ax.xaxis.label.set_color(pal['text'])
    ax.yaxis.label.set_color(pal['text'])
    ax.title.set_color(pal['text'])
    ax.tick_params(colors=pal['text'])


def _style_legend(ax, pal):
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor(pal['bg'])
    legend.get_frame().set_edgecolor(pal['muted'])
    for text in legend.get_texts():
        text.set_color(pal['text'])
    title = legend.get_title()
    if title is not None:
        title.set_color(pal['text'])


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
_ID_NAMES = {'id', 'index', 'rowid', 'recordid', 'userid', 'uuid', 'guid', 'key'}
# Matches: id, ID, user_id, user id, customer-id, userId, customerID
# Does NOT match: paid, valid, android, grid, liquid
_ID_PATTERN = re.compile(r'(?:^|[_\s\-.])(?:id|ID|Id)$|[a-z](?:ID|Id)$')


def _identifier_columns(df):
    """Find identifier-like columns that should not appear in EDA charts."""
    identifier_columns = []
    row_count = len(df)

    for column in df.columns:
        name = str(column)
        normalized = ''.join(ch for ch in name.lower() if ch.isalnum())
        series = df[column]

        is_named_identifier = normalized in _ID_NAMES or bool(_ID_PATTERN.search(name))
        # Names that merely end in "id" only count when the values look unique.
        # Float columns are skipped: continuous measurements are almost never IDs.
        if (not is_named_identifier and normalized.endswith('id') and row_count > 1
                and not pd.api.types.is_float_dtype(series)):
            is_named_identifier = series.nunique(dropna=True) / row_count >= 0.9

        is_sequential_integer = False
        if row_count and pd.api.types.is_integer_dtype(series) and series.notna().all():
            values = series.to_numpy(dtype='int64')
            if np.unique(values).size == row_count:
                sorted_values = np.sort(values)
                expected = np.arange(sorted_values[0], sorted_values[0] + row_count)
                is_sequential_integer = bool(np.array_equal(sorted_values, expected))

        if is_named_identifier or is_sequential_integer:
            identifier_columns.append(column)

    return identifier_columns


def _numeric_feature_columns(df, exclude_columns=None):
    """Numeric columns excluding identifier-like and explicitly excluded columns."""
    excluded = set(_identifier_columns(df)) | set(exclude_columns or [])
    return [column for column in df.select_dtypes(include=[np.number]).columns
            if column not in excluded]


def dataset_summary(df):
    return {
        'rows': int(len(df)),
        'cols': int(len(df.columns)),
        'missing': int(df.isnull().sum().sum()),
        'types': {str(dtype): int(count) for dtype, count in df.dtypes.value_counts().items()},
    }


# ---------------------------------------------------------------------------
# EDA charts (themed, transparent background)
# ---------------------------------------------------------------------------
def plot_missing_values(df, exclude_columns=None, theme=None, output_dir=None):
    directory = _resolve_dir(output_dir)
    filename = _chart_name('missing_values', theme)
    pal = _palette(theme)

    excluded = set(_identifier_columns(df)) | set(exclude_columns or [])
    data = df.drop(columns=list(excluded), errors='ignore')
    if data.shape[1] == 0:
        _remove_chart(directory, filename)
        return None

    missing = data.isnull().sum().sort_values(ascending=False)
    total_columns = len(missing)
    truncated = total_columns > MAX_MISSING_BARS
    if truncated:
        missing = missing.iloc[:MAX_MISSING_BARS]

    names = [str(column) for column in missing.index]
    values = missing.to_numpy()
    positions = np.arange(len(names))

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    colors = [pal['pink'] if value > 0 else pal['blue'] for value in values]
    ax.bar(positions, values, color=colors)

    title = 'Missing Values Overview'
    if truncated:
        title += f' (top {MAX_MISSING_BARS} of {total_columns} columns)'
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Missing Count')
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha='right')

    peak = float(values.max()) if len(values) else 0.0
    if peak == 0:
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, 'No missing values', transform=ax.transAxes,
                ha='center', va='center', color=pal['muted'], fontsize=13)
    else:
        for position, value in zip(positions, values):
            if value > 0:
                ax.text(position, value + peak * 0.01, str(int(value)),
                        ha='center', va='bottom', color=pal['text'])

    ax.legend(handles=[
        Patch(color=pal['pink'], label='Missing values present'),
        Patch(color=pal['blue'], label='No missing values'),
    ], loc='upper right')

    _style_axes(ax, pal)
    _style_legend(ax, pal)
    return _save_figure(fig, directory, filename)


def plot_distribution(df, column=None, exclude_columns=None, theme=None, output_dir=None):
    directory = _resolve_dir(output_dir)
    filename = _chart_name('dist_column', theme)  # consistent filename for dashboard
    pal = _palette(theme)

    if column is None:
        numeric_cols = _numeric_feature_columns(df, exclude_columns)
        if not numeric_cols:
            _remove_chart(directory, filename)
            return None
        column = numeric_cols[0]
    elif column not in df.columns:
        _remove_chart(directory, filename)
        return None

    values = pd.to_numeric(df[column], errors='coerce').dropna()
    if values.empty:
        _remove_chart(directory, filename)
        return None

    unique_count = values.nunique()
    is_integer_valued = bool(np.all(np.mod(values.to_numpy(), 1) == 0))
    use_discrete = is_integer_valued and unique_count <= 15
    use_kde = unique_count > 2 and float(values.std()) > 0

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    sns.histplot(values, color=pal['purple'], kde=use_kde, discrete=use_discrete,
                 ax=ax, label='Observed values')
    ax.set_title(f'Distribution of {column}', fontsize=14)
    ax.set_xlabel(str(column))
    ax.set_ylabel('Frequency')
    ax.legend(title='Series', loc='upper right')

    _style_axes(ax, pal)
    _style_legend(ax, pal)
    return _save_figure(fig, directory, filename)


def plot_correlation_heatmap(df, exclude_columns=None, theme=None, output_dir=None):
    directory = _resolve_dir(output_dir)
    filename = _chart_name('correlation_heatmap', theme)
    pal = _palette(theme)

    features = [column for column in _numeric_feature_columns(df, exclude_columns)
                if df[column].nunique(dropna=True) > 1]
    if len(features) < 2:
        _remove_chart(directory, filename)
        return None

    total_features = len(features)
    truncated = total_features > MAX_HEATMAP_COLUMNS
    features = features[:MAX_HEATMAP_COLUMNS]
    count = len(features)

    corr = df[features].corr()
    labels = [str(column) for column in features]
    corr.columns = labels
    corr.index = labels

    side = min(max(6.0, 0.6 * count + 3.0), 14.0)
    fig = Figure(figsize=(side * 1.1, side))
    ax = fig.subplots()

    cmap = sns.diverging_palette(270, 10, as_cmap=True)
    sns.heatmap(corr, annot=count <= 12, fmt='.2f', cmap=cmap, center=0,
                vmin=-1, vmax=1, linewidths=.5,
                cbar_kws={'shrink': .8, 'label': 'Correlation'}, ax=ax)

    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.tick_params(colors=pal['text'])
        colorbar.set_label('Correlation', color=pal['text'])
        colorbar.outline.set_edgecolor(pal['muted'])

    title = 'Correlation Heatmap'
    if truncated:
        title += f' (first {MAX_HEATMAP_COLUMNS} of {total_features} features)'
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Numeric features')
    ax.set_ylabel('Numeric features')
    ax.tick_params(axis='x', labelrotation=45)

    _style_axes(ax, pal)
    return _save_figure(fig, directory, filename)


def plot_outliers(df, exclude_columns=None, theme=None, output_dir=None):
    """One box plot per feature (each with its own y-axis) so scales don't hide each other."""
    directory = _resolve_dir(output_dir)
    filename = _chart_name('outliers', theme)
    pal = _palette(theme)

    candidates = [column for column in _numeric_feature_columns(df, exclude_columns)
                  if df[column].notna().any()]
    # Prefer informative columns over binary flags.
    informative = [column for column in candidates if df[column].nunique(dropna=True) > 2]
    columns = (informative or candidates)[:4]
    if not columns:
        _remove_chart(directory, filename)
        return None

    box_colors = [pal['blue'], pal['green'], pal['orange'], pal['pink']]
    fig = Figure(figsize=(3.2 * len(columns) + 2, 6))
    axes = fig.subplots(1, len(columns), squeeze=False)[0]

    for ax, column, color in zip(axes, columns, box_colors):
        values = pd.to_numeric(df[column], errors='coerce').dropna().to_numpy()
        ax.boxplot(
            values,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=color, edgecolor=pal['muted'], alpha=0.85),
            medianprops=dict(color=pal['text'], linewidth=1.5),
            whiskerprops=dict(color=pal['muted']),
            capprops=dict(color=pal['muted']),
            flierprops=dict(marker='o', markerfacecolor=pal['muted'],
                            markeredgecolor=pal['muted'], markersize=3, alpha=0.6),
        )
        ax.set_xticks([1])
        ax.set_xticklabels([str(column)])
        _style_axes(ax, pal)

    axes[0].set_ylabel('Feature values')
    axes[0].yaxis.label.set_color(pal['text'])
    fig.suptitle('Outlier Detection', color=pal['text'], fontsize=14)
    fig.set_facecolor(pal['bg'])
    return _save_figure(fig, directory, filename)


def plot_class_balance(df, target, theme=None, output_dir=None):
    directory = _resolve_dir(output_dir)
    filename = _chart_name(f'class_balance_{_safe_name(target)}', theme)
    pal = _palette(theme)

    if target not in df.columns:
        return None
    counts = df[target].value_counts(dropna=True).head(MAX_CLASSES_SHOWN)
    if counts.empty:
        _remove_chart(directory, filename)
        return None

    names = [str(label) for label in counts.index]
    class_colors = sns.color_palette('viridis', n_colors=len(counts))
    positions = np.arange(len(names))

    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.bar(positions, counts.to_numpy(), color=class_colors)
    ax.set_title(f'Class Balance: {target}', fontsize=14)
    ax.set_xlabel(str(target))
    ax.set_ylabel('Record count')
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha='right')

    peak = float(counts.max())
    for position, value in zip(positions, counts.to_numpy()):
        ax.text(position, value + peak * 0.01, str(int(value)),
                ha='center', va='bottom', color=pal['text'])

    if len(names) <= 10:
        ax.legend(handles=[Patch(color=color, label=name)
                           for color, name in zip(class_colors, names)],
                  title='Classes', loc='upper right')

    _style_axes(ax, pal)
    _style_legend(ax, pal)
    return _save_figure(fig, directory, filename)


def plot_feature_importance(importance_data, top_n=10, theme=None, output_dir=None):
    """Generate feature importance chart."""
    directory = _resolve_dir(output_dir)
    filename = _chart_name('feature_importance', theme)
    pal = _palette(theme)

    if not importance_data:
        _remove_chart(directory, filename)
        return None

    top_features = importance_data[:top_n]
    # Reverse so the highest value is drawn at the top.
    features = [str(item['feature'])[:30] for item in top_features][::-1]
    importances = [float(item['importance']) for item in top_features][::-1]

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    bars = ax.barh(features, importances, color=pal['purple'])
    ax.set_title('Feature Importance', fontsize=14)
    ax.set_ylabel('Features')
    ax.set_xlabel('Importance')
    ax.margins(x=0.18)
    ax.legend([Patch(color=pal['purple'])], ['Model importance'], loc='lower right')

    span = max((abs(value) for value in importances), default=0.0) or 1.0
    offset = span * 0.01
    for bar, value in zip(bars, importances):
        width = bar.get_width()
        ax.text(width + (offset if width >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center',
                ha='left' if width >= 0 else 'right', color=pal['text'])

    _style_axes(ax, pal)
    _style_legend(ax, pal)
    return _save_figure(fig, directory, filename)


# ---------------------------------------------------------------------------
# Model diagnostics
#
# Each diagnostic is saved twice: `<name>.png` (white background, used by the
# PDF report) and `<name>_dark.png` (transparent, light text, for dark mode).
# The functions return the plain filename, as before.
# ---------------------------------------------------------------------------
def _render_diagnostic(draw, filename, output_dir=None):
    directory = _resolve_dir(output_dir)
    base, extension = os.path.splitext(filename)
    dark_filename = f'{base}_dark{extension}'

    light_fig = draw(_palette('light'))
    if light_fig is None:
        _remove_chart(directory, filename)
        _remove_chart(directory, dark_filename)
        return None
    _save_figure(light_fig, directory, filename, transparent=False, facecolor='white', dpi=160)

    dark_fig = draw(_palette('dark'))
    if dark_fig is not None:
        _save_figure(dark_fig, directory, dark_filename, transparent=True, dpi=160)
    return filename


def plot_confusion_matrix(matrix, labels, filename='confusion_matrix.png', output_dir=None):
    directory = _resolve_dir(output_dir)
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        base, extension = os.path.splitext(filename)
        _remove_chart(directory, filename)
        _remove_chart(directory, f'{base}_dark{extension}')
        return None

    matrix = matrix.astype(int)
    size_count = matrix.shape[0]
    labels = list(labels) if labels is not None else []
    # If some classes never appear in the test set the matrix is smaller than
    # the label list; fall back to generic labels rather than mislabel cells.
    tick_labels = [str(label) for label in labels] if len(labels) == size_count \
        else [str(index) for index in range(size_count)]

    def draw(pal):
        side = min(max(5.0, 0.7 * size_count + 3.0), 12.0)
        fig = Figure(figsize=(side * 1.1, side))
        ax = fig.subplots()
        sns.heatmap(matrix, annot=size_count <= 15, fmt='d', cmap='Blues', cbar=False,
                    ax=ax, xticklabels=tick_labels, yticklabels=tick_labels,
                    linewidths=.5, linecolor=pal['bg'])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('Actual label')
        _style_axes(ax, pal)
        return fig

    return _render_diagnostic(draw, filename, output_dir)


def _prepare_probabilities(y_true, probabilities, labels):
    """Validate inputs; return (y_true, probabilities, n_classes) or None."""
    if probabilities is None or labels is None:
        return None
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        return None
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    if probabilities.shape[0] != y_true.shape[0]:
        return None
    n_classes = min(len(labels), probabilities.shape[1])
    if n_classes < 2:
        return None
    return y_true, probabilities, n_classes


def _class_indices(n_classes):
    # Binary problems: the two one-vs-rest curves are mirror images, so only
    # the positive class (index 1) is drawn.
    return [1] if n_classes == 2 else list(range(n_classes))


def plot_roc_curve(y_true, probabilities, labels, filename='roc_curve.png', output_dir=None):
    prepared = _prepare_probabilities(y_true, probabilities, labels)
    if prepared is None:
        return _render_diagnostic(lambda pal: None, filename, output_dir)
    y_true, probabilities, n_classes = prepared

    curves = []
    for index in _class_indices(n_classes):
        truth = (y_true == index).astype(int)
        positives = int(truth.sum())
        if positives == 0 or positives == len(truth):
            continue  # ROC is undefined without both outcomes
        false_positive, true_positive, _ = roc_curve(truth, probabilities[:, index])
        curves.append((str(labels[index]), false_positive, true_positive,
                       auc(false_positive, true_positive)))

    if not curves:
        return _render_diagnostic(lambda pal: None, filename, output_dir)

    title = 'ROC Curve' if n_classes == 2 else 'One-vs-Rest ROC Curve'

    def draw(pal):
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        for label, false_positive, true_positive, area in curves:
            ax.plot(false_positive, true_positive, label=f'{label} (AUC = {area:.3f})')
        ax.plot([0, 1], [0, 1], '--', color=pal['muted'], label='Random baseline')
        ax.set_title(title)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(title='Class', loc='lower right')
        ax.grid(alpha=0.25, color=pal['grid'])
        _style_axes(ax, pal)
        _style_legend(ax, pal)
        return fig

    return _render_diagnostic(draw, filename, output_dir)


def plot_precision_recall_curve(y_true, probabilities, labels,
                                filename='precision_recall_curve.png', output_dir=None):
    prepared = _prepare_probabilities(y_true, probabilities, labels)
    if prepared is None:
        return _render_diagnostic(lambda pal: None, filename, output_dir)
    y_true, probabilities, n_classes = prepared

    curves = []
    prevalence = None
    for index in _class_indices(n_classes):
        truth = (y_true == index).astype(int)
        positives = int(truth.sum())
        if positives == 0:
            continue
        precision, recall, _ = precision_recall_curve(truth, probabilities[:, index])
        average_precision = average_precision_score(truth, probabilities[:, index])
        curves.append((str(labels[index]), recall, precision, average_precision))
        if n_classes == 2:
            prevalence = positives / len(truth)

    if not curves:
        return _render_diagnostic(lambda pal: None, filename, output_dir)

    title = 'Precision-Recall Curve' if n_classes == 2 else 'One-vs-Rest Precision-Recall Curve'

    def draw(pal):
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        for label, recall, precision, average_precision in curves:
            ax.plot(recall, precision, label=f'{label} (AP = {average_precision:.3f})')
        if prevalence is not None:
            ax.axhline(prevalence, linestyle='--', color=pal['muted'],
                       label=f'Baseline ({prevalence:.2f})')
        ax.set_title(title)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Class', loc='lower left')
        ax.grid(alpha=0.25, color=pal['grid'])
        _style_axes(ax, pal)
        _style_legend(ax, pal)
        return fig

    return _render_diagnostic(draw, filename, output_dir)


def plot_actual_vs_predicted(actual, predicted, filename='actual_vs_predicted.png', output_dir=None):
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.size == 0 or actual.size != predicted.size:
        return _render_diagnostic(lambda pal: None, filename, output_dir)

    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[finite], predicted[finite]
    if actual.size == 0:
        return _render_diagnostic(lambda pal: None, filename, output_dir)

    minimum = float(min(actual.min(), predicted.min()))
    maximum = float(max(actual.max(), predicted.max()))

    # Draw at most MAX_SCATTER_POINTS points to keep rendering fast.
    if actual.size > MAX_SCATTER_POINTS:
        chosen = np.random.default_rng(42).choice(actual.size, MAX_SCATTER_POINTS, replace=False)
        plot_actual, plot_predicted = actual[chosen], predicted[chosen]
    else:
        plot_actual, plot_predicted = actual, predicted

    def draw(pal):
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        ax.scatter(plot_actual, plot_predicted, color=pal['blue'], alpha=0.75, label='Predictions')
        ax.plot([minimum, maximum], [minimum, maximum], '--', color=pal['red'],
                label='Perfect prediction')
        ax.set_title('Actual vs Predicted Values')
        ax.set_xlabel('Actual values')
        ax.set_ylabel('Predicted values')
        ax.legend(loc='best')
        ax.grid(alpha=0.25, color=pal['grid'])
        _style_axes(ax, pal)
        _style_legend(ax, pal)
        return fig

    return _render_diagnostic(draw, filename, output_dir)


def plot_model_comparison(model_scores, filename='model_comparison.png',
                          best_model=None, output_dir=None):
    items = []
    for name, score in (model_scores or {}).items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            items.append((str(name), value))
    if not items:
        return _render_diagnostic(lambda pal: None, filename, output_dir)

    names = [name for name, _ in items]
    scores = [score for _, score in items]
    best_name = str(best_model) if best_model is not None else None

    def draw(pal):
        fig = Figure(figsize=(9, 5))
        ax = fig.subplots()
        colors = []
        for index, name in enumerate(names):
            if best_name is not None and name == best_name:
                colors.append(pal['green'])
            else:
                colors.append(pal['purple'] if index % 2 == 0 else pal['blue'])
        bars = ax.bar(names, scores, color=colors)

        ax.set_title('Model Comparison')
        ax.set_ylabel('Primary score')
        ax.set_xlabel('Model')

        lower = min(0.0, min(scores) * 1.15)
        top_score = max(scores)
        upper = 1.05 if top_score <= 1.0 else top_score * 1.15
        if upper <= lower:
            upper = lower + 1.0
        ax.set_ylim(lower, upper)
        ax.tick_params(axis='x', labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_ha('right')

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, score, f'{score:.3f}',
                    ha='center', va='bottom' if score >= 0 else 'top', color=pal['text'])
        ax.grid(axis='y', alpha=0.25, color=pal['grid'])

        if best_name is not None and best_name in names:
            ax.legend(handles=[Patch(color=pal['green'], label='Selected model')],
                      loc='lower right')

        _style_axes(ax, pal)
        _style_legend(ax, pal)
        return fig

    return _render_diagnostic(draw, filename, output_dir)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
_EDA_PLOTTERS = (
    ('missing_values', 'missing_values', plot_missing_values),
    ('distribution', 'dist_column', plot_distribution),
    ('correlation', 'correlation_heatmap', plot_correlation_heatmap),
    ('outliers', 'outliers', plot_outliers),
)


def plot_all_eda(df, exclude_columns=None, theme=None, output_dir=None):
    """
    Draw the four EDA charts. One failing chart no longer blocks the others;
    a failed chart is reported as None and its stale file is removed.
    """
    directory = _resolve_dir(output_dir)
    results = {}
    for key, base, plotter in _EDA_PLOTTERS:
        try:
            results[key] = plotter(df, exclude_columns=exclude_columns,
                                   theme=theme, output_dir=directory)
        except Exception:
            logger.exception('EDA chart "%s" failed', key)
            _remove_chart(directory, _chart_name(base, theme))
            results[key] = None
    return results


def plot_all_eda_themes(df, exclude_columns=None, output_dir=None):
    """
    Draw every EDA chart in dark and light themes without touching global state.
    Writes `<name>.png` and `<name>_dark.png`; returns the light results dict.
    """
    plot_all_eda(df, exclude_columns=exclude_columns, theme='dark', output_dir=output_dir)
    return plot_all_eda(df, exclude_columns=exclude_columns, theme='light', output_dir=output_dir)


def clear_chart_directory(output_dir):
    """
    Delete all generated PNGs in a per-session folder. Refuses to run on the
    shared default folder so it can never wipe other users' charts.
    """
    if not output_dir:
        raise ValueError('output_dir is required')
    if os.path.abspath(output_dir) == os.path.abspath(SAVE_DIR):
        raise ValueError('Refusing to clear the shared chart folder')
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name.endswith('.png') or name.endswith('.png.tmp'):
            _remove_chart(output_dir, name)