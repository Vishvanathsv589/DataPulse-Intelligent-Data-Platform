# modules/reporter.py
"""
PDF report for the Data Platform.

What changed compared with the previous version
-----------------------------------------------
* Works with current fpdf2 releases (the old code crashed with "Not enough
  horizontal space to render a single character" because multi_cell() leaves the
  cursor at the right margin). Cursor handling now goes through small helpers and
  also works with the legacy `fpdf` package.
* Unicode text: if matplotlib's bundled DejaVu fonts are found they are used, so
  accented, Greek and Cyrillic column names print correctly. Characters the font
  can't draw (and every non-Latin-1 character when only the built-in fonts are
  available) become "?" instead of crashing the whole report.
* Charts come from `chart_dir` (the visitor's own folder). Model charts are only
  embedded when a model was actually trained, so a cleaning-only report can no
  longer show leftovers from an earlier run.
* Images are scaled to fit the page and flattened onto white, so tall charts can't
  run off the page and transparency renders the same everywhere.
* The table of contents and section numbers are built from what the report really
  contains (no "6. Machine Learning Results" when no model was trained, no blank
  page after the contents).
* Feature summary is a proper table (type, missing, unique, notes) instead of
  pandas text in a proportional font.
* Model table adds MAE/RMSE for regression, marks the selected model, and lists
  models that could not be trained. Quality section covers leakage and class
  imbalance. The cleaning section reports what the cleaner actually did.
* Executive summary no longer says "N/A" for ties and no longer calls tree
  importances "coefficients". Body text is dark grey instead of purple.
* Slim running header after page 1; errors keep their original traceback.
"""
import inspect
import os
import tempfile
import warnings
from datetime import datetime

from fpdf import FPDF

COLORS = {
    'purple': (109, 91, 245),   # #6d5bf5
    'blue': (59, 130, 246),     # #3b82f6
    'text': (31, 41, 55),       # body text
    'muted': (107, 114, 128),
    'dark': (11, 15, 25),       # banner background
}

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(PROJECT_DIR, 'static', 'generated')  # default; pass chart_dir= to override

MAX_FEATURE_ROWS = 40
MAX_LIST_ITEMS = 12
FEATURE_TABLE_NAME_WIDTH = 66

IS_FPDF2 = 'new_x' in inspect.signature(FPDF.cell).parameters

_REPLACEMENTS = {
    '\u2013': '-', '\u2014': '-', '\u2212': '-', '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2265': '>=', '\u2264': '<=',
    '\u2192': '->', '\u00a0': ' ', '\u2022': '-', '\u20ac': 'EUR',
}


class ReportError(Exception):
    """Raised when the PDF could not be generated."""


def _find_unicode_fonts():
    """Locate the DejaVu fonts that ship with matplotlib."""
    try:
        import matplotlib
        directory = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    except Exception:
        return None
    fonts = {
        '': os.path.join(directory, 'DejaVuSans.ttf'),
        'B': os.path.join(directory, 'DejaVuSans-Bold.ttf'),
        'I': os.path.join(directory, 'DejaVuSans-Oblique.ttf'),
    }
    return fonts if all(os.path.exists(path) for path in fonts.values()) else None


def _font_codepoints(path):
    try:
        from fontTools.ttLib import TTFont
        return set(TTFont(path, lazy=True).getBestCmap().keys())
    except Exception:
        return None


def _format_number(value, digits=4):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def _format_shape(shape):
    try:
        rows, columns = shape
        return f'{int(rows):,} rows x {int(columns):,} columns'
    except (TypeError, ValueError):
        return str(shape)


def _format_timestamp(value):
    if not value:
        return 'Unknown'
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed.strftime('%Y-%m-%d %H:%M') + (' UTC' if parsed.tzinfo else '')
    except ValueError:
        return str(value)


class DataPlatformReport(FPDF):
    def __init__(self, chart_dir):
        super().__init__()
        self.chart_dir = chart_dir
        self.temp_files = []
        self.section_number = 0
        self.dataset_name = ''
        self.base_font = 'Helvetica'
        self.unicode_ok = False
        self.codepoints = None
        self._setup_fonts()
        self.set_auto_page_break(True, margin=18)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                self.alias_nb_pages()
            except AttributeError:
                pass

    # ---- fonts and text -------------------------------------------------
    def _setup_fonts(self):
        fonts = _find_unicode_fonts() if IS_FPDF2 else None
        if not fonts:
            return
        try:
            for style, path in fonts.items():
                self.add_font('DejaVu', style, path)
        except Exception:
            return
        self.base_font = 'DejaVu'
        self.unicode_ok = True
        self.codepoints = _font_codepoints(fonts[''])

    def _t(self, value):
        """Make any value safe to print with the current font."""
        text = '' if value is None else str(value)
        text = text.replace('\r', '').replace('\t', '    ')
        if self.unicode_ok:
            if self.codepoints is not None:
                text = ''.join(
                    character if (character == '\n' or ord(character) in self.codepoints) else '?'
                    for character in text
                )
            return text
        for source, replacement in _REPLACEMENTS.items():
            text = text.replace(source, replacement)
        return text.encode('latin-1', 'replace').decode('latin-1')

    def _fit(self, text, width):
        """Shorten text so it fits in a table cell."""
        text = self._t(text)
        if self.get_string_width(text) <= width - 2:
            return text
        while text and self.get_string_width(text + '...') > width - 2:
            text = text[:-1]
        return text + '...'

    # ---- cursor-safe drawing helpers ------------------------------------
    def line(self, text, h=7, align=''):
        """One line of text, then move to the start of the next line."""
        if IS_FPDF2:
            self.cell(0, h, self._t(text), new_x='LMARGIN', new_y='NEXT', align=align)
        else:
            self.cell(0, h, self._t(text), ln=1, align=align)

    def paragraph(self, text, h=6):
        """Wrapped text, then move to the start of the next line."""
        if IS_FPDF2:
            self.multi_cell(0, h, self._t(text), align='L', new_x='LMARGIN', new_y='NEXT')
        else:
            self.multi_cell(0, h, self._t(text), align='L')

    def cell_text(self, width, h, text, border=0, fill=False, align=''):
        """A cell that leaves the cursor to its right (for table rows)."""
        self.cell(width, h, self._t(text), border=border, fill=fill, align=align)

    def set_body(self, size=10, style=''):
        self.set_font(self.base_font, style, size)
        self.set_text_color(*COLORS['text'])

    # ---- page furniture --------------------------------------------------
    def header(self):
        self.set_fill_color(*COLORS['dark'])
        if self.page_no() == 1:
            self.rect(0, 0, self.w, 38, 'F')
            self.set_text_color(255, 255, 255)
            self.set_font(self.base_font, 'B', 18)
            self.set_xy(0, 9)
            self.cell(self.w, 10, 'Data Platform Intelligence Report', align='C')
            self.set_font(self.base_font, '', 10)
            self.set_xy(0, 21)
            self.cell(self.w, 8, 'Automated data analysis and machine-learning results', align='C')
            self.set_y(46)
        else:
            self.rect(0, 0, self.w, 12, 'F')
            self.set_text_color(255, 255, 255)
            self.set_font(self.base_font, '', 8)
            self.set_xy(10, 4)
            self.cell(100, 5, 'Data Platform Intelligence Report')
            self.set_xy(self.w - 110, 4)
            self.cell(100, 5, self._fit(self.dataset_name, 100), align='R')
            self.set_y(20)
        self.set_body()

    def footer(self):
        self.set_y(-15)
        self.set_font(self.base_font, 'I', 8)
        self.set_text_color(*COLORS['muted'])
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    # ---- building blocks -------------------------------------------------
    def ensure_space(self, height):
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def section(self, title):
        """Numbered section heading, kept together with the content that follows."""
        self.section_number += 1
        self.ensure_space(36)
        self.set_font(self.base_font, 'B', 14)
        self.set_text_color(*COLORS['purple'])
        self.line(f'{self.section_number}. {title}', h=10)
        self.ln(1)
        self.set_body()

    def subheading(self, title):
        self.ensure_space(24)
        self.set_font(self.base_font, 'B', 11)
        self.set_text_color(*COLORS['text'])
        self.line(title, h=8)
        self.set_body()

    def key_values(self, data):
        self.set_body(10)
        for key, value in data.items():
            self.ensure_space(8)
            self.set_font(self.base_font, 'B', 10)
            self.cell_text(58, 7, f'{key}:')
            self.set_font(self.base_font, '', 10)
            if IS_FPDF2:
                self.multi_cell(0, 7, self._t(value), align='L', new_x='LMARGIN', new_y='NEXT')
            else:
                self.multi_cell(0, 7, self._t(value), align='L')
        self.ln(3)

    def bullet_list(self, values, empty_text='None detected.', limit=MAX_LIST_ITEMS):
        self.set_body(10)
        values = list(values)
        if not values:
            self.line(empty_text, h=7)
            return
        for value in values[:limit]:
            self.paragraph(f'- {value}', h=6)
        if len(values) > limit:
            self.paragraph(f'... and {len(values) - limit} more', h=6)

    def table(self, headers, rows, widths, row_fills=None, bold_rows=(), right_columns=None):
        """
        rows: list of lists of strings. row_fills: optional {(row, col): (r, g, b)}.
        right_columns: column indexes to right-align (default: every column but the first).
        The header row is repeated when the table continues on a new page.
        """
        row_height = 7
        if right_columns is None:
            right_columns = set(range(1, len(headers)))

        def draw_header():
            self.set_font(self.base_font, 'B', 9)
            self.set_fill_color(238, 240, 250)
            self.set_text_color(*COLORS['text'])
            for header, width in zip(headers, widths):
                self.cell_text(width, row_height, self._fit(header, width), border=1, fill=True)
            self.ln(row_height)

        self.ensure_space(row_height * 3)
        draw_header()
        for row_index, row in enumerate(rows):
            if self.get_y() + row_height > self.page_break_trigger:
                self.add_page()
                draw_header()
            self.set_font(self.base_font, 'B' if row_index in bold_rows else '', 9)
            self.set_text_color(*COLORS['text'])
            for column, (value, width) in enumerate(zip(row, widths)):
                fill_color = (row_fills or {}).get((row_index, column))
                if fill_color:
                    self.set_fill_color(*fill_color)
                self.cell_text(width, row_height, self._fit(value, width), border=1,
                               fill=bool(fill_color), align='R' if column in right_columns else '')
            self.ln(row_height)
        self.set_body()
        self.ln(4)

    def score_fill(self, score):
        if score is None:
            return (229, 231, 235)
        if score >= 0.9:
            return (220, 252, 231)
        if score >= 0.7:
            return (254, 249, 195)
        return (254, 226, 226)

    def add_chart(self, title, filename, max_width=170):
        """Embed a chart scaled to fit the page. Returns False if the file is missing."""
        path = os.path.join(self.chart_dir, filename)
        if not os.path.exists(path):
            return False

        flat_path, size = self._flatten_image(path)
        aspect = (size[1] / size[0]) if size else 0.7
        usable_width = self.w - self.l_margin - self.r_margin
        width = min(max_width, usable_width)
        full_page_height = self.h - 20 - 18 - 16  # below running header, above footer, minus title
        height = width * aspect
        if height > full_page_height:
            width = full_page_height / aspect
            height = full_page_height

        self.ensure_space(height + 16)
        self.subheading(title)
        top = self.get_y()
        left = self.l_margin + (usable_width - width) / 2
        self.image(flat_path, x=left, y=top, w=width)
        self.set_y(top + height + 6)
        return True

    def _flatten_image(self, path):
        """Return (path, (width, height)) with transparency flattened onto white."""
        try:
            from PIL import Image
        except ImportError:
            return path, None
        try:
            with Image.open(path) as image:
                size = image.size
                rgba = image.convert('RGBA')
                background = Image.new('RGB', rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[3])
            handle, temporary = tempfile.mkstemp(suffix='.png')
            os.close(handle)
            background.save(temporary, format='PNG')
            self.temp_files.append(temporary)
            return temporary, size
        except Exception:
            return path, None

    def cleanup(self):
        for path in self.temp_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self.temp_files = []


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------
def _best_metrics(ml_results):
    metrics = ml_results.get('metrics') or {}
    best_model = ml_results.get('best_model')
    if best_model in metrics:
        return best_model, metrics[best_model]
    for name, values in metrics.items():  # tolerate combined names such as "A / B"
        if best_model and name in str(best_model):
            return name, values
    return best_model, {}


def _executive_summary(pdf, df, ml_results, cleaning_summary):
    pdf.set_font(pdf.base_font, 'B', 14)
    pdf.set_text_color(*COLORS['purple'])
    pdf.line('Executive Summary', h=10)
    pdf.ln(1)
    pdf.set_body(10)

    rows, columns = len(df), len(df.columns)
    if not ml_results:
        pdf.paragraph(
            f'The prepared dataset contains {rows:,} rows and {columns:,} columns. '
            'No machine-learning model was trained for this run.', h=6)
        pdf.ln(3)
        return

    problem_type = ml_results.get('prob_type', 'N/A')
    best_name, best_metrics = _best_metrics(ml_results)
    note = str(ml_results.get('selection_note') or '').strip().rstrip('.')
    target = ml_results.get('target')
    top_feature = ((ml_results.get('importance') or [{}])[0]).get('feature')

    parts = []
    if target:
        parts.append(f"The target was '{target}' ({problem_type}).")
    if note and best_name and note.startswith(str(best_name)):
        parts.append(note + '.')   # the note already names the model
    else:
        if best_name:
            parts.append(f'Selected model: {best_name}.')
        if note:
            parts.append(note + '.')
    if problem_type == 'classification':
        parts.append(
            f"On held-out test data it reached accuracy {_format_number(best_metrics.get('accuracy'))} "
            f"and macro-F1 {_format_number(best_metrics.get('macro_f1'))}.")
    else:
        parts.append(
            f"On held-out test data it reached R2 {_format_number(best_metrics.get('r2'))} "
            f"with RMSE {_format_number(best_metrics.get('rmse'))}.")
    if top_feature:
        parts.append(
            f'The feature with the strongest influence on the model was {top_feature}. '
            'Importance scores describe the model, not cause and effect.')
    parts.append(f'The prepared dataset contains {rows:,} rows and {columns:,} columns.')
    pdf.paragraph(' '.join(parts), h=6)
    pdf.ln(3)


def _contents(pdf, plan):
    pdf.set_font(pdf.base_font, 'B', 14)
    pdf.set_text_color(*COLORS['purple'])
    pdf.line('Contents', h=10)
    pdf.ln(1)
    pdf.set_body(10)
    for number, title in enumerate(plan, start=1):
        pdf.line(f'{number}. {title}', h=6)
    pdf.ln(4)


def _dataset_overview(pdf, df, cleaning_summary):
    pdf.section('Dataset Overview')
    numeric_columns = df.select_dtypes(include='number').shape[1]
    overview = {
        'Rows': f'{len(df):,}',
        'Columns': f'{len(df.columns):,}',
        'Numeric columns': str(numeric_columns),
        'Other columns': str(len(df.columns) - numeric_columns),
        'Missing values remaining': f'{int(df.isnull().sum().sum()):,}',
    }
    if cleaning_summary and cleaning_summary.get('initial_shape'):
        overview['Original size'] = _format_shape(cleaning_summary['initial_shape'])
    pdf.key_values(overview)


def _feature_summary(pdf, df, ml_results, cleaning_summary):
    pdf.section('Feature Summary')
    encoded = set(str(c) for c in (cleaning_summary or {}).get('encoded_columns', []))
    quality = (ml_results or {}).get('quality') or {}
    identifiers = set(str(c) for c in quality.get('identifier_columns', []))
    target = str((ml_results or {}).get('target', ''))

    rows = []
    for column in list(df.columns)[:MAX_FEATURE_ROWS]:
        name = str(column)
        note = ''
        if name == target:
            note = 'target'
        elif name in identifiers:
            note = 'identifier (excluded)'
        elif name in encoded:
            note = 'encoded category'
        series = df[column]
        rows.append([name, str(series.dtype), str(int(series.isnull().sum())),
                     str(int(series.nunique(dropna=True))), note])
    pdf.table(['Column', 'Type', 'Missing', 'Unique', 'Note'], rows,
              [FEATURE_TABLE_NAME_WIDTH, 28, 24, 24, 48], right_columns={2, 3})
    if len(df.columns) > MAX_FEATURE_ROWS:
        pdf.set_body(9)
        pdf.line(f'... and {len(df.columns) - MAX_FEATURE_ROWS} more columns not shown.', h=6)
        pdf.ln(2)


def _cleaning_section(pdf, cleaning_summary, df=None):
    pdf.section('Data Cleaning')
    if not cleaning_summary:
        pdf.set_body(10)
        pdf.line('Cleaning details were not recorded for this run.', h=7)
        pdf.ln(3)
        return

    summary = cleaning_summary
    items = []
    if summary.get('initial_shape') and summary.get('final_shape'):
        items.append(f"Size changed from {_format_shape(summary['initial_shape'])} "
                     f"to {_format_shape(summary['final_shape'])}.")
    if 'duplicate_rows_detected' in summary:
        items.append(f"Duplicate rows: {summary.get('duplicate_rows_detected', 0)} detected, "
                     f"{summary.get('duplicate_rows_removed', 0)} removed.")

    strategy_value = summary.get('missing_strategy')
    if strategy_value is not None and str(strategy_value).strip():
        strategy = str(strategy_value).strip().lower()
        if strategy == 'drop':
            items.append(
                "Missing values: rows with gaps were dropped "
                f"({summary.get('rows_dropped_for_missing', 0)} rows)."
            )
        elif strategy in {'none', 'no_imputation', 'keep', 'preserve'}:
            remaining = summary.get('missing_values_remaining')
            if remaining is None and df is not None:
                try:
                    remaining = int(df.isna().sum().sum())
                except Exception:
                    remaining = None
            if remaining is None:
                items.append(
                    "Missing values: no imputation performed; missing values were preserved."
                )
            else:
                items.append(
                    "Missing values: no imputation performed; "
                    f"{int(remaining):,} missing value(s) remain."
                )
        else:
            strategy_labels = {
                'median': 'median imputation',
                'mean': 'mean imputation',
                'mode': 'mode / most-frequent imputation',
            }
            label = strategy_labels.get(strategy, f'{strategy} imputation')
            items.append(
                f"Missing values: {label} applied "
                f"({summary.get('missing_values_filled', 0)} value(s) filled)."
            )

    if summary.get('outliers_removed'):
        items.append(f"Outliers: {summary.get('outlier_rows_removed', 0)} rows removed using "
                     f"{summary.get('outlier_method', 'N/A')}.")
    else:
        items.append('Outlier removal: not applied.')
    if summary.get('dtype_columns_converted'):
        items.append(f"Data types fixed in {len(summary['dtype_columns_converted'])} column(s); "
                     f"{summary.get('placeholders_to_missing', 0)} placeholder value(s) "
                     'such as "?" or "N/A" were treated as missing.')
    if summary.get('empty_columns_dropped'):
        items.append(f"Empty columns dropped: {', '.join(map(str, summary['empty_columns_dropped'][:5]))}.")
    encoded = summary.get('encoded_columns', summary.get('encoders', []))
    if encoded:
        items.append(f'Categorical encoding applied to {len(encoded)} column(s).')
    if summary.get('scaled_columns'):
        items.append(f"Numeric scaling applied to {len(summary['scaled_columns'])} column(s).")
    if summary.get('engine'):
        items.append(f"Cleaning engine: {summary.get('engine')} (version {summary.get('version', 'N/A')}).")
    if summary.get('quality_score_before') is not None and summary.get('quality_score_after') is not None:
        items.append(f"Structural Data Quality Score: {summary['quality_score_before']}% before and "
                     f"{summary['quality_score_after']}% after cleaning. This is an indicative completeness/duplicate-based score, not a measure of factual accuracy.")
    if summary.get('anomaly_rows_flagged') is not None:
        items.append(
            f"Potential anomalies: {summary.get('anomaly_rows_flagged')} rows "
            f"({summary.get('anomaly_percentage', 0)}%) flagged by "
            f"{summary.get('anomaly_method_detail', summary.get('anomaly_method', 'the anomaly detector'))} and retained for review."
        )
        if summary.get('anomaly_features_used'):
            items.append("Anomaly features considered: " + ", ".join(summary['anomaly_features_used']) + ".")
        if summary.get('anomaly_features_excluded'):
            items.append("Excluded from anomaly detection: " + ", ".join(summary['anomaly_features_excluded']) + ".")
        if summary.get('anomaly_group_columns'):
            items.append("Contextual comparison groups: " + ", ".join(summary['anomaly_group_columns']) + f" (selection: {summary.get('anomaly_grouping_mode', 'automatic')}).")
        elif summary.get('anomaly_grouping_mode'):
            items.append("Anomaly grouping: " + str(summary['anomaly_grouping_mode']) + ".")
        if summary.get('anomaly_policy_detail'):
            items.append(summary['anomaly_policy_detail'])
        if summary.get('anomaly_selected_method'):
            items.append(f"Automatically selected anomaly detector: {summary['anomaly_selected_method']}.")
        if summary.get('anomaly_algorithm_selection_note'):
            items.append(summary['anomaly_algorithm_selection_note'])
    for detail in summary.get('missing_category_details', [])[:12]:
        items.append(
            f"{detail.get('column')}: {detail.get('values_replaced')} missing values were labelled "
            f"{detail.get('replacement')} because the original category was unavailable; "
            "this does not mean the category was absent."
        )
    pdf.bullet_list(items)

    components_before = summary.get('quality_components_before') or {}
    components_after = summary.get('quality_components_after') or {}
    if components_before and components_after:
        pdf.ln(1)
        pdf.subheading('Quality component breakdown')
        pdf.bullet_list([
            f"Completeness: {components_before.get('completeness_pct', 'N/A')}% before; "
            f"{components_after.get('completeness_pct', 'N/A')}% after.",
            f"Duplicate-free rate: {components_before.get('duplicate_free_pct', 'N/A')}% before; "
            f"{components_after.get('duplicate_free_pct', 'N/A')}% after.",
            "The indicative score does not measure validity or factual accuracy."
        ])

    decisions = summary.get('decisions') or []
    if decisions:
        pdf.ln(1)
        pdf.subheading('Adaptive imputation decisions')
        for decision in decisions[:12]:
            benchmark = decision.get('benchmark') or {}
            metrics = benchmark.get('metrics') or {}
            metric_text = []
            for method in ('mean', 'median', 'knn', 'missforest', 'mice'):
                values = metrics.get(method) or {}
                if values.get('mae') is not None:
                    metric_text.append(
                        f"{method}: MAE {values['mae']}, RMSE {values['rmse']}"
                    )
            line = (
                f"{decision.get('column')}: selected {decision.get('selected_method')}. "
                f"{decision.get('reason')}"
            )
            if metric_text:
                line += " Benchmark — " + "; ".join(metric_text) + "."
            if benchmark.get('repeats'):
                line += f" Repeated masking runs: {benchmark.get('repeats')}; predictors: {', '.join(benchmark.get('predictor_features', [])) or 'none'}."
            pdf.paragraph(line, h=5)
        pdf.ln(2)

    detector_comparison = summary.get('anomaly_algorithm_comparison') or {}
    if detector_comparison:
        pdf.ln(1)
        pdf.subheading('Anomaly detector comparison')
        for method, values in detector_comparison.items():
            if values.get('status') == 'unavailable':
                pdf.paragraph(f"{method}: unavailable — {values.get('unavailable_reason', 'not supported for this dataset')}.", h=5)
            else:
                pdf.paragraph(
                    f"{method}: {values.get('flagged_rows')} flagged ({values.get('flagged_pct')}%); "
                    f"consensus agreement {values.get('mean_balanced_agreement')}; "
                    f"selection score {values.get('selection_score')}. "
                    f"{'SELECTED' if method == summary.get('anomaly_selected_method') else ''}", h=5
                )
        explanations = summary.get('anomaly_explanations') or []
        if explanations:
            pdf.ln(1)
            pdf.subheading('Examples of anomaly explanations')
            for explanation in explanations[:8]:
                pdf.paragraph(f"Row {explanation.get('row_index')}: {explanation.get('reason')}", h=5)
        pdf.paragraph('Consensus scores are not verified accuracy. Labelled anomaly examples are needed to calculate precision and recall.', h=5)

    synthetic = summary.get('synthetic_imputation_benchmark') or {}
    if synthetic:
        pdf.ln(1)
        pdf.subheading('Controlled synthetic imputation benchmark')
        adaptive = synthetic.get('adaptive_method') or {}
        mean_values = synthetic.get('mean_baseline') or {}
        median_values = synthetic.get('median_baseline') or {}
        pdf.bullet_list([
            f"Masked values: {synthetic.get('masked_cells', 'N/A')}.",
            f"Adaptive: MAE {adaptive.get('mae', 'N/A')}, RMSE {adaptive.get('rmse', 'N/A')}.",
            f"Mean baseline: MAE {mean_values.get('mae', 'N/A')}, RMSE {mean_values.get('rmse', 'N/A')}.",
            f"Median baseline: MAE {median_values.get('mae', 'N/A')}, RMSE {median_values.get('rmse', 'N/A')}.",
            str(synthetic.get('note', 'Synthetic benchmark only.'))
        ])

    if summary.get('warnings'):
        pdf.ln(2)
        pdf.subheading('Notes from the cleaning step')
        pdf.bullet_list(summary['warnings'])
    pdf.ln(3)


def _quality_section(pdf, ml_results):
    pdf.section('Data Quality Warnings')
    quality = ml_results.get('quality') or {}
    pdf.bullet_list(quality.get('warnings', []))
    pairs = quality.get('high_correlation_pairs') or []
    if pairs:
        pdf.ln(2)
        pdf.subheading('Most correlated feature pairs')
        pdf.bullet_list([f"{p['left']} vs {p['right']}: {p['correlation']:.3f}" for p in pairs])
    suspects = quality.get('leakage_suspects') or []
    if suspects:
        pdf.ln(2)
        pdf.subheading('Possible target leakage')
        pdf.bullet_list([f"{s['feature']}: correlation with target {s['correlation']:.3f}" for s in suspects])
    pdf.ln(4)


def _split_section(pdf, ml_results):
    pdf.section('Train/Test Split')
    split = ml_results.get('split_info') or {}
    if not split:
        pdf.set_body(10)
        pdf.line('Training metadata was not available for this run.', h=7)
        pdf.ln(3)
        return

    test_size = split.get('test_size', 0.2)
    details = {
        'Training samples': split.get('train_size'),
        'Test samples': split.get('test_count'),
        'Split ratio': f'{1 - test_size:.0%} / {test_size:.0%}',
        'Shuffled': 'yes' if split.get('shuffle') else 'no',
        'Stratified': 'yes' if split.get('stratified') else 'no',
    }
    if 'scaled' in split:
        details['Features standardised'] = 'yes (using training rows only)' if split['scaled'] else 'no'
    if split.get('imputed_cells'):
        details['Missing values imputed'] = (
            f"{split['imputed_cells']} (inside training-pipeline preprocessing)"
        )
    imputation_details = (
        split.get('imputation_strategies')
        or ml_results.get('imputation_strategies')
        or split.get('imputation_summary')
    )
    if imputation_details:
        if isinstance(imputation_details, dict):
            details['Imputation strategies'] = '; '.join(
                f'{key}: {value}' for key, value in imputation_details.items()
            )
        else:
            details['Imputation strategies'] = str(imputation_details)
    if split.get('n_features') is not None:
        details['Features used'] = split['n_features']
    dropped = split.get('dropped_columns') or {}
    labels = {'empty': 'Empty columns', 'datetime': 'Date/time columns', 'text_like': 'Text-like columns'}
    for key, label in labels.items():
        if dropped.get(key):
            details[f'{label} excluded'] = ', '.join(map(str, dropped[key][:5]))
    if split.get('identifier_columns'):
        details['Identifier columns excluded'] = ', '.join(map(str, split['identifier_columns'][:5]))
    if split.get('one_hot_encoded_codes'):
        details['One-hot encoded'] = ', '.join(map(str, split['one_hot_encoded_codes'][:5]))
    pdf.key_values(details)

    counts = ml_results.get('class_counts')
    if counts:
        pdf.subheading('Class counts in the training set')
        pdf.table(['Class', 'Rows'], [[str(k), str(v)] for k, v in counts.items()], [80, 30])


def _model_results_section(pdf, ml_results):
    pdf.section('Machine Learning Results')
    problem_type = ml_results.get('prob_type', 'N/A')
    best_name, _ = _best_metrics(ml_results)
    pdf.key_values({
        'Target': ml_results.get('target', 'N/A'),
        'Problem type': problem_type,
        'Selected model': best_name or 'N/A',
    })

    metrics = ml_results.get('metrics') or {}
    if problem_type == 'classification':
        headers = ['Model', 'Accuracy', 'Macro-F1', 'CV mean', 'CV std', 'ROC-AUC']
        widths = [46, 28, 28, 28, 28, 28]
        keys = ['accuracy', 'macro_f1', 'cv_mean', 'cv_std', 'roc_auc']
        fill_keys = {'accuracy', 'macro_f1'}
    else:
        headers = ['Model', 'R2', 'MAE', 'RMSE', 'CV mean', 'CV std']
        widths = [46, 28, 28, 28, 28, 28]
        keys = ['r2', 'mae', 'rmse', 'cv_mean', 'cv_std']
        fill_keys = {'r2'}

    rows, fills, bold = [], {}, []
    for row_index, (name, values) in enumerate(metrics.items()):
        label = f'{name} *' if name == best_name else name
        if name == best_name:
            bold.append(row_index)
        rows.append([label] + [_format_number(values.get(key)) for key in keys])
        for column, key in enumerate(keys, start=1):
            if key in fill_keys:
                fills[(row_index, column)] = pdf.score_fill(values.get(key))
    if rows:
        pdf.table(headers, rows, widths, row_fills=fills, bold_rows=bold)

    metric_name = next((v.get('cv_metric') for v in metrics.values() if v.get('cv_metric')), None)
    pdf.set_body(8)
    note = '* marks the selected model.'
    if metric_name:
        note += f' CV mean/std are cross-validation scores ({metric_name}) on the training set only.'
    pdf.paragraph(note, h=5)

    errors = ml_results.get('model_errors') or {}
    if errors:
        pdf.ln(2)
        pdf.subheading('Models that could not be trained')
        pdf.bullet_list([f'{name}: {message}' for name, message in errors.items()])
    pdf.ln(4)


def _diagnostics_section(pdf, ml_results, files):
    pdf.section('Diagnostics')
    for title, filename in files:
        pdf.add_chart(title, filename)


def _importance_section(pdf, ml_results, has_chart):
    pdf.section('Feature Importance')
    importance = ml_results.get('importance') or []
    rows = [[str(item['feature']), _format_number(item['importance'])] for item in importance[:10]]
    pdf.table(['Feature', 'Importance'], rows, [110, 40])
    pdf.set_body(8)
    pdf.paragraph('Importance scores describe how much the model relies on a feature. '
                  'They are a heuristic and do not prove cause and effect.', h=5)
    pdf.ln(2)
    if has_chart:
        pdf.add_chart('Top features', 'feature_importance.png')


def _visual_section(pdf, files):
    pdf.section('Visual Analysis')
    for title, filename in files:
        pdf.add_chart(title, filename)


def _safe_param_value(value, max_length=180):
    """Format a hyperparameter value without dumping huge object representations."""
    if value is None or isinstance(value, (bool, int, float, str)):
        text = repr(value) if isinstance(value, str) else str(value)
    elif isinstance(value, (list, tuple, set)):
        text = repr(value)
    elif isinstance(value, dict):
        text = repr(value)
    else:
        cls_name = value.__class__.__name__
        if hasattr(value, 'get_params'):
            text = f'<{cls_name} estimator>'
        else:
            text = str(value)
    text = text.replace("\\n", " ")
    if len(text) > max_length:
        text = text[:max_length - 3] + '...'
    return text


def _readable_hyperparameters(params, prefix=""):
    """Flatten parameter dictionaries and format estimator/Pipeline objects."""
    if isinstance(params, dict):
        output = []
        for key, value in params.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                output.extend(_readable_hyperparameters(value, prefix=label))
            elif isinstance(value, (list, tuple)) and value and all(
                    isinstance(item, (list, tuple)) and len(item) == 2
                    for item in value):
                # Handle serialized Pipeline steps such as
                # [['preprocessor', 'ColumnTransformer'], ['model', 'SVC']].
                formatted = "; ".join(
                    f"{item[0]}: {item[1]}" for item in value
                )
                output.append((label, _safe_param_value(formatted)))
            elif hasattr(value, 'named_steps') or (
                    hasattr(value, 'get_params') and not isinstance(value, type)):
                nested = _readable_hyperparameters(value)
                if nested:
                    output.append((f'{label} type', value.__class__.__name__))
                    output.extend((f'{label}.{nested_key}', nested_value)
                                  for nested_key, nested_value in nested)
                else:
                    output.append((label, _safe_param_value(value)))
            else:
                output.append((label, _safe_param_value(value)))
        return output

    if hasattr(params, 'named_steps'):
        steps = params.named_steps
        output = []
        last_step = list(steps.keys())[-1] if steps else None
        for step_name, estimator in steps.items():
            if hasattr(estimator, 'get_params'):
                if step_name == 'model' or step_name == last_step:
                    output.append((f'{step_name} estimator', estimator.__class__.__name__))
                    try:
                        estimator_params = estimator.get_params(deep=False)
                    except Exception:
                        estimator_params = {}
                    output.extend(
                        (f'{step_name}.{key}', _safe_param_value(value))
                        for key, value in estimator_params.items()
                    )
                else:
                    output.append((str(step_name), estimator.__class__.__name__))
            else:
                output.append((str(step_name), _safe_param_value(estimator)))
        return output

    if hasattr(params, 'get_params'):
        try:
            estimator_params = params.get_params(deep=False)
        except Exception:
            estimator_params = {}
        return [(str(key), _safe_param_value(value))
                for key, value in estimator_params.items()]

    return [('Configuration', _safe_param_value(params))]


def _hyperparameter_explanations(model_name, params):
    """Describe the most meaningful hyperparameters in plain language."""
    name = str(model_name).lower()
    descriptions = {
        "C": "Regularisation strength: balances training errors against a wider decision margin.",
        "kernel": "The function SVM uses to form its decision boundary.",
        "gamma": "Controls the reach of each example's influence for an RBF SVM.",
        "degree": "Polynomial degree when a polynomial SVM kernel is used.",
        "n_estimators": "Number of trees or boosting stages in the ensemble.",
        "max_depth": "Maximum tree depth; deeper trees can learn more complex patterns.",
        "min_samples_split": "Minimum samples required before a tree node can split.",
        "min_samples_leaf": "Minimum samples required in a leaf node.",
        "max_features": "Features considered at each tree split.",
        "criterion": "Rule used to measure the quality of a tree split.",
        "bootstrap": "Whether trees use bootstrap samples.",
        "learning_rate": "Step size used while updating predictions during iterative learning.",
        "subsample": "Fraction of training rows sampled for each boosting stage.",
        "colsample_bytree": "Fraction of features sampled for each boosted tree.",
        "reg_alpha": "L1 regularisation strength.",
        "reg_lambda": "L2 regularisation strength.",
        "hidden_layer_sizes": "Number of hidden layers and neurons in each layer.",
        "activation": "Activation function used by neural-network neurons.",
        "solver": "Optimisation algorithm used to fit the model.",
        "max_iter": "Maximum number of optimisation or training iterations.",
        "penalty": "Type of regularisation applied to model coefficients.",
        "fit_intercept": "Whether the model learns an intercept term.",
        "class_weight": "Whether classes receive different weights during training.",
        "probability": "Whether SVM probability estimates are enabled.",
        "random_state": "Seed used to make random choices reproducible.",
        "tol": "Stopping tolerance for optimisation.",
        "alpha": "Regularisation strength used by the estimator.",
        "early_stopping": "Whether training may stop when validation performance stops improving.",
    }

    if "logistic" in name:
        preferred = ["C", "penalty", "solver", "max_iter", "class_weight", "fit_intercept"]
    elif "decision tree" in name:
        preferred = ["criterion", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"]
    elif "random forest" in name:
        preferred = ["n_estimators", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "class_weight"]
    elif "support vector" in name or name.strip() == "svm" or "svc" in name:
        preferred = ["C", "kernel", "gamma", "degree", "class_weight", "probability"]
    elif "neural" in name or "mlp" in name:
        preferred = ["hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate", "max_iter", "early_stopping"]
    elif "xgboost" in name or "xgb" in name:
        preferred = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    else:
        preferred = ["n_estimators", "max_depth", "C", "kernel", "learning_rate", "max_iter", "random_state"]

    values = {}
    for key, value in params:
        short_key = str(key).split(".")[-1].split("__")[-1]
        if short_key not in values:
            values[short_key] = value

    result = []
    for key in preferred:
        if key not in values or str(values[key]).strip().lower() in {"none", "null"}:
            continue
        result.append((key, values[key], descriptions.get(key, "Model configuration setting.")))
    return result


def _hyperparameter_section(pdf, ml_results):
    pdf.section('Hyperparameters: Model Configuration')
    hyperparameters = ml_results.get('hyperparameters') or {}
    if not hyperparameters:
        pdf.line('Hyperparameter details were not provided for this run.', h=7)
        return

    pdf.set_body(9)
    pdf.paragraph(
        'Hyperparameters are settings that control how a model learns. '
        'They describe the configuration used in this run and do not, by themselves, '
        'mean that automatic hyperparameter tuning was performed.',
        h=5
    )
    pdf.ln(2)

    for name, params in hyperparameters.items():
        pdf.ensure_space(30)
        pdf.set_font(pdf.base_font, 'B', 10)
        pdf.set_text_color(*COLORS['text'])
        pdf.line(str(name), h=7)
        pdf.set_font(pdf.base_font, '', 8)

        readable = _readable_hyperparameters(params)
        for key, value in readable:
            if str(key).split(".")[-1].lower() == "estimator":
                pdf.paragraph(f"Estimator: {value}", h=5)
                break

        important = _hyperparameter_explanations(name, readable)
        if important:
            for key, value, explanation in important:
                pdf.paragraph(f"{key} = {value}", h=5)
                pdf.set_font(pdf.base_font, 'I', 8)
                pdf.paragraph(f"Meaning: {explanation}", h=4)
                pdf.set_font(pdf.base_font, '', 8)
        else:
            ignored = {"memory", "steps", "transform_input", "verbose", "n_jobs"}
            shown = 0
            for key, value in readable:
                short_key = str(key).split(".")[-1].split("__")[-1]
                if short_key.lower() in ignored or str(value).lower() in {"none", "null"}:
                    continue
                pdf.paragraph(f"{key} = {value}", h=5)
                shown += 1
                if shown >= 8:
                    break
            if shown == 0:
                pdf.paragraph("No key hyperparameters were available for display.", h=5)
        pdf.ln(3)

    pdf.set_font(pdf.base_font, 'I', 8)
    pdf.paragraph(
        'Note: Values shown are the settings recorded for this run. A value may be a '
        'library default unless the application records that it was tuned.',
        h=4
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate_full_report(df, output_path, ml_results=None, cleaning_summary=None,
                         metadata=None, chart_dir=None):
    pdf = None
    try:
        charts = chart_dir or CHART_DIR
        metadata = metadata or {}
        pdf = DataPlatformReport(charts)
        pdf.dataset_name = str(metadata.get('filename', ''))

        def available(files):
            return [(title, name) for title, name in files
                    if os.path.exists(os.path.join(charts, name))]

        # Charts are only shown when they belong to this run.
        diagnostics, importance_chart = [], False
        if ml_results:
            if ml_results.get('prob_type') == 'classification':
                wanted = [('Confusion matrix', 'confusion_matrix.png'),
                          ('ROC curve', 'roc_curve.png'),
                          ('Precision-recall curve', 'precision_recall_curve.png')]
            else:
                wanted = [('Actual vs predicted values', 'actual_vs_predicted.png')]
            diagnostics = available([('Model comparison', 'model_comparison.png')] + wanted)
            importance_chart = bool(available([('', 'feature_importance.png')]))
        eda_charts = available([
            ('Missing values overview', 'missing_values.png'),
            ('Correlation heatmap', 'correlation_heatmap.png'),
            ('Data distribution', 'dist_column.png'),
            ('Outlier analysis', 'outliers.png'),
        ])

        plan = ['Dataset Overview', 'Feature Summary', 'Data Cleaning']
        if ml_results:
            plan += ['Data Quality Warnings', 'Train/Test Split', 'Machine Learning Results']
            if diagnostics:
                plan.append('Diagnostics')
            if ml_results.get('importance'):
                plan.append('Feature Importance')
        if eda_charts:
            plan.append('Visual Analysis')
        if ml_results and ml_results.get('hyperparameters'):
            plan.append('Hyperparameters Appendix')

        pdf.add_page()
        pdf.set_body(10)
        pdf.section_number = 0

        # Report metadata
        pdf.set_font(pdf.base_font, 'B', 14)
        pdf.set_text_color(*COLORS['purple'])
        pdf.line('Report Metadata', h=10)
        pdf.key_values({
            'Dataset': metadata.get('filename', 'Unknown'),
            'Generated': _format_timestamp(metadata.get('generated_at')),
            'Pipeline run ID': metadata.get('run_id', 'Unknown'),
        })

        _executive_summary(pdf, df, ml_results, cleaning_summary)
        _contents(pdf, plan)

        _dataset_overview(pdf, df, cleaning_summary)
        _feature_summary(pdf, df, ml_results, cleaning_summary)
        _cleaning_section(pdf, cleaning_summary, df=df)
        if ml_results:
            _quality_section(pdf, ml_results)
            _split_section(pdf, ml_results)
            _model_results_section(pdf, ml_results)
            if diagnostics:
                _diagnostics_section(pdf, ml_results, diagnostics)
            if ml_results.get('importance'):
                _importance_section(pdf, ml_results, importance_chart)
        if eda_charts:
            _visual_section(pdf, eda_charts)
        if ml_results and ml_results.get('hyperparameters'):
            _hyperparameter_section(pdf, ml_results)

        pdf.output(output_path)
    except Exception as error:
        raise ReportError(f'Failed to generate report: {error}') from error
    finally:
        if pdf is not None:
            pdf.cleanup()