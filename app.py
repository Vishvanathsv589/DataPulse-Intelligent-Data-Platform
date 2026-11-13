# app.py
"""
Data Platform web app.

Layout of per-user data (nothing user-specific is shared between sessions):

    user_data/sessions/<session id>/
        dataset.csv|xlsx|xls     the uploaded file
        cleaned.csv              output of the cleaning step
        cleaning_summary.json    what the cleaner did (kept out of the cookie)
        ml_results.json          model results (kept out of the cookie)
        report.pdf               the generated report
        charts/                  every chart for this session

The browser cookie only holds the session id, the display file name and the
small pipeline-status dict, so it can never outgrow the ~4 KB cookie limit.

Charts are still requested as /static/generated/<name>.png, so the existing
templates keep working, but that URL is answered from the visitor's own chart
folder instead of a shared one.
"""
import inspect
import json
import math
import os
import re
import secrets
import shutil
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from flask import (
    Flask, abort, flash, redirect, render_template, request, send_file,
    send_from_directory, session, url_for
)
from werkzeug.utils import secure_filename

from modules import reporter as reporter_module
from modules.cleaner import full_clean_pipeline, fix_dtypes
from modules.intelligent_cleaner import adaptive_clean, profile_dataset
from modules.cleaning_evaluator import compare_quality, evaluate_injected_numeric_noise
from modules.eda import (
    plot_all_eda_themes, plot_feature_importance, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, plot_actual_vs_predicted,
    plot_model_comparison
)
from modules.ml_pipeline import (
    detect_problem_type, prepare_features, train_classification_models,
    train_regression_models, compare_models, get_best_model_selection,
    feature_importance, data_quality_report, collect_model_errors,
    get_hyperparameters
)
from modules.reporter import generate_full_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_secret_key():
    """
    Use the SECRET_KEY environment variable if set. Otherwise generate a random
    key once and store it in .secret_key so sessions survive restarts.
    (Add .secret_key to your .gitignore.)
    """
    env_key = os.environ.get('SECRET_KEY')
    if env_key:
        return env_key

    key_path = os.path.join(BASE_DIR, '.secret_key')
    try:
        with open(key_path, 'r') as f:
            key = f.read().strip()
            if key:
                return key
    except OSError:
        pass

    key = secrets.token_hex(32)
    try:
        with open(key_path, 'w') as f:
            f.write(key)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
    except OSError:
        pass  # Fall back to an in-memory key (sessions reset on restart)
    return key


app.secret_key = load_secret_key()
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Set SESSION_COOKIE_SECURE=1 in the environment when serving over HTTPS.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE') == '1'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

DATA_DIR = os.environ.get('DATA_DIR') or os.path.join(BASE_DIR, 'user_data')
SESSIONS_DIR = os.path.join(DATA_DIR, 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
SESSION_ID_PATTERN = re.compile(r'^[0-9a-f]{16,64}$')

# Hosts (besides this server's own) allowed to send requests, e.g. behind a proxy:
# TRUSTED_HOSTS="data.example.com,www.example.com"
TRUSTED_HOSTS = {h.strip() for h in os.environ.get('TRUSTED_HOSTS', '').split(',') if h.strip()}

# Exact frontend origins allowed to call this API with browser credentials.
# Example: FRONTEND_ORIGINS="https://datapulse-frontend.onrender.com"
FRONTEND_ORIGINS = {
    origin.strip().rstrip('/')
    for origin in os.environ.get('FRONTEND_ORIGINS', '').split(',')
    if origin.strip()
}

# Idle sessions (and their files) are deleted after this many hours.
SESSION_MAX_AGE_SECONDS = int(float(os.environ.get('SESSION_MAX_AGE_HOURS', '24')) * 3600)
CLEANUP_INTERVAL_SECONDS = 3600

EDA_CHART_FILES = {
    'missing_values': 'missing_values.png',
    'distribution': 'dist_column.png',
    'correlation': 'correlation_heatmap.png',
    'outliers': 'outliers.png',
}
ML_CHART_FILES = (
    'confusion_matrix.png',
    'roc_curve.png',
    'precision_recall_curve.png',
    'actual_vs_predicted.png',
    'model_comparison.png',
    'feature_importance.png',
)
MODEL_CHECKBOXES = {
    'model_lr': 'Logistic Regression',  # becomes Linear Regression for regression problems
    'model_dt': 'Decision Tree',
    'model_rf': 'Random Forest',
    'model_svm': 'SVM',
    'model_mlp': 'Neural Network',
    'model_xgb': 'XGBoost',
}
MODEL_SCORE_KEYS = {'classification': 'macro_f1', 'regression': 'r2'}
ARRAY_METRICS = {'y_true', 'y_pred', 'probabilities'}  # large; only used for charts

report_lock = threading.Lock()
progress_lock = threading.Lock()
progress_by_session = {}

def set_progress(percent, message, title=None, state='running'):
    """Publish completed-work progress for the current browser session."""
    sid = get_session_id()
    payload = {'percent': max(0, min(100, int(percent))), 'message': str(message),
               'title': title or 'Processing your request...', 'state': state}
    with progress_lock:
        progress_by_session[sid] = payload

_cleanup_lock = threading.Lock()
_last_cleanup = 0.0


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
def is_cross_origin():
    """
    True when the request was started by a different website. Browsers always
    send Origin (or Referer) on cross-site requests; Sec-Fetch-Site covers the
    case where both are suppressed. Typed URLs, bookmarks and curl send none of
    these and are allowed.
    """
    source = request.headers.get('Origin') or request.headers.get('Referer')
    if source:
        host = urlparse(source).netloc
        return host != request.host and host not in TRUSTED_HOSTS
    return request.headers.get('Sec-Fetch-Site') == 'cross-site'


@app.before_request
def block_cross_origin_writes():
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and is_cross_origin():
        abort(403)


@app.before_request
def keep_session_alive():
    """Touch the session folder so active users are never cleaned up."""
    session_id = session.get('session_id')
    if session_id and SESSION_ID_PATTERN.match(str(session_id)):
        try:
            os.utime(os.path.join(SESSIONS_DIR, session_id))
        except OSError:
            pass
    cleanup_old_sessions()


@app.after_request
def add_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    return response


@app.after_request
def add_frontend_cors(response):
    """Allow credentialed browser requests from the configured frontend only."""
    origin = request.headers.get('Origin')

    if origin and origin.rstrip('/') in FRONTEND_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers.add('Vary', 'Origin')

    return response


def same_origin_only(view):
    """GET routes that change state (reset, logout) must not be triggerable from other sites."""
    def wrapper(*args, **kwargs):
        if is_cross_origin():
            abort(403)
        return view(*args, **kwargs)
    wrapper.__name__ = view.__name__
    return wrapper


@app.errorhandler(413)
def file_too_large(_error):
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# Per-session storage
# ---------------------------------------------------------------------------
def get_session_id():
    """Get or create this visitor's session id."""
    session_id = session.get('session_id')
    if not session_id or not SESSION_ID_PATTERN.match(str(session_id)):
        session_id = uuid.uuid4().hex
        session['session_id'] = session_id
    return session_id


def session_path(*parts, create=False):
    """Path inside this session's folder. create=True makes the folder."""
    base = os.path.join(SESSIONS_DIR, get_session_id())
    if create:
        os.makedirs(base, exist_ok=True)
    return os.path.join(base, *parts)


def chart_dir(create=True):
    directory = session_path('charts')
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory


def reset_session_files():
    """Delete everything stored for this session."""
    session_id = session.get('session_id')
    if session_id and SESSION_ID_PATTERN.match(str(session_id)):
        shutil.rmtree(os.path.join(SESSIONS_DIR, session_id), ignore_errors=True)


def cleanup_old_sessions():
    """Delete session folders that have been idle too long (at most once an hour)."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < CLEANUP_INTERVAL_SECONDS:
        return
    if not _cleanup_lock.acquire(blocking=False):
        return
    try:
        _last_cleanup = now
        for name in os.listdir(SESSIONS_DIR):
            path = os.path.join(SESSIONS_DIR, name)
            try:
                if os.path.isdir(path) and now - os.path.getmtime(path) > SESSION_MAX_AGE_SECONDS:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                continue
    finally:
        _cleanup_lock.release()


def to_jsonable(value):
    """Make results JSON-safe (numpy types, NaN, non-string keys)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return str(value)


def save_state(name, data):
    path = session_path(name, create=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(to_jsonable(data), handle)
    os.replace(temporary, path)


def load_state(name, default=None):
    try:
        with open(session_path(name), 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def delete_state(name):
    try:
        os.remove(session_path(name))
    except OSError:
        pass


def load_cleaning_summary():
    summary = load_state('cleaning_summary.json') or {}
    for key in ('initial_shape', 'final_shape'):
        if isinstance(summary.get(key), list):
            summary[key] = tuple(summary[key])  # JSON turns tuples into lists
    return summary


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def dataset_file():
    """Path of the uploaded file for this session, or None."""
    if not session.get('session_id'):
        return None
    for extension in sorted(ALLOWED_EXTENSIONS):
        path = session_path(f'dataset.{extension}')
        if os.path.exists(path):
            return path
    return None


def cleaned_file():
    if not session.get('session_id'):
        return None
    path = session_path('cleaned.csv')
    return path if os.path.exists(path) else None



def read_dataset(path):
    """Read CSV or Excel files with common encoding fallbacks."""
    extension = os.path.splitext(path)[1].lower()

    if extension == '.csv':
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']

        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=None,
                    engine='python'
                )
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                last_error = error

        raise ValueError(
            f'Could not read CSV file. Please check its encoding or format: {last_error}'
        )

    if extension in ('.xlsx', '.xls'):
        return pd.read_excel(path)

    raise ValueError('Unsupported file type. Upload CSV or Excel.')



def get_df():
    """The originally uploaded dataset."""
    path = dataset_file()
    if not path:
        return None
    try:
        return read_dataset(path)
    except Exception:
        app.logger.exception('Could not read the uploaded dataset')
        return None


def get_working_df():
    """Cleaned data if available, otherwise the original upload. Returns (df, path)."""
    path = cleaned_file()
    if path:
        try:
            return pd.read_csv(path), path
        except Exception:
            app.logger.exception('Could not read the cleaned dataset')
    path = dataset_file()
    if path:
        try:
            return read_dataset(path), path
        except Exception:
            app.logger.exception('Could not read the uploaded dataset')
    return None, None


def sync_session_state():
    """If the files are gone (cleanup, restart) drop the cookie's stale dataset info."""
    if session.get('filename') and not dataset_file():
        for key in ('filename', 'run_id', 'pipeline_status'):
            session.pop(key, None)


def update_status(**flags):
    status = dict(session.get('pipeline_status', {}))
    status.update(flags)
    session['pipeline_status'] = status


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def existing_charts(directory, names):
    return {key: (name if os.path.exists(os.path.join(directory, name)) else None)
            for key, name in names.items()}


def ensure_eda_charts(df, source_path, exclude_columns):
    """
    Draw the EDA charts for this session unless they are already up to date
    for this exact file. Returns {chart key: filename or None}.
    """
    directory = chart_dir()
    stamp_path = os.path.join(directory, '.eda_stamp')
    try:
        stat = os.stat(source_path)
        stamp = json.dumps([os.path.basename(source_path), stat.st_mtime_ns, stat.st_size,
                            sorted(str(c) for c in exclude_columns)])
    except OSError:
        stamp = None

    current = None
    try:
        with open(stamp_path, 'r', encoding='utf-8') as handle:
            current = handle.read()
    except OSError:
        pass

    if stamp is not None and stamp == current:
        return existing_charts(directory, EDA_CHART_FILES)

    results = plot_all_eda_themes(df, exclude_columns=exclude_columns, output_dir=directory)
    if stamp is not None:
        try:
            with open(stamp_path, 'w', encoding='utf-8') as handle:
                handle.write(stamp)
        except OSError:
            pass
    return results


def clear_ml_charts(directory):
    for name in ML_CHART_FILES:
        base, extension = os.path.splitext(name)
        for variant in (name, f'{base}_dark{extension}'):
            try:
                os.remove(os.path.join(directory, variant))
            except OSError:
                pass


def generate_feature_importance_charts(importance, directory):
    """Light and dark versions; an empty list removes any old chart."""
    for theme in ('dark', 'light'):
        plot_feature_importance(importance or [], theme=theme, output_dir=directory)


@app.route('/static/generated/<path:filename>')
def chart_file(filename):
    """
    Serve generated charts from the visitor's own folder. This URL matches the
    one the templates already use, so no template changes are needed.
    """
    session_id = session.get('session_id')
    if not session_id or not SESSION_ID_PATTERN.match(str(session_id)):
        abort(404)
    if not filename.lower().endswith('.png'):
        abort(404)
    directory = os.path.join(SESSIONS_DIR, session_id, 'charts')
    return send_from_directory(directory, filename, max_age=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def index():
    sync_session_state()
    df, source_path = get_working_df()
    if df is None:
        return render_template('index.html', dataset_loaded=False)

    # Generate basic stats for the dashboard
    stats = {
        'rows': len(df),
        'cols': len(df.columns),
        'missing': int(df.isnull().sum().sum()),
        'types': len(df.dtypes.unique())
    }

    # Get first 5 rows for preview
    preview = df.head(5).to_html(classes='dataset-table', index=False)

    # Charts for the dashboard (only redrawn when the data changed)
    cleaning_summary = load_cleaning_summary()
    try:
        ensure_eda_charts(df, source_path, cleaning_summary.get('encoded_columns', []))
    except Exception:
        app.logger.exception('EDA charts failed')

    ml_results = load_state('ml_results.json')
    if ml_results and ml_results.get('importance'):
        if not os.path.exists(os.path.join(chart_dir(), 'feature_importance.png')):
            try:
                generate_feature_importance_charts(ml_results['importance'], chart_dir())
            except Exception:
                app.logger.exception('Feature importance chart failed')

    return render_template(
        'index.html',
        dataset_loaded=True,
        stats=stats,
        preview=preview,
        filename=session.get('filename'),
        ml_data=ml_results
    )


@same_origin_only
def logout():
    """Delete this session's files, clear session data and redirect to index"""
    reset_session_files()
    session.clear()
    return redirect(url_for('index'))


def _shutdown_server():
    os.kill(os.getpid(), signal.SIGTERM)


@same_origin_only
def shutdown():
    """Stop the server. Only allowed from the machine it runs on."""
    if request.remote_addr not in ('127.0.0.1', '::1'):
        abort(403)

    reset_session_files()
    session.clear()
    # Delay the kill slightly so the response can be sent first.
    threading.Timer(1.0, _shutdown_server).start()
    return 'Server shutting down...'


@same_origin_only
def new_analysis():
    reset_session_files()
    for key in ('filename', 'run_id', 'pipeline_status'):
        session.pop(key, None)
    return redirect(url_for('index'))


def progress_start():
    set_progress(1, 'Starting operation...', 'Processing your request', 'running')
    from flask import jsonify
    return jsonify({'started': True})


def progress_status():
    from flask import jsonify
    sid = get_session_id()
    with progress_lock:
        payload = dict(progress_by_session.get(sid, {'percent': 0, 'message': 'Waiting to start...',
                                                     'title': 'Processing your request...', 'state': 'idle'}))
    return jsonify(payload)


def upload_file():
    file = request.files.get('file')
    if file is None or file.filename == '':
        flash('No file selected. Please choose a dataset first.', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload CSV or Excel.', 'error')
        return redirect(url_for('index'))

    # Take the extension from the original name (secure_filename can strip it
    # from names that contain only non-ASCII characters).
    stem, extension = file.filename.rsplit('.', 1)
    extension = extension.lower()
    display_name = f"{secure_filename(stem) or 'dataset'}.{extension}"

    # Start from a clean slate for this session
    reset_session_files()
    path = session_path(f'dataset.{extension}', create=True)
    set_progress(55, 'File received. Reading and validating dataset...', 'Uploading dataset')
    file.save(path)
    set_progress(80, 'File saved. Reading and validating dataset...', 'Uploading dataset')

    try:
        df = read_dataset(path)
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError('The file contains no data.')
    except Exception as error:
        reset_session_files()
        for key in ('filename', 'run_id', 'pipeline_status'):
            session.pop(key, None)
        flash(f'Could not read that file: {error}', 'error')
        return redirect(url_for('index'))

    set_progress(100, 'Dataset uploaded and validated successfully.', 'Uploading dataset', 'complete')
    session['filename'] = display_name
    session['run_id'] = uuid.uuid4().hex[:8]
    session['pipeline_status'] = {'upload': True, 'clean': False, 'eda': False,
                                  'model': False, 'results': False}
    flash('Dataset uploaded successfully!', 'success')
    return redirect(url_for('index'))


def clean_page():
    df = get_df()
    if df is None:
        return redirect(url_for('index'))
    profile = profile_dataset(df)
    return render_template('clean.html', filename=session.get('filename'),
                           columns=[str(column) for column in df.columns], profile=profile,
                           cleaning_summary=load_cleaning_summary())


def process_cleaning():
    """Run the one-click adaptive cleaning workflow on the original upload."""
    df = get_df()
    if df is None:
        return redirect(url_for('index'))

    try:
        set_progress(3, 'Reading dataset and cleaning options...', 'Cleaning dataset')
        selected_target_columns = request.form.getlist('target_columns')
        selected_group_columns = request.form.getlist('group_columns')
        set_progress(12, 'Profiling columns and identifying data types...', 'Cleaning dataset')
        cleaned_df, summary = adaptive_clean(
            df,
            exclude_columns=selected_target_columns,
            group_columns=selected_group_columns or None,
            progress_callback=lambda pct, msg: set_progress(pct, msg, 'Cleaning dataset'),
        )
        set_progress(96, 'Cleaning completed. Comparing data quality...', 'Cleaning dataset')
        summary['quality_comparison'] = compare_quality(df, cleaned_df.drop(
            columns=['_auto_clean_anomaly', '_auto_clean_anomaly_reason'], errors='ignore'))
        try:
            summary['synthetic_imputation_benchmark'] = evaluate_injected_numeric_noise(adaptive_clean)
            set_progress(90, 'Synthetic imputation benchmark completed...', 'Cleaning dataset')
        except Exception as benchmark_error:
            summary.setdefault('warnings', []).append(
                f'Synthetic benchmark could not run: {benchmark_error}')
    except ValueError as error:
        flash(str(error), 'error')
        return redirect(url_for('clean_page'))
    except Exception as error:
        app.logger.exception('Adaptive cleaning failed')
        flash(f'Automatic cleaning failed: {error}', 'error')
        return redirect(url_for('clean_page'))

    set_progress(95, 'Saving cleaned dataset and audit report...', 'Cleaning dataset')
    delete_state('ml_results.json')
    clear_ml_charts(chart_dir())
    cleaned_df.to_csv(session_path('cleaned.csv', create=True), index=False)
    save_state('cleaning_summary.json', summary)
    update_status(clean=True, eda=False, model=False, results=False)
    set_progress(100, 'Cleaning completed successfully.', 'Cleaning dataset', 'complete')
    flash('Adaptive intelligent cleaning completed successfully!', 'success')
    for note in summary.get('warnings', []):
        flash(note, 'warning')
    return redirect(url_for('cleaning_report_page'))


def cleaning_report_page():
    if not cleaned_file():
        return redirect(url_for('clean_page'))
    summary = load_cleaning_summary()
    return render_template('cleaning_report.html', summary=summary,
                           filename=session.get('filename'))


def eda_page():
    df, source_path = get_working_df()
    if df is None:
        return redirect(url_for('index'))

    cleaning_summary = load_cleaning_summary()
    excluded_columns = list(cleaning_summary.get('encoded_columns', [])) + ['_auto_clean_anomaly', '_auto_clean_anomaly_reason']
    try:
        set_progress(10, 'Preparing dataset for visual analysis...', 'Generating EDA')
        eda_results = ensure_eda_charts(df, source_path, excluded_columns)
        set_progress(95, 'Charts generated successfully.', 'Generating EDA')
    except Exception:
        app.logger.exception('EDA charts failed')
        eda_results = {}
        flash('Some charts could not be drawn for this dataset.', 'warning')

    update_status(eda=True)
    set_progress(100, 'Exploratory data analysis is ready.', 'Generating EDA', 'complete')
    return render_template('eda.html', eda_results=eda_results, filename=session.get('filename'))


def ml_page():
    cleaned_path = cleaned_file()
    if not cleaned_path:
        flash('Clean your data first.', 'info')
        return redirect(url_for('clean_page'))

    df = pd.read_csv(cleaned_path)
    return render_template('ml.html', columns=[column for column in df.columns if column not in {'_auto_clean_anomaly', '_auto_clean_anomaly_reason'}], filename=session.get('filename'))


def label_for_code(value, names):
    """Turn an integer category code into its original name when we know it."""
    label = str(value)
    if names and isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        index = int(value)
        if 0 <= index < len(names):
            label = names[index]
    return label


def run_training(df, target, selected_models, problem_override, cleaning_summary, directory, progress_callback=None):
    """Train, compare, chart and describe. Returns (ml_results, notices)."""
    notices = []
    prob_type = detect_problem_type(df, target, override=problem_override)
    clear_ml_charts(directory)

    quality = data_quality_report(df, target)
    X_train, X_test, y_train, y_test, encoder, feature_names, split_info = prepare_features(
        df, target, return_metadata=True, problem_type=prob_type,
        categorical_columns=cleaning_summary.get('encoded_columns')
    )

    if prob_type == 'classification':
        results = train_classification_models(X_train, X_test, y_train, y_test, selected_models, progress_callback=progress_callback)
    else:
        results = train_regression_models(X_train, X_test, y_train, y_test, selected_models, progress_callback=progress_callback)

    model_errors = collect_model_errors(results)
    best_model_name, final_results = compare_models(results, prob_type)
    if best_model_name is None:
        details = '; '.join(f'{name}: {message}' for name, message in model_errors.items())
        raise ValueError('No model could be trained.' + (f' {details}' if details else ''))

    _, _, selection_note = get_best_model_selection(results, prob_type)
    best = final_results[best_model_name]
    best_metrics = best['metrics']

    importance = feature_importance(
        best['model'], feature_names, X_test=X_test, y_test=y_test,
        problem_type=prob_type, target_scale=best.get('target_scale')
    )

    # Class names: encoder classes are integer codes when the cleaner encoded the target
    class_labels, class_counts, full_class_counts = [], {}, {}
    if prob_type == 'classification' and encoder is not None:
        names = (cleaning_summary.get('encoded_labels') or {}).get(str(target))
        class_labels = [label_for_code(code, names) for code in encoder.classes_.tolist()]
        counts = np.bincount(np.asarray(y_train).astype(int), minlength=len(class_labels))
        class_counts = {class_labels[i]: int(counts[i]) for i in range(len(class_labels))}
        value_counts = df[target].value_counts(dropna=False)
        try:
            value_counts = value_counts.sort_index()
        except TypeError:
            pass
        full_class_counts = {label_for_code(key, names): int(count)
                             for key, count in value_counts.items()}

    # Charts: one failing chart must not lose the whole training run
    failed_charts = []

    def draw(label, function, *args, **kwargs):
        try:
            function(*args, output_dir=directory, **kwargs)
        except Exception:
            app.logger.exception('Chart "%s" failed', label)
            failed_charts.append(label)

    score_key = MODEL_SCORE_KEYS[prob_type]
    model_scores = {
        name: values['metrics'].get(score_key)
        for name, values in final_results.items()
        if values.get('model') is not None and values['metrics'].get(score_key) is not None
    }
    if model_scores:
        draw('model comparison', plot_model_comparison, model_scores, best_model=best_model_name)

    if prob_type == 'classification':
        draw('confusion matrix', plot_confusion_matrix,
             best_metrics.get('confusion_matrix', []), class_labels)
        probabilities = best_metrics.get('probabilities')
        if probabilities is not None and len(probabilities) > 0:
            draw('ROC curve', plot_roc_curve, best_metrics['y_true'], probabilities, class_labels)
            draw('precision-recall curve', plot_precision_recall_curve,
                 best_metrics['y_true'], probabilities, class_labels)
    elif best_metrics.get('y_true') and best_metrics.get('y_pred'):
        draw('actual vs predicted', plot_actual_vs_predicted,
             best_metrics['y_true'], best_metrics['y_pred'])

    try:
        generate_feature_importance_charts(importance, directory)
    except Exception:
        app.logger.exception('Feature importance chart failed')
        failed_charts.append('feature importance')

    ml_results = {
        'best_model': best_model_name,
        'selection_note': selection_note,
        'metrics': {
            name: {key: value for key, value in result['metrics'].items() if key not in ARRAY_METRICS}
            for name, result in final_results.items() if result.get('model') is not None
        },
        'importance': importance,
        'prob_type': prob_type,
        'feature_names': feature_names,
        'split_info': split_info,
        'quality': quality,
        'class_labels': class_labels,
        'class_counts': class_counts,
        'full_class_counts': full_class_counts,
        'target': target,
        'hyperparameters': {
            name: get_hyperparameters(result['model'])
            for name, result in final_results.items() if result.get('model') is not None
        },
        'model_errors': model_errors,
        'failed_charts': failed_charts,
    }

    notices.append(('info', f'Detected problem type: {prob_type}.'))
    for name, message in model_errors.items():
        notices.append(('warning', f'{name} was skipped: {message}'))
    if failed_charts:
        notices.append(('warning', 'Some charts could not be drawn: ' + ', '.join(failed_charts) + '.'))
    return ml_results, notices


def train_model():
    cleaned_path = cleaned_file()
    original_df = get_df()
    if not cleaned_path or original_df is None:
        flash('Upload and clean your data first.', 'info')
        return redirect(url_for('clean_page'))

    # The cleaned CSV is a downloadable artifact. It may contain full-dataset
    # imputations and integer category codes, so it must not be used as the
    # source for unbiased model evaluation. Train from the original upload,
    # normalizing placeholders/dtypes only; learned preprocessing happens after
    # the split inside ml_pipeline's sklearn pipelines.
    df = fix_dtypes(original_df.copy())
    target = request.form.get('target')
    goal = request.form.get('goal')  # "cleaning", "viz", "ml"

    selected_models = [name for key, name in MODEL_CHECKBOXES.items() if key in request.form]

    if goal != 'ml':
        delete_state('ml_results.json')
        clear_ml_charts(chart_dir())
        update_status(model=True, results=True)
        return redirect(url_for('results_page'))

    if not target or target not in df.columns:
        flash('Please choose a valid target column from the uploaded dataset.', 'error')
        return redirect(url_for('ml_page'))
    if not selected_models:
        flash('Select at least one algorithm.', 'error')
        return redirect(url_for('ml_page'))

    override = request.form.get('problem_type')
    if override not in ('classification', 'regression'):
        override = None

    try:
        set_progress(5, 'Preparing the original dataset and selected models...', 'Training machine-learning models')
        # Original values are preserved here, so no cleaner-generated category
        # codes/mappings should be applied during model preparation.
        model_cleaning_summary = {'encoded_columns': [], 'encoded_labels': {}}
        set_progress(15, 'Splitting data and preparing model features...', 'Training machine-learning models')
        ml_results, notices = run_training(
            df, target, selected_models, override, model_cleaning_summary, chart_dir(),
            progress_callback=lambda current, total, name: set_progress(15 + int(65 * current / total), f'Trained/evaluated {current} of {total} models: {name}', 'Training machine-learning models'),
        )
        set_progress(88, 'Models trained. Preparing evaluation charts and results...', 'Training machine-learning models')
    except ValueError as error:
        flash(f'ML Training Error: {error}', 'error')
        return redirect(url_for('ml_page'))
    except Exception as error:
        app.logger.exception('ML training failed')
        flash(f'ML Training Error: {error}', 'error')
        return redirect(url_for('ml_page'))

    save_state('ml_results.json', ml_results)
    update_status(model=True, results=True)
    set_progress(100, 'Model training and evaluation completed.', 'Training machine-learning models', 'complete')
    for category, message in notices:
        flash(message, category)
    return redirect(url_for('results_page'))


def results_page():
    if not cleaned_file():
        return redirect(url_for('index'))

    ml_data = load_state('ml_results.json')
    if ml_data and ml_data.get('importance'):
        if not os.path.exists(os.path.join(chart_dir(), 'feature_importance.png')):
            try:
                generate_feature_importance_charts(ml_data['importance'], chart_dir())
            except Exception:
                app.logger.exception('Feature importance chart failed')

    diagnostic_charts = [name for name in ML_CHART_FILES
                         if os.path.exists(os.path.join(chart_dir(), name))]
    return render_template(
        'results.html',
        ml_data=ml_data,
        filename=session.get('filename'),
        cleaning_summary=load_cleaning_summary(),
        chart_files=diagnostic_charts
    )


def download_stem():
    stem = os.path.splitext(session.get('filename') or 'dataset')[0]
    return secure_filename(stem) or 'dataset'


def download_cleaned():
    path = cleaned_file()
    if not path:
        flash('Clean a dataset first; there is no cleaned data to download yet.', 'info')
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True, download_name=f'{download_stem()}_cleaned.csv')


def download_anomalies():
    """Download all rows flagged by the anomaly detector for human review."""
    path = cleaned_file()
    if not path:
        flash('Clean a dataset first; there are no anomaly flags to download.', 'info')
        return redirect(url_for('index'))

    try:
        cleaned = pd.read_csv(path)
        if '_auto_clean_anomaly' not in cleaned.columns:
            flash('This cleaned dataset does not contain anomaly flags.', 'info')
            return redirect(url_for('cleaning_report_page'))

        flags = cleaned['_auto_clean_anomaly'].astype(str).str.strip().str.lower().isin(
            {'true', '1', 'yes'}
        )
        flagged = cleaned.loc[flags].copy()
        output_path = session_path('anomaly_review.csv', create=True)
        flagged.to_csv(output_path, index=False)
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f'{download_stem()}_flagged_records.csv'
        )
    except Exception as error:
        app.logger.exception('Could not create anomaly review download')
        flash(f'Could not prepare flagged records: {error}', 'error')
        return redirect(url_for('cleaning_report_page'))


def build_report(df, report_path, ml_results, cleaning_summary, metadata, directory):
    """Create the PDF using this session's charts."""
    arguments = (df, report_path, ml_results, cleaning_summary)
    if 'chart_dir' in inspect.signature(generate_full_report).parameters:
        generate_full_report(*arguments, metadata=metadata, chart_dir=directory)
        return
    # Older reporter: it reads a module-level chart folder, so point it at this
    # session's charts for the duration of the call (one report at a time).
    with report_lock:
        original = reporter_module.CHART_DIR
        reporter_module.CHART_DIR = directory
        try:
            generate_full_report(*arguments, metadata=metadata)
        finally:
            reporter_module.CHART_DIR = original


def download_report():
    cleaned_path = cleaned_file()
    if not cleaned_path:
        flash('Upload and clean a dataset before downloading a report.', 'info')
        return redirect(url_for('index'))

    set_progress(8, 'Loading cleaned dataset and report inputs...', 'Generating PDF report')
    df = pd.read_csv(cleaned_path)
    try:  # the report includes the EDA charts even if the EDA page was never opened
        set_progress(20, 'Preparing charts for the PDF report...', 'Generating PDF report')
        ensure_eda_charts(df, cleaned_path, load_cleaning_summary().get('encoded_columns', []))
    except Exception:
        app.logger.exception('EDA charts failed')
    report_path = session_path('report.pdf', create=True)
    metadata = {
        'filename': session.get('filename', 'Unknown'),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run_id': session.get('run_id', 'Unknown'),
    }
    try:
        set_progress(55, 'Building PDF pages, tables and visualisations...', 'Generating PDF report')
        build_report(df, report_path, load_state('ml_results.json'),
                     load_cleaning_summary() or None, metadata, chart_dir())
        set_progress(100, 'PDF report generated successfully.', 'Generating PDF report', 'complete')
    except Exception as error:
        app.logger.exception('Report generation failed')
        flash(f'Could not build the PDF report: {error}', 'error')
        return redirect(url_for('results_page'))

    return send_file(report_path, as_attachment=True, download_name=f'{download_stem()}_report.pdf')


# React-facing JSON API. Legacy server-rendered page routes have been removed.
from backend.api import api as api_blueprint
app.register_blueprint(api_blueprint)


if __name__ == '__main__':
    # Don't run with debug=True in production!
    app.run(host=os.environ.get('HOST', '127.0.0.1'),
            port=int(os.environ.get('PORT', '5000')),
            debug=False)