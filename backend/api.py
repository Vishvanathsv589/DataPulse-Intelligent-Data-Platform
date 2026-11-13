"""JSON API for the React frontend. All processing delegates to the original modules."""
from flask import Blueprint, jsonify, request, session, send_file
from werkzeug.utils import secure_filename
import os
import sys
import uuid
import pandas as pd

api = Blueprint('api', __name__, url_prefix='/api')


def _core():
    # Support both `python app.py` and `flask --app app run` without creating
    # a second Flask application (and a second, disconnected session store).
    running_module = sys.modules.get('__main__')
    if running_module is not None and hasattr(running_module, 'app'):
        return running_module
    import app as core
    return core


def _error(message, status=400):
    return jsonify({'error': str(message)}), status


def _dataset_payload(df, filename=None):
    return {
        'filename': filename,
        'stats': {
            'rows': int(len(df)), 'columns': int(len(df.columns)),
            'missing': int(df.isna().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'numeric_columns': int(len(df.select_dtypes(include='number').columns)),
        },
        'columns': [{'name': str(c), 'dtype': str(df[c].dtype),
                     'missing': int(df[c].isna().sum()),
                     'unique': int(df[c].nunique(dropna=True))} for c in df.columns],
        'preview': df.head(8).astype(object).where(pd.notna(df.head(8)), None).to_dict(orient='records'),
    }


@api.get('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Intelligent Data Platform API'})


@api.get('/session')
def session_info():
    core = _core()
    core.sync_session_state()
    df = core.get_working_df()[0]
    payload = {
        'loaded': df is not None,
        'filename': session.get('filename'),
        'status': session.get('pipeline_status', {}),
        'cleaning_summary': core.load_cleaning_summary(),
        'ml_results': core.load_state('ml_results.json'),
    }
    if df is not None:
        payload['dataset'] = _dataset_payload(df, session.get('filename'))
    return jsonify(core.to_jsonable(payload))


@api.post('/upload')
def upload():
    core = _core()
    file = request.files.get('file')
    if file is None or not file.filename:
        return _error('Choose a CSV or Excel file.')
    if not core.allowed_file(file.filename):
        return _error('Unsupported file type. Upload CSV, XLSX or XLS.')
    if request.content_length and request.content_length > 16 * 1024 * 1024:
        return _error('File is too large. Maximum upload size is 16 MB.', 413)
    stem, extension = file.filename.rsplit('.', 1)
    display_name = f"{secure_filename(stem) or 'dataset'}.{extension.lower()}"
    core.reset_session_files()
    path = core.session_path(f"dataset.{extension.lower()}", create=True)
    file.save(path)
    try:
        df = core.read_dataset(path)
        if df.empty or df.shape[1] == 0:
            raise ValueError('The file contains no data.')
    except Exception as exc:
        core.reset_session_files()
        session.clear()
        return _error(f'Could not read the dataset: {exc}')
    session['filename'] = display_name
    session['run_id'] = uuid.uuid4().hex[:8]
    session['pipeline_status'] = {'upload': True, 'clean': False, 'eda': False, 'model': False, 'results': False}
    core.set_progress(100, 'Dataset uploaded and validated successfully.', 'Uploading dataset', 'complete')
    return jsonify({'message': 'Dataset uploaded successfully.', 'dataset': _dataset_payload(df, display_name), 'status': session['pipeline_status']})


@api.get('/clean/profile')
def clean_profile():
    core = _core()
    df = core.get_df()
    if df is None:
        return _error('Upload a dataset first.', 404)
    return jsonify(core.to_jsonable({'profile': core.profile_dataset(df), 'columns': [str(c) for c in df.columns], 'summary': core.load_cleaning_summary()}))


@api.post('/clean')
def clean_dataset():
    core = _core()
    df = core.get_df()
    if df is None:
        return _error('Upload a dataset first.', 404)
    options = request.get_json(silent=True) or {}
    try:
        core.set_progress(3, 'Reading dataset and cleaning options...', 'Cleaning dataset')
        cleaned_df, summary = core.adaptive_clean(
            df,
            exclude_columns=options.get('target_columns') or [],
            group_columns=options.get('group_columns') or None,
            progress_callback=lambda pct, msg: core.set_progress(pct, msg, 'Cleaning dataset'),
        )
        core.set_progress(96, 'Comparing before-and-after data quality...', 'Cleaning dataset')
        summary['quality_comparison'] = core.compare_quality(
            df, cleaned_df.drop(columns=['_auto_clean_anomaly', '_auto_clean_anomaly_reason'], errors='ignore'))
        try:
            summary['synthetic_imputation_benchmark'] = core.evaluate_injected_numeric_noise(core.adaptive_clean)
        except Exception as exc:
            summary.setdefault('warnings', []).append(f'Synthetic benchmark could not run: {exc}')
        core.delete_state('ml_results.json')
        core.clear_ml_charts(core.chart_dir())
        cleaned_df.to_csv(core.session_path('cleaned.csv', create=True), index=False)
        core.save_state('cleaning_summary.json', summary)
        core.update_status(clean=True, eda=False, model=False, results=False)
        core.set_progress(100, 'Cleaning completed successfully.', 'Cleaning dataset', 'complete')
        return jsonify(core.to_jsonable({'message': 'Adaptive intelligent cleaning completed.', 'summary': summary,
                                         'stats': _dataset_payload(cleaned_df, session.get('filename'))['stats']}))
    except ValueError as exc:
        core.set_progress(100, str(exc), 'Cleaning dataset', 'error')
        return _error(exc)
    except Exception as exc:
        core.app.logger.exception('API cleaning failed')
        core.set_progress(100, 'Cleaning failed.', 'Cleaning dataset', 'error')
        return _error(f'Automatic cleaning failed: {exc}', 500)


@api.get('/eda')
def eda():
    core = _core()
    df, source_path = core.get_working_df()
    if df is None:
        return _error('Upload a dataset first.', 404)
    summary = core.load_cleaning_summary()
    excluded = list(summary.get('encoded_columns', [])) + ['_auto_clean_anomaly', '_auto_clean_anomaly_reason']
    try:
        core.set_progress(10, 'Preparing visual analysis...', 'Generating EDA')
        charts = core.ensure_eda_charts(df, source_path, excluded)
        core.update_status(eda=True)
        core.set_progress(100, 'Exploratory data analysis is ready.', 'Generating EDA', 'complete')
        chart_urls = {key: (f"/static/generated/{name}" if name else None) for key, name in charts.items()}
        return jsonify(core.to_jsonable({'stats': _dataset_payload(df, session.get('filename'))['stats'],
                                         'columns': [str(c) for c in df.columns], 'charts': chart_urls}))
    except Exception as exc:
        core.app.logger.exception('API EDA failed')
        return _error(f'Could not generate EDA: {exc}', 500)


@api.get('/ml/options')
def ml_options():
    core = _core()
    if not core.cleaned_file():
        return _error('Clean your dataset first.', 409)
    df = pd.read_csv(core.cleaned_file())
    return jsonify({'targets': [str(c) for c in df.columns if c not in {'_auto_clean_anomaly', '_auto_clean_anomaly_reason'}],
                    'models': list(core.MODEL_CHECKBOXES.values()), 'problem_types': ['auto', 'classification', 'regression']})


@api.post('/train')
def train():
    core = _core()
    original_df = core.get_df()
    if not core.cleaned_file() or original_df is None:
        return _error('Upload and clean your dataset first.', 409)
    data = request.get_json(silent=True) or {}
    target = data.get('target')
    if target not in original_df.columns:
        return _error('Choose a valid target column.')
    selected = data.get('models') or []
    selected = [name for name in selected if name in core.MODEL_CHECKBOXES.values()]
    if not selected:
        return _error('Select at least one algorithm.')
    override = data.get('problem_type')
    if override not in ('classification', 'regression'):
        override = None
    try:
        df = core.fix_dtypes(original_df.copy())
        core.set_progress(5, 'Preparing original dataset and selected models...', 'Training machine-learning models')
        results, notices = core.run_training(
            df, target, selected, override, {'encoded_columns': [], 'encoded_labels': {}}, core.chart_dir(),
            progress_callback=lambda current, total, name: core.set_progress(
                15 + int(65 * current / total), f'Trained/evaluated {current} of {total} models: {name}', 'Training machine-learning models'))
        core.save_state('ml_results.json', results)
        core.update_status(model=True, results=True)
        core.set_progress(100, 'Model training and evaluation completed.', 'Training machine-learning models', 'complete')
        return jsonify(core.to_jsonable({'results': results, 'notices': notices}))
    except ValueError as exc:
        core.set_progress(100, str(exc), 'Training machine-learning models', 'error')
        return _error(exc)
    except Exception as exc:
        core.app.logger.exception('API model training failed')
        core.set_progress(100, 'Model training failed.', 'Training machine-learning models', 'error')
        return _error(f'ML training failed: {exc}', 500)


@api.get('/results')
def results():
    core = _core()
    if not core.cleaned_file():
        return _error('Clean a dataset first.', 409)
    return jsonify(core.to_jsonable({'results': core.load_state('ml_results.json'),
                                     'cleaning_summary': core.load_cleaning_summary()}))


@api.get('/progress')
def progress():
    core = _core()
    sid = core.get_session_id()
    with core.progress_lock:
        payload = dict(core.progress_by_session.get(sid, {'percent': 0, 'message': 'Waiting to start.', 'title': 'Ready', 'state': 'idle'}))
    return jsonify(payload)


@api.post('/new-analysis')
def new_analysis():
    core = _core()
    core.reset_session_files()
    session.clear()
    return jsonify({'message': 'New analysis started.'})


@api.get('/downloads/cleaned')
def download_cleaned_api():
    core = _core()
    path = core.cleaned_file()
    if not path:
        return _error('Clean a dataset first.', 404)
    return send_file(path, as_attachment=True, download_name=f"{core.download_stem()}_cleaned.csv")


@api.get('/downloads/anomalies')
def download_anomalies_api():
    core = _core()
    path = core.cleaned_file()
    if not path:
        return _error('Clean a dataset first.', 404)
    cleaned = pd.read_csv(path)
    if '_auto_clean_anomaly' not in cleaned.columns:
        return _error('No anomaly flags are available.', 404)
    flags = cleaned['_auto_clean_anomaly'].astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})
    output = core.session_path('anomaly_review.csv', create=True)
    cleaned.loc[flags].to_csv(output, index=False)
    return send_file(output, as_attachment=True, download_name=f"{core.download_stem()}_flagged_records.csv")


@api.get('/downloads/report')
def download_report_api():
    # Reuse the original report generation implementation and its per-session chart handling.
    core = _core()
    cleaned_path = core.cleaned_file()
    if not cleaned_path:
        return _error('Upload and clean a dataset before downloading a report.', 409)
    try:
        df = pd.read_csv(cleaned_path)
        summary = core.load_cleaning_summary()
        core.ensure_eda_charts(df, cleaned_path, summary.get('encoded_columns', []))
        report_path = core.session_path('report.pdf', create=True)
        metadata = {'filename': session.get('filename', 'Unknown'), 'generated_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), 'run_id': session.get('run_id', 'Unknown')}
        core.build_report(df, report_path, core.load_state('ml_results.json'), summary or None, metadata, core.chart_dir())
        return send_file(report_path, as_attachment=True, download_name=f"{core.download_stem()}_report.pdf")
    except Exception as exc:
        core.app.logger.exception('API report generation failed')
        return _error(f'Could not build the PDF report: {exc}', 500)
