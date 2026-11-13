# Intelligent Data Platform — React + Python

A React frontend for the existing Python data platform. The original data-processing modules are retained and are called by a Flask JSON API; the existing Flask-rendered pages and routes remain available for backwards compatibility.

## Architecture

- **Frontend:** React 18 + Vite, with a responsive interface inspired by the supplied Enode reference.
- **Python backend:** Existing Flask application (`app.py`) plus JSON API routes in `backend/api.py`.
- **Data processing:** Existing modules in `modules/` — no algorithm rewrites.
- **Charts:** Existing Matplotlib chart generation, served from per-session storage.
- **Reports:** Existing PDF report generator and download workflow.

The API is intentionally added to the existing Flask server instead of replacing Flask with FastAPI. This preserves the current secure session handling, per-user storage, upload validation, progress tracking, chart isolation and report generation while React takes over the user interface.

## Requirements

- Python 3.10+
- Node.js 20.19+ or 22.12+
- npm

## Setup on Windows

### 1. Python backend

Open a terminal in the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The backend runs at `http://127.0.0.1:5000`.

### 2. React frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

After setup, you can also launch the backend and frontend with `start_backend.bat` and `start_frontend.bat` in two separate terminals/windows.

Vite proxies `/api` and `/static/generated` requests to Flask. This keeps the frontend and API on one browser origin during development so the existing Flask session cookie and same-origin write protection continue to work.

## React API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend health check |
| `GET /api/session` | Current dataset and workflow state |
| `POST /api/upload` | CSV / XLSX / XLS upload |
| `GET /api/clean/profile` | Adaptive column profile |
| `POST /api/clean` | Run the original adaptive cleaner |
| `GET /api/eda` | Generate/retrieve existing EDA charts |
| `GET /api/ml/options` | Available targets and algorithms |
| `POST /api/train` | Run the original ML pipeline |
| `GET /api/results` | Latest ML and cleaning results |
| `GET /api/progress` | Current operation progress |
| `POST /api/new-analysis` | Reset the current session |
| `GET /api/downloads/cleaned` | Download cleaned CSV |
| `GET /api/downloads/anomalies` | Download flagged rows for review |
| `GET /api/downloads/report` | Generate and download PDF report |

## Preserved processing modules

- `modules/cleaner.py`
- `modules/intelligent_cleaner.py`
- `modules/cleaning_evaluator.py`
- `modules/eda.py`
- `modules/ml_pipeline.py`
- `modules/reporter.py`

The original `app.py` routes and templates are retained as a compatibility path. The React frontend uses the new `/api/*` endpoints. The original upload is preserved; the cleaning output remains a separate artifact. Anomaly flags are retained for human review, not silently deleted. ML preprocessing continues to be fitted inside the train/test pipeline to reduce leakage.

## Tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Upload size remains limited to 16 MB.
- Session data is stored under `user_data/sessions/` and is isolated per browser session.
- Set `SECRET_KEY` in the environment for deployments; `.secret_key` remains ignored by Git.
- Do not expose the development server directly to the public internet. Use a production WSGI server, HTTPS, and an appropriately configured reverse proxy for deployment.
