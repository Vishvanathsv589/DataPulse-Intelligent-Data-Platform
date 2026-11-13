
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, ArrowDownToLine, ArrowRight, BarChart3, Bell, Check, CheckCircle2,
  ChevronDown, CircleHelp, Database, FileBarChart2, FileCheck2, FileSpreadsheet,
  FlaskConical, Gauge, LayoutDashboard, LoaderCircle, Menu, MoreHorizontal,
  PanelLeftClose, Plus, RefreshCw, Search, Settings2, ShieldCheck, Sparkles, Moon, Sun,
  Table2, UploadCloud, WandSparkles, X, Zap, Maximize2, Copy, Download,
} from 'lucide-react'
import { api } from './services/api'

const nav = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, group: 'Workspace' },
  { id: 'clean', label: 'Data cleaning', icon: WandSparkles, group: 'Workspace' },
  { id: 'eda', label: 'Exploratory analysis', icon: BarChart3, group: 'Workspace' },
  { id: 'models', label: 'Machine learning', icon: FlaskConical, group: 'Workspace' },
  { id: 'reports', label: 'Reports & exports', icon: FileBarChart2, group: 'Manage' },
]

function DataPulseMark({ className = '' }) {
  return (
    <img
      className={className}
      src="/datapulse-logo.png"
      alt="DataPulse logo"
      draggable="false"
    />
  )
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString('en-IN')
}

function Metric({ label, value, detail, icon: Icon }) {
  return <div className="metric-card">
    <div className="metric-top">
      <span>{label}</span>
      <span className="metric-icon"><Icon size={17}/></span>
    </div>
    <div className="metric-value">{value ?? '—'}</div>
    <div className="metric-detail">{detail}</div>
  </div>
}

function EmptyState({ icon: Icon = Database, title, text, action }) {
  return <div className="empty-state">
    <div className="empty-icon"><Icon size={24}/></div>
    <h3>{title}</h3>
    <p>{text}</p>
    {action}
  </div>
}

function SectionHeading({ eyebrow, title, description, action }) {
  return <div className="section-heading">
    <div>
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
    {action}
  </div>
}

export default function App() {
  const [page, setPage] = useState('overview')
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState({ percent: 0, message: '', title: '' })
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [cleanProfile, setCleanProfile] = useState(null)
  const [cleanResult, setCleanResult] = useState(null)
  const [eda, setEda] = useState(null)
  const [mlOptions, setMlOptions] = useState(null)
  const [target, setTarget] = useState('')
  const [models, setModels] = useState([
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'SVM',
    'Neural Network',
    'XGBoost'
  ])
  const [targetColumns, setTargetColumns] = useState([])
  const [groupColumns, setGroupColumns] = useState([])
  const [problemType, setProblemType] = useState('auto')
  const [chartDialog, setChartDialog] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [search, setSearch] = useState('')
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('northstar-theme') || 'light' } catch { return 'light' }
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('northstar-theme', theme) } catch { /* storage may be unavailable */ }
  }, [theme])

  useEffect(() => {
    document.title = 'DataPulse | Intelligent Data Platform'
  }, [])

  useEffect(() => {
    if (!notice) return undefined
    const timer = window.setTimeout(() => setNotice(''), 30000)
    return () => window.clearTimeout(timer)
  }, [notice])

  useEffect(() => {
    if (!chartDialog) return undefined
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setChartDialog(null)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [chartDialog])

  const refreshSession = useCallback(async () => {
    try {
      setSession(await api.session())
    } catch (e) {
      setError(e.message)
    }
  }, [])

  useEffect(() => {
    refreshSession()
  }, [refreshSession])

  useEffect(() => {
    if (!busy) return undefined

    const timer = window.setInterval(async () => {
      try {
        setProgress(await api.progress())
      } catch {
        // Keep current progress
      }
    }, 700)

    return () => window.clearInterval(timer)
  }, [busy])

  const runTask = async (task, successMessage) => {
    setError('')
    setNotice('')
    setBusy(true)

    try {
      const result = await task()

      if (successMessage) setNotice(successMessage)

      await refreshSession()
      return result
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setBusy(false)
    }
  }

  const handleUpload = async (selected) => {
    if (!selected) return

    const valid = /\.(csv|xlsx|xls)$/i.test(selected.name)

    if (!valid) {
      setError('Please choose a CSV, XLSX or XLS file.')
      return
    }

    setFile(selected)

    const result = await runTask(
      () => api.upload(selected),
      'Your dataset is ready to explore.'
    )

    if (result) {
      setPage('overview')
      setFile(null)
      setCleanResult(null)
      setEda(null)
      setMlOptions(null)
    }
  }

  const loadCleaning = async () => {
    setError('')

    try {
      const result = await api.cleanProfile()
      setCleanProfile(result)
      setPage('clean')
    } catch (e) {
      setError(e.message)
      setPage('clean')
    }
  }

  const startCleaning = async () => {
    const result = await runTask(
      () => api.clean({
        target_columns: targetColumns,
        group_columns: groupColumns
      }),
      'Adaptive cleaning completed.'
    )

    if (result) {
      setCleanResult(result)
      setPage('clean')
    }
  }

  const loadEda = async () => {
    const result = await runTask(
      () => api.eda(),
      'Exploratory analysis is ready.'
    )

    if (result) {
      setEda(result)
      setPage('eda')
    }
  }

  const loadMl = async () => {
    setError('')

    try {
      const result = await api.mlOptions()
      setMlOptions(result)
      setTarget(result.targets[0] || '')
      setPage('models')
    } catch (e) {
      setError(e.message)
      setPage('models')
    }
  }

  const train = async () => {
    const result = await runTask(
      () => api.train({
        target,
        models,
        problem_type: problemType === 'auto' ? null : problemType
      }),
      'Model training completed.'
    )

    if (result) {
      setPage('models')
      setSession((old) => ({
        ...old,
        ml_results: result.results
      }))
    }
  }

  const reset = async () => {
    const result = await runTask(
      () => api.reset(),
      'A fresh workspace is ready.'
    )

    if (result) {
      setCleanProfile(null)
      setCleanResult(null)
      setEda(null)
      setMlOptions(null)
      setFile(null)
      setPage('overview')
    }
  }

  const dataset = session?.dataset
  const summary = cleanResult?.summary || session?.cleaning_summary
  const mlResults = session?.ml_results
  const status = session?.status || {}

  const activeNav = nav.find((item) => item.id === page)?.label || 'Overview'

  const filteredNav = useMemo(
    () => nav.filter((item) =>
      item.label.toLowerCase().includes(search.toLowerCase())
    ),
    [search]
  )

  const navigate = async (id) => {
    setPage(id)

    if (id === 'clean') await loadCleaning()
    if (id === 'eda') await loadEda()
    if (id === 'models') await loadMl()

    if (id === 'reports') {
      try {
        const result = await api.results()
        setSession((old) => ({
          ...old,
          ml_results: result.results,
          cleaning_summary: result.cleaning_summary
        }))
      } catch (e) {
        setError(e.message)
      }
    }
  }

  return <div className={`app-shell ${sidebarOpen ? 'sidebar-open' : 'sidebar-collapsed'}`} data-theme={theme}>

    {sidebarOpen && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}/>}
    <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar-hidden'}`}>

      <div className="brand">
        <div className="brand-mark"><DataPulseMark className="brand-logo"/></div>
        <div className="brand-copy">
          <strong>DataPulse<span>.</span></strong>
          <small>INTELLIGENT DATA PLATFORM</small>
        </div>
        <button
          className="icon-button sidebar-toggle"
          onClick={() => setSidebarOpen(false)}
          aria-label="Collapse sidebar"
        >
          <PanelLeftClose size={17}/>
        </button>
      </div>

      <div className="workspace-switch">
        <div className="workspace-avatar">ID</div>
        <div className="workspace-copy">
          <strong>Personal workspace</strong>
          <small>Data intelligence</small>
        </div>
        <ChevronDown size={15}/>
      </div>

      <div className="nav-search">
        <Search size={15}/>
        <input
          placeholder="Search workspace"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <kbd>⌘ K</kbd>
      </div>

      <nav className="main-nav">

        <div className="nav-label">WORKSPACE</div>

        {filteredNav.filter((item) => item.group === 'Workspace').map((item) =>
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? 'active' : ''}`}
            onClick={() => navigate(item.id)}
          >
            <item.icon size={17}/>
            <span>{item.label}</span>
            {item.id === 'clean' && status.clean &&
              <Check size={14} className="nav-check"/>
            }
          </button>
        )}

        <div className="nav-label manage-label">MANAGE</div>

        {filteredNav.filter((item) => item.group === 'Manage').map((item) =>
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? 'active' : ''}`}
            onClick={() => navigate(item.id)}
          >
            <item.icon size={17}/>
            <span>{item.label}</span>
          </button>
        )}

      </nav>

      <div className="sidebar-bottom">
        <div className="plan-card">
          <div className="plan-icon"><Sparkles size={16}/></div>
          <strong>Your data, made clearer.</strong>
          <p>Clean, understand and model datasets in one place.</p>
          <span className="plan-link">
            Explore the workflow <ArrowRight size={13}/>
          </span>
        </div>

        <button className="profile-row">
          <div className="profile-avatar">V</div>
          <div>
            <strong>Workspace owner</strong>
            <small>Local environment</small>
          </div>
          <MoreHorizontal size={17}/>
        </button>
      </div>

    </aside>

    <main className="main-area">

      <header className="topbar">
        <div className="topbar-left">
          <button
            className="icon-button mobile-menu-button"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label={sidebarOpen ? "Close navigation" : "Open navigation"}
          >
            <Menu size={18}/>
          </button>
          {!sidebarOpen && <button className="icon-button desktop-menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Menu size={18}/></button>}

          <div className="breadcrumb">
            Workspace <span>/</span> <strong>{activeNav}</strong>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="connection">
            <span className="connection-dot"/> API connected
          </div>
          <button
            className="icon-button theme-toggle"
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>}
          </button>
          <button className="icon-button" title="Help">
            <CircleHelp size={18}/>
          </button>
          <button className="icon-button" title="Notifications">
            <Bell size={18}/>
          </button>
          <div className="top-avatar">V</div>
        </div>
      </header>

      <div className="page-content">

        {error &&
          <div className="alert alert-error">
            <X size={17}/>
            <span>{error}</span>
            <button onClick={() => setError('')}>
              <X size={15}/>
            </button>
          </div>
        }

        {notice &&
          <div className="chart-toast" role="status" aria-live="polite"><CheckCircle2 size={16}/><span>{notice}</span></div>
        }

        {busy &&
          <div className="progress-panel">
            <div className="progress-heading">
              <span>
                <LoaderCircle className="spin" size={16}/>
                {progress.title || 'Working on your data'}
              </span>
              <strong>{progress.percent || 1}%</strong>
            </div>

            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: `${Math.max(1, progress.percent || 1)}%`
                }}
              />
            </div>

            <small>{progress.message || 'Preparing your request...'}</small>
          </div>
        }

        {page === 'overview' && <>
          <SectionHeading
            eyebrow="YOUR WORKSPACE"
            title="A clearer view of your data."
            description="Bring your dataset in. We'll help you prepare it, understand it and turn it into useful insight."
            action={
              <button className="button button-secondary" onClick={reset}>
                <Plus size={16}/> New analysis
              </button>
            }
          />

          {!session?.loaded ?
            <div className="welcome-grid">
              <div className="welcome-copy">
                <div className="welcome-kicker">
                  <span className="sparkle-dot"/>
                  YOUR DATA WORKFLOW, SIMPLIFIED
                </div>

                <h2>
                  Good data makes<br/>
                  <em>better decisions.</em>
                </h2>

                <p>
                  Start with a dataset and move from raw information to
                  confident insights — all in one thoughtful workspace.
                </p>

                <div className="welcome-points">
                  <span><CheckCircle2 size={15}/> Intelligent cleaning</span>
                  <span><CheckCircle2 size={15}/> Exploratory analysis</span>
                  <span><CheckCircle2 size={15}/> Predictive modelling</span>
                </div>
              </div>

              <div
                className={`upload-card ${dragging ? 'dragging' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragging(true)
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDragging(false)
                  handleUpload(e.dataTransfer.files[0])
                }}
              >
                <div className="upload-symbol">
                  <UploadCloud size={25}/>
                </div>

                <h3>Start with your dataset</h3>
                <p>Drag and drop a file here, or browse your computer.</p>

                <label className="button button-primary upload-button">
                  Choose a file
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={(e) => handleUpload(e.target.files[0])}
                  />
                </label>

                <div className="file-support">
                  <span>CSV</span>
                  <span>XLSX</span>
                  <span>XLS</span>
                  <small>Maximum file size 16 MB</small>
                </div>
              </div>
            </div>
            : <>
              <div className="dataset-banner">
                <div className="dataset-file-icon">
                  <FileSpreadsheet size={21}/>
                </div>

                <div className="dataset-file-info">
                  <strong>{session.filename}</strong>
                  <span>Dataset loaded · Ready for analysis</span>
                </div>

                <div className="dataset-actions">
                  <span className="status-pill">
                    <span/> Active dataset
                  </span>
                  <button
                    className="button button-secondary button-small"
                    onClick={reset}
                  >
                    <RefreshCw size={14}/> New dataset
                  </button>
                </div>
              </div>

              <div className="metrics-grid">
                <Metric
                  label="Total records"
                  value={formatNumber(dataset?.stats.rows)}
                  detail="Rows in current dataset"
                  icon={Database}
                />
                <Metric
                  label="Features"
                  value={formatNumber(dataset?.stats.columns)}
                  detail="Available data columns"
                  icon={Table2}
                />
                <Metric
                  label="Missing values"
                  value={formatNumber(dataset?.stats.missing)}
                  detail="Values to review"
                  icon={Gauge}
                />
                <Metric
                  label="Duplicate rows"
                  value={formatNumber(dataset?.stats.duplicate_rows)}
                  detail="Exact duplicate records"
                  icon={CopyIcon}
                />
              </div>

              <div className="content-grid">
                <section className="surface-card table-card">
                  <div className="card-heading">
                    <div>
                      <span className="eyebrow">DATA PREVIEW</span>
                      <h3>A look at your dataset</h3>
                    </div>
                    <span className="subtle-count">
                      First {dataset?.preview?.length || 0} rows
                    </span>
                  </div>
                  <DataPreview dataset={dataset}/>
                </section>

                <section className="surface-card workflow-card">
                  <div className="card-heading">
                    <div>
                      <span className="eyebrow">NEXT STEPS</span>
                      <h3>Your workflow</h3>
                    </div>
                    <span className="workflow-count">
                      {[status.clean, status.eda, status.model].filter(Boolean).length}/3
                    </span>
                  </div>

                  <WorkflowStep
                    number="01"
                    title="Clean your data"
                    text="Resolve missing values and review anomalies."
                    done={status.clean}
                    onClick={() => navigate('clean')}
                    icon={WandSparkles}
                  />

                  <WorkflowStep
                    number="02"
                    title="Explore patterns"
                    text="Visualise distributions and relationships."
                    done={status.eda}
                    onClick={() => navigate('eda')}
                    icon={BarChart3}
                  />

                  <WorkflowStep
                    number="03"
                    title="Train a model"
                    text="Compare algorithms and evaluate results."
                    done={status.model}
                    onClick={() => navigate('models')}
                    icon={FlaskConical}
                  />
                </section>
              </div>
            </>
          }

          <div className="feature-strip">
            <div>
              <div className="feature-icon"><ShieldCheck size={18}/></div>
              <div>
                <strong>Your original data stays intact</strong>
                <p>Cleaning creates a separate output. Your uploaded dataset is preserved.</p>
              </div>
            </div>

            <div>
              <div className="feature-icon"><Zap size={18}/></div>
              <div>
                <strong>One connected workflow</strong>
                <p>Move from preparation to analysis without juggling tools.</p>
              </div>
            </div>

            <div>
              <div className="feature-icon"><FileCheck2 size={18}/></div>
              <div>
                <strong>Explainable results</strong>
                <p>Review cleaning decisions, metrics and downloadable reports.</p>
              </div>
            </div>
          </div>
        </>}

        {page === 'clean' && <>
          <SectionHeading
            eyebrow="PREPARE YOUR DATA"
            title="Make your data analysis-ready."
            description="An adaptive cleaning workflow profiles your dataset and selects suitable methods based on its characteristics."
            action={session?.loaded && !status.clean ?
              <button
                className="button button-primary"
                disabled={busy}
                onClick={startCleaning}
              >
                <WandSparkles size={16}/> Run automatic cleaning
              </button>
              : null
            }
          />

          {!session?.loaded ?
            <EmptyState
              icon={WandSparkles}
              title="No dataset to clean yet"
              text="Upload a CSV or Excel dataset to inspect its quality and start cleaning."
              action={
                <button
                  className="button button-primary"
                  onClick={() => setPage('overview')}
                >
                  Go to overview
                </button>
              }
            />
            : <>
              <div className="clean-intro">
                <div className="clean-intro-icon">
                  <Sparkles size={20}/>
                </div>
                <div>
                  <strong>Adaptive Intelligent Cleaner</strong>
                  <p>
                    Methods are selected per column using data type,
                    missingness, distribution and dataset size. Potential
                    anomalies are flagged for review, not silently deleted.
                  </p>
                </div>
                <span className="engine-tag">ADAPTIVE ENGINE</span>
              </div>

              {cleanProfile?.profile && <>
                <div className="surface-card">
                  <div className="card-heading">
                    <div>
                      <span className="eyebrow">DATA PROFILE</span>
                      <h3>Column-level quality overview</h3>
                    </div>
                    <span className="subtle-count">
                      {cleanProfile.columns?.length} columns
                    </span>
                  </div>

                  <ProfileTable profile={cleanProfile.profile.columns}/>
                </div>

                <div className="surface-card cleaning-options">
                  <div className="card-heading">
                    <div>
                      <span className="eyebrow">OPTIONAL CONTEXT</span>
                      <h3>Guide the cleaning engine</h3>
                    </div>
                  </div>

                  <div className="form-grid cleaning-picker-grid">
                    <ColumnPicker
                      title="Target / outcome columns"
                      description="Exclude selected columns from imputation benchmarking and anomaly detection."
                      columns={cleanProfile.columns}
                      selected={targetColumns}
                      onChange={setTargetColumns}
                    />
                    <ColumnPicker
                      title="Anomaly comparison groups"
                      description="Choose grouping features, or leave empty for automatic detection."
                      columns={cleanProfile.columns}
                      selected={groupColumns}
                      onChange={setGroupColumns}
                    />
                  </div>
                </div>
              </>}

              {summary &&
                <CleaningSummary
                  summary={summary}
                  stats={cleanResult?.stats}
                />
              }

              {!status.clean &&
                <div className="surface-card clean-action-card">
                  <div>
                    <h3>Ready to improve your dataset?</h3>
                    <p>
                      Run the cleaning pipeline to compare data quality
                      before and after processing.
                    </p>
                  </div>
                  <button
                    className="button button-primary"
                    disabled={busy}
                    onClick={startCleaning}
                  >
                    <WandSparkles size={16}/> Run automatic cleaning
                  </button>
                </div>
              }

              {status.clean &&
                <div className="download-row">
                  <div>
                    <CheckCircle2 size={18}/>
                    <span>
                      <strong>Cleaning complete</strong>
                      <small>Your cleaned dataset is ready to download.</small>
                    </span>
                  </div>
                  <a
                    className="button button-secondary"
                    href="/api/downloads/cleaned"
                  >
                    <ArrowDownToLine size={15}/> Download cleaned CSV
                  </a>
                </div>
              }
              <NextWorkflow current="clean" onNavigate={navigate}/>
            </>
          }
        </>}

        {page === 'eda' && <>
          <SectionHeading
            eyebrow="UNDERSTAND YOUR DATA"
            title="Find the patterns that matter."
            description="Explore missingness, distributions, relationships and potential outliers across your dataset."
            action={session?.loaded &&
              <button
                className="button button-secondary"
                disabled={busy}
                onClick={loadEda}
              >
                <RefreshCw size={15}/> Refresh analysis
              </button>
            }
          />

          {!session?.loaded ?
            <EmptyState
              icon={BarChart3}
              title="Your analysis starts with data"
              text="Upload a dataset to generate exploratory visualisations."
              action={
                <button
                  className="button button-primary"
                  onClick={() => setPage('overview')}
                >
                  Upload dataset
                </button>
              }
            />
            : !eda ?
              <EmptyState
                icon={BarChart3}
                title="Generate your visual analysis"
                text="The EDA engine will create charts using your cleaned dataset when available."
                action={
                  <button
                    className="button button-primary"
                    disabled={busy}
                    onClick={loadEda}
                  >
                    Generate analysis
                  </button>
                }
              />
              : <>
                <div className="metrics-grid metrics-three">
                  <Metric
                    label="Records analysed"
                    value={formatNumber(eda.stats?.rows)}
                    detail="Rows in analysis dataset"
                    icon={Database}
                  />
                  <Metric
                    label="Features"
                    value={formatNumber(eda.stats?.columns)}
                    detail="Columns included"
                    icon={Table2}
                  />
                  <Metric
                    label="Missing values"
                    value={formatNumber(eda.stats?.missing)}
                    detail="Remaining missing cells"
                    icon={Gauge}
                  />
                </div>

                <div className="chart-grid">
                  {[
                    ['missing_values', 'Missing value overview', 'Where data is incomplete.'],
                    ['distribution', 'Feature distributions', 'Understand the shape of your data.'],
                    ['correlation', 'Correlation matrix', 'Explore relationships between numeric features.'],
                    ['outliers', 'Outlier overview', 'Identify values that may need attention.']
                  ].map(([key, title, desc]) =>
                    <ChartCard
                      key={key}
                      title={title}
                      description={desc}
                      src={theme === 'dark'
                        ? (eda.charts?.[`${key}_dark`] || eda.charts?.[key]?.replace(/\.png$/i, '_dark.png') || eda.charts?.[key])
                        : eda.charts?.[key]}
                      chartKey={key}
                      chartData={eda.chart_data?.[key] ?? eda.data?.[key] ?? null}
                      onAction={(action, payload) => {
                        if (action === 'copied') {
                          setNotice('Chart Copied')
                          return
                        }
                        setChartDialog({ action, title, src: theme === 'dark'
                          ? (eda.charts?.[`${key}_dark`] || eda.charts?.[key]?.replace(/\.png$/i, '_dark.png') || eda.charts?.[key])
                          : eda.charts?.[key], data: payload })
                      }}
                    />
                  )}
                </div>
                <NextWorkflow current="eda" onNavigate={navigate}/>
              </>
          }
        </>}

        {page === 'models' && <>
          <SectionHeading
            eyebrow="PREDICTIVE MODELLING"
            title="Put your data to work."
            description="Train suitable algorithms, compare their performance and inspect the strongest model."
          />

          {!status.clean ?
            <EmptyState
              icon={FlaskConical}
              title="Clean your data before modelling"
              text="The existing workflow requires a completed cleaning run before model training."
              action={
                <button
                  className="button button-primary"
                  onClick={() => navigate('clean')}
                >
                  Go to data cleaning
                </button>
              }
            />
            : <>
              {mlOptions &&
                <div className="surface-card model-config">
                  <div className="card-heading">
                    <div>
                      <span className="eyebrow">MODEL CONFIGURATION</span>
                      <h3>Choose what to predict</h3>
                    </div>
                    <span className="engine-tag">SCIKIT-LEARN</span>
                  </div>

                  <div className="form-grid">
                    <label className="field-label">
                      Problem type
                      <select
                        value={problemType}
                        onChange={(e) => setProblemType(e.target.value)}
                      >
                        <option value="auto">Automatic detection</option>
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                      </select>
                    </label>

                    <label className="field-label">
                      Target column
                      <select
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                      >
                        {mlOptions.targets.map((item) =>
                          <option key={item}>{item}</option>
                        )}
                      </select>
                    </label>
                  </div>

                  <div className="field-label model-label">
                    Select algorithms
                  </div>

                  <div className="model-options">
                    {mlOptions.models.map((model) =>
                      <label
                        className={`model-option ${models.includes(model) ? 'selected' : ''}`}
                        key={model}
                      >
                        <input
                          type="checkbox"
                          checked={models.includes(model)}
                          onChange={(e) =>
                            setModels((old) =>
                              e.target.checked
                                ? [...old, model]
                                : old.filter((m) => m !== model)
                            )
                          }
                        />
                        <span className="custom-check">
                          {models.includes(model) && <Check size={12}/>}
                        </span>
                        <span>{model}</span>
                      </label>
                    )}
                  </div>

                  <div className="model-footer">
                    <span>
                      Preprocessing is fitted within the training split
                      to help prevent data leakage.
                    </span>
                    <button
                      className="button button-primary"
                      onClick={train}
                      disabled={busy || !target || !models.length}
                    >
                      <FlaskConical size={16}/> Train selected models
                    </button>
                  </div>
                </div>
              }

              {mlResults && <>
                <ModelResults results={mlResults}/>
                <div className="model-report-action">
                  <p>Ready to share your model evaluation?</p>
                  <button className="button button-primary" onClick={() => navigate('reports')}>
                    <FileBarChart2 size={15}/> Report &amp; Export
                  </button>
                </div>
              </>}
              <NextWorkflow current="models" onNavigate={navigate}/>

              {!mlOptions &&
                <button className="button button-primary" onClick={loadMl}>
                  Load model options
                </button>
              }
            </>
          }
        </>}

        {page === 'reports' && <>
          <SectionHeading
            eyebrow="YOUR OUTPUTS"
            title="Results, ready to share."
            description="Download the cleaned data, anomaly review and a consolidated PDF report of your analysis."
          />

          {!status.clean ?
            <EmptyState
              icon={FileBarChart2}
              title="No reports available yet"
              text="Complete a cleaning run to create downloadable outputs."
              action={
                <button
                  className="button button-primary"
                  onClick={() => navigate('clean')}
                >
                  Start cleaning
                </button>
              }
            />
            :
            <div className="report-list">
              <ReportItem
                icon={FileSpreadsheet}
                title="Cleaned dataset"
                description="Processed CSV with audit columns and anomaly flags."
                href="/api/downloads/cleaned"
                type="CSV"
              />
              <ReportItem
                icon={Activity}
                title="Flagged anomaly records"
                description="A separate file containing records marked for human review."
                href="/api/downloads/anomalies"
                type="CSV"
              />
              <ReportItem
                icon={FileBarChart2}
                title="Full analysis report"
                description="PDF report with data quality, cleaning decisions, charts and model results when available."
                href="/api/downloads/report"
                type="PDF"
              />
            </div>
          }

          {mlResults &&
            <div className="surface-card report-model-note">
              <div className="feature-icon"><FlaskConical size={18}/></div>
              <div>
                <strong>Model evaluation included</strong>
                <p>
                  Your latest model metrics and feature importance are included
                  in the full report.
                </p>
              </div>
            </div>
          }
        </>}

        {chartDialog && <ChartActionDialog
          dialog={chartDialog}
          onClose={() => setChartDialog(null)}
        />}

        <footer className="page-footer">
          <span>
            DataPulse <span className="footer-dot">·</span>
            Built for thoughtful analysis
          </span>
          <span>
            Local workspace <span className="connection-dot"/>
          </span>
        </footer>

      </div>
    </main>
  </div>
}

function CopyIcon(props) {
  return <FileSpreadsheet {...props}/>
}

function DataPreview({ dataset }) {
  if (!dataset?.preview?.length) {
    return <div className="table-empty">No preview available.</div>
  }

  const columns = Object.keys(dataset.preview[0])

  return <div className="table-scroll">
    <table className="data-table">
      <thead>
        <tr>
          {columns.map((col) => <th key={col}>{col}</th>)}
        </tr>
      </thead>
      <tbody>
        {dataset.preview.map((row, i) =>
          <tr key={i}>
            {columns.map((col) =>
              <td key={col}>
                {row[col] === null
                  ? <span className="null-value">NULL</span>
                  : String(row[col])
                }
              </td>
            )}
          </tr>
        )}
      </tbody>
    </table>
  </div>
}

function WorkflowStep({ number, title, text, done, onClick, icon: Icon }) {
  return <button className="workflow-step" onClick={onClick}>
    <div className={`step-number ${done ? 'done' : ''}`}>
      {done ? <Check size={14}/> : number}
    </div>
    <div className="step-copy">
      <strong>{title}</strong>
      <small>{text}</small>
    </div>
    <span className="step-icon"><Icon size={17}/></span>
    <ArrowRight className="step-arrow" size={16}/>
  </button>
}

function ProfileTable({ profile }) {
  const rows = Array.isArray(profile)
    ? profile
    : Object.entries(profile || {}).map(([name, value]) => ({
        column: name,
        ...(typeof value === 'object' ? value : { profile: value })
      }))

  if (!rows.length) {
    return <p className="muted-copy">Profile details are not available.</p>
  }

  return <div className="table-scroll">
    <table className="data-table profile-table">
      <thead>
        <tr>
          <th className="profile-column-heading">Column</th>
          <th>Type</th>
          <th>Missing</th>
          <th>Unique</th>
          <th>Cleaning insight</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => {
          const name = row.column || row.name || row.column_name || `Column ${i + 1}`

          return <tr key={name}>
            <td className="profile-column-cell"><strong>{name}</strong></td>
            <td>{row.dtype || row.data_type || row.type || '—'}</td>
            <td>{row.missing_count ?? row.missing ?? row.missing_values ?? '—'}</td>
            <td>{row.unique_count ?? row.unique ?? '—'}</td>
            <td>
              {row.recommendation || row.strategy || row.kind || row.reason || 'Profiled'}
            </td>
          </tr>
        })}
      </tbody>
    </table>
  </div>
}

function CleaningSummary({ summary, stats }) {
  const quality = summary.quality_comparison || {}
  const before = quality.before || quality.original || {}
  const after = quality.after || quality.cleaned || {}

  const metrics = [
    ['Quality before', summary.quality_score_before],
    ['Quality after', summary.quality_score_after],
    ['Duplicates removed', summary.duplicate_rows_removed],
    ['Anomalies flagged', summary.anomaly_rows_flagged],
  ]

  return <div className="surface-card cleaning-result">
    <div className="card-heading">
      <div>
        <span className="eyebrow">CLEANING OUTCOME</span>
        <h3>Your dataset, after cleaning</h3>
      </div>
      <span className="status-pill"><span/> Completed</span>
    </div>

    <div className="result-metrics">
      {metrics.map(([label, value]) =>
        <div key={label}>
          <small>{label}</small>
          <strong>
            {value == null
              ? '—'
              : label.includes('Quality')
                ? `${Number(value).toFixed(2)}%`
                : formatNumber(value)
            }
          </strong>
        </div>
      )}
    </div>

    {summary.warnings?.length > 0 &&
      <div className="warning-list">
        <strong>Review notes</strong>
        {summary.warnings.map((warning, i) =>
          <p key={i}>• {warning}</p>
        )}
      </div>
    }

    <details className="details-block">
      <summary>View full cleaning summary</summary>
      <pre>{JSON.stringify(summary, null, 2)}</pre>
    </details>
  </div>
}

function ChartCard({ title, description, src, chartKey, chartData, onAction }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    if (!menuOpen) return undefined
    const closeOnOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) setMenuOpen(false)
    }
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setMenuOpen(false)
    }
    document.addEventListener('pointerdown', closeOnOutside)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutside)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [menuOpen])

  const handleCopy = async () => {
    setMenuOpen(false)
    if (!src) return
    try {
      const response = await fetch(src)
      if (!response.ok) throw new Error('Chart image could not be loaded')
      const blob = await response.blob()
      if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') throw new Error('Clipboard image copy is unavailable')
      await navigator.clipboard.write([new ClipboardItem({ [blob.type || 'image/png']: blob })])
      onAction?.('copied', null)
    } catch {
      onAction?.('copy-error', null)
    }
  }

  const runAction = (action, payload) => {
    setMenuOpen(false)
    onAction?.(action, payload)
  }

  return <div className="surface-card chart-card">
    <div className="chart-title">
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      <div className="chart-actions-wrap" ref={menuRef}>
        <button className="icon-button" type="button" aria-label={`Options for ${title}`} aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}><MoreHorizontal size={18}/></button>
        {menuOpen && <div className="chart-menu-popover" role="menu">
          <button type="button" role="menuitem" onClick={() => runAction('expand', null)}><Maximize2 size={14}/> Expand chart</button>
          <button type="button" role="menuitem" onClick={() => runAction('data', chartData)}><Table2 size={14}/> View chart data</button>
          <a role="menuitem" href={src || '#'} download={`${chartKey || 'chart'}.png`} onClick={(event) => { setMenuOpen(false); if (!src) event.preventDefault() }}><Download size={14}/> Download PNG</a>
          <button type="button" role="menuitem" onClick={handleCopy}><Copy size={14}/> Copy chart</button>
        </div>}
      </div>
    </div>
    {src
      ? <button className="chart-image-link" onClick={() => runAction('expand', null)} aria-label={`Expand ${title}`}>
          <img src={src} alt={title}/>
        </button>
      : <div className="chart-placeholder"><BarChart3 size={25}/><span>Chart unavailable for this dataset</span></div>
    }
  </div>
}

function ChartDataTable({ data }) {
  if (Array.isArray(data?.columns) && data.columns.length && data.columns[0]?.column) {
    return <>
      <p className="chart-data-summary">Total missing values: <strong>{formatNumber(data.total_missing)}</strong></p>
      <div className="table-scroll chart-data-table"><table className="data-table"><thead><tr><th>Column</th><th>Missing</th><th>Present</th></tr></thead><tbody>
        {data.columns.map((row) => <tr key={row.column}><td>{row.column}</td><td>{formatNumber(row.missing)}</td><td>{formatNumber(row.present)}</td></tr>)}
      </tbody></table></div>
    </>
  }
  if (Array.isArray(data?.bins)) {
    return <>
      <div className="chart-data-summary-grid">
        {[['Feature', data.column], ['Records', formatNumber(data.count)], ['Mean', Number(data.mean).toFixed(3)], ['Median', Number(data.median).toFixed(3)], ['Minimum', Number(data.min).toFixed(3)], ['Maximum', Number(data.max).toFixed(3)]].map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}
      </div>
      <div className="table-scroll chart-data-table"><table className="data-table"><thead><tr><th>Value range</th><th>Frequency</th></tr></thead><tbody>
        {data.bins.map((row, index) => <tr key={`${row.range}-${index}`}><td>{row.range}</td><td>{formatNumber(row.count)}</td></tr>)}
      </tbody></table></div>
    </>
  }
  if (Array.isArray(data?.matrix) && Array.isArray(data?.columns)) {
    return <div className="table-scroll chart-data-table"><table className="data-table"><thead><tr><th>Feature</th>{data.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>
      {data.matrix.map((row) => <tr key={row.feature}><td>{row.feature}</td>{data.columns.map((column) => <td key={column}>{row[column] == null ? '—' : Number(row[column]).toFixed(3)}</td>)}</tr>)}
    </tbody></table></div>
  }
  if (Array.isArray(data?.features)) {
    return <>
      <p className="chart-data-summary">Outlier rule: <strong>{data.method}</strong></p>
      <div className="table-scroll chart-data-table"><table className="data-table"><thead><tr><th>Feature</th><th>Records</th><th>Outliers</th><th>Lower fence</th><th>Upper fence</th></tr></thead><tbody>
        {data.features.map((row) => <tr key={row.column}><td>{row.column}</td><td>{formatNumber(row.count)}</td><td>{formatNumber(row.outlier_count)}</td><td>{Number(row.lower_fence).toFixed(3)}</td><td>{Number(row.upper_fence).toFixed(3)}</td></tr>)}
      </tbody></table></div>
    </>
  }
  return <pre className="chart-data-view">{JSON.stringify(data, null, 2)}</pre>
}

function ChartActionDialog({ dialog, onClose }) {
  const isData = dialog.action === 'data'
  const isCopyError = dialog.action === 'copy-error'
  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="chart-dialog" role="dialog" aria-modal="true" aria-label={dialog.title}>
      <header><div><span className="eyebrow">{isData ? 'CHART DATA' : 'CHART DETAILS'}</span><h3>{dialog.title}</h3></div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={18}/></button></header>
      {isData ? (dialog.data != null
        ? <ChartDataTable data={dialog.data}/>
        : <div className="chart-data-empty"><Table2 size={24}/><strong>Chart values aren't available yet.</strong><p>This chart service currently returns an image rather than the values used to plot it. The backend needs to expose the chart's underlying data before this view can show a proper table.</p></div>)
        : isCopyError ? <p className="dialog-feedback">Your browser couldn't copy the chart image. Use Download PNG instead.</p>
        : <img className="chart-dialog-image" src={dialog.src} alt={dialog.title}/>
      }
      <footer><button className="button button-secondary" onClick={onClose}>Close</button>{dialog.src && <a className="button button-primary" href={dialog.src} download="datapulse-chart.png"><Download size={14}/> Download PNG</a>}</footer>
    </section>
  </div>
}

function ColumnPicker({ title, description, columns = [], selected = [], onChange }) {
  const [query, setQuery] = useState('')
  const filtered = columns.filter((column) => String(column).toLowerCase().includes(query.toLowerCase()))
  const toggle = (column) => onChange(selected.includes(column) ? selected.filter((item) => item !== column) : [...selected, column])
  return <div className="column-picker">
    <div className="picker-heading"><strong>{title}</strong><span>{selected.length} selected</span></div>
    <div className="picker-search"><Search size={14}/><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find a column..."/></div>
    <div className="picker-actions"><button type="button" onClick={() => onChange([...columns])}>Select all</button><button type="button" onClick={() => onChange([])}>Clear</button></div>
    <div className="picker-list">
      {filtered.map((column) => <label key={column} className={`picker-option ${selected.includes(column) ? 'selected' : ''}`}>
        <input type="checkbox" checked={selected.includes(column)} onChange={() => toggle(column)}/><span className="picker-check">{selected.includes(column) ? <Check size={12}/> : null}</span><span>{column}</span>
      </label>)}
      {!filtered.length && <p className="picker-empty">No matching columns.</p>}
    </div>
    <p className="picker-description">{description}</p>
  </div>
}

function NextWorkflow({ current, onNavigate }) {
  const steps = [
    { id: 'clean', label: 'Clean your data', detail: 'Resolve missing values and review anomalies.', icon: WandSparkles },
    { id: 'eda', label: 'Explore patterns', detail: 'Visualise distributions and relationships.', icon: BarChart3 },
    { id: 'models', label: 'Train a model', detail: 'Compare algorithms and evaluate results.', icon: FlaskConical },
    { id: 'reports', label: 'Export your results', detail: 'Download datasets and analysis reports.', icon: FileBarChart2 },
  ]
  const index = steps.findIndex((step) => step.id === current)
  const next = steps.slice(index + 1, index + 2)[0] || steps[3]
  if (!next || next.id === current) return null
  const Icon = next.icon
  return <section className="next-workflow surface-card">
    <div><span className="eyebrow">NEXT WORKFLOW</span><h3>{next.label}</h3><p>{next.detail}</p></div>
    <button className="button button-primary" onClick={() => onNavigate(next.id)}><Icon size={15}/> Continue <ArrowRight size={14}/></button>
  </section>
}

function ModelResults({ results }) {
  const metricRows = Object.entries(results.metrics || {})

  return <div className="surface-card model-results">
    <div className="card-heading">
      <div>
        <span className="eyebrow">EVALUATION RESULTS</span>
        <h3>Model comparison</h3>
      </div>
      <span className="best-model">
        <Sparkles size={14}/> Best: {results.best_model || '—'}
      </span>
    </div>

    {results.selection_note &&
      <p className="selection-note">{results.selection_note}</p>
    }

    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Performance metrics</th>
          </tr>
        </thead>
        <tbody>
          {metricRows.map(([name, metrics]) =>
            <tr key={name}>
              <td>
                <strong>{name}</strong>
                {name === results.best_model &&
                  <span className="best-tag">BEST</span>
                }
              </td>
              <td>
                <div className="metric-chips">
                  {Object.entries(metrics).map(([key, value]) =>
                    <span key={key}>
                      {key.replaceAll('_', ' ')}
                      {' '}
                      <strong>
                        {typeof value === 'number'
                          ? value.toFixed(4)
                          : String(value)
                        }
                      </strong>
                    </span>
                  )}
                </div>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>

    {results.importance?.length > 0 &&
      <div className="importance-list">
        <h4>Feature importance</h4>

        {results.importance.slice(0, 8).map((item, i) => {
          const name = item.feature || item.name || item[0] || `Feature ${i + 1}`
          const raw = item.importance ?? item.score ?? item[1] ?? 0
          const width = Math.max(2, Math.min(100, Math.abs(Number(raw)) * 100))

          return <div className="importance-row" key={name}>
            <span>{name}</span>
            <div>
              <i style={{ width: `${width}%` }}/>
            </div>
            <strong>{Number(raw).toFixed(3)}</strong>
          </div>
        })}
      </div>
    }
  </div>
}

function ReportItem({ icon: Icon, title, description, href, type }) {
  return <div className="report-item">
    <div className="report-icon"><Icon size={20}/></div>
    <div className="report-copy">
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
    <span className="file-type">{type}</span>
    <a className="button button-secondary button-small" href={href}>
      <ArrowDownToLine size={15}/> Download
    </a>
  </div>
}
