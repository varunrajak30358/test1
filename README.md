# ⚡ ChargeAgent — EV Demand & Charging Infrastructure Forecasting Dashboard

**Enterprise-grade Streamlit application for regional EV demand forecasting and
charging infrastructure planning across Indian states.**

Built for IEEE publication research. Reproduces the exact prediction pipeline
from the training notebook using only pre-trained `.pkl` artifacts — no retraining,
and the notebook is not required to run this application.

---

## 📁 Project Files

```
mandeep interface/
├── app.py                  # Main Streamlit application (production-ready)
├── utils.py                # Core inference engine & Plotly chart utilities
├── style.css               # Custom light-theme enterprise dashboard CSS
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Forces light theme, removes dark mode toggle
│
│   — Pre-trained model artifacts —
├── best_model.pkl          # Trained model (loaded via joblib)
├── selected_features.pkl   # 27 consensus-selected features
├── feature_columns.pkl     # All 28 candidate feature names
├── scaler.pkl              # StandardScaler (fit on train only)
├── label_encoders.pkl      # LabelEncoder for State & Vehicle_Category
├── metadata.pkl            # Test metrics, states, categories, historical averages
├── forecast_config.pkl     # Train/val/test/live date boundaries
└── series_history.pkl      # Last 12 known monthly values per series (170 series)
```

> The original training notebook (`mandeep-ieee.ipynb`) is **not included and not needed**.  
> All model knowledge is already captured in the `.pkl` files above.

---

## 🚀 Setup & Run on Any System

Follow these steps exactly — works on Windows, macOS, and Linux.

### ✅ Prerequisites

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| Python | 3.9 or higher | `python --version` |
| pip | Latest | `pip --version` |

> **Don't have Python?** Download it from [https://www.python.org/downloads/](https://www.python.org/downloads/)  
> ✔ During installation on Windows — check **"Add Python to PATH"**

---

### 📁 Step 1 — Copy the Project Folder

Copy the **mandeep interface** folder to your system. Only the files listed above are needed — the original training notebook is **not required**.

> ⚠️ All 8 `.pkl` files are required. The app will not start if any are missing.

---

### 💻 Step 2 — Open a Terminal in the Project Folder

**Windows:**
1. Open the project folder in File Explorer
2. Click the address bar, type `cmd`, press Enter

**macOS / Linux:**
```bash
cd "/path/to/mandeep interface"
```

---

### 📦 Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Installs: `streamlit`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `plotly`

> ⏳ First-time install may take 2–3 minutes.

**Permissions error on Windows:**
```bash
pip install -r requirements.txt --user
```

**On macOS / Linux:**
```bash
pip3 install -r requirements.txt
```

---

### ▶️ Step 4 — Run the Application

```bash
streamlit run app.py
```

Opens automatically in your default browser at `http://localhost:8501`.

> If the browser doesn't open automatically, copy the URL and paste it manually.

---

### 🛑 How to Stop

Press `Ctrl + C` in the terminal.

---

## 🖥️ Platform-Specific Commands

### Windows
```cmd
cd "C:\path\to\mandeep interface"
pip install -r requirements.txt
streamlit run app.py
```

### macOS
```bash
cd "/path/to/mandeep interface"
pip3 install -r requirements.txt
streamlit run app.py
```

### Linux
```bash
cd "/path/to/mandeep interface"
pip3 install -r requirements.txt
streamlit run app.py
```

---

## 🐍 Using a Virtual Environment (Recommended)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
streamlit run app.py
```

Deactivate when done:
```bash
deactivate
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `streamlit: command not found` | Run `python -m streamlit run app.py` instead |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| `FileNotFoundError: *.pkl` | Make sure all `.pkl` files are in the same folder as `app.py` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Blank page in browser | Wait 5 seconds and refresh — model is still loading |
| `python` not recognized (Windows) | Use `py` instead: `py -m streamlit run app.py` |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🔮 Forecast Dashboard | Metric cards, AI recommendation, interactive Plotly charts, CSV download |
| 📦 Batch Prediction | Upload CSV → forecast all rows → download results |
| 📊 Model Information | Algorithm details, feature list, dataset coverage, historical averages |

---

## 🧠 How It Works

1. **No retraining** — all `.pkl` artifacts are loaded once at startup via `@st.cache_resource`
2. **Recursive multi-step forecast** — predicts one month at a time; each prediction is fed back as the next step's `lag_1` input
3. **27 selected features** — lag values (1, 2, 3, 6, 12), rolling statistics, temporal encodings (sin/cos month), categorical codes, and infrastructure indices
4. **Scenario multipliers** — Conservative ×0.85 / Normal ×1.00 / High Growth ×1.20 applied to raw model output
5. **170 unique series** — State × Vehicle Category (34 states × 5 categories); each series has its own 12-month seed history

---

## 📈 Model Performance (Test Set — 2023, held-out)

| Metric | Value |
|--------|-------|
| Algorithm | RandomForest (best_model.pkl) |
| Test R² | 0.9060 |
| Test RMSE | 717.54 |
| Test MAE | 180.89 |
| Test MAPE | 72.36% |
| Residual Std | 717.34 |
| Training Period | 2014 – 2022 |
| Test Period | 2023 (held-out) |
| Live Seed Date | January 2024 |
| Forecast Start | February 2024 |

---

## 📂 Dataset

- **Source:** Kaggle — [`mafzal19/electric-vehicle-sales-by-state-in-india`](https://www.kaggle.com/datasets/mafzal19/electric-vehicle-sales-by-state-in-india)
- **Records:** 96,845 rows, 8 columns
- **Columns:** Year, Month, Date, State, Vehicle_Class, Category, Type, Sales
- **Period:** January 2014 – January 2024
- **Coverage:** 34 states, 73 vehicle classes, 5 categories, 12 vehicle types
- **Zero-sales rows:** ~61.8% (handled natively by tree-based model)

---

## 📤 Batch Prediction CSV Format

Minimum required columns:

| Column | Example |
|--------|---------|
| `State` | Maharashtra |
| `Vehicle_Category` | 2-Wheelers |

Optional columns (override defaults per row):

| Column | Example |
|--------|---------|
| `Vehicle_Class` | MOTOR CYCLE / SCOOTER |
| `Vehicle_Type` | 2W_Personal |
| `Forecast_Horizon` | 6 |
| `Scenario` | Normal |

Download a pre-filled template from the **Batch Prediction** page inside the app.

---

## 🗂️ Key Source Files

| File | Purpose |
|------|---------|
| `app.py` | Page routing, sidebar controls, metric cards, chart tabs, session state |
| `utils.py` | `load_artifacts()`, `run_forecast()`, `run_batch_forecast()`, `generate_recommendation()`, all Plotly chart functions |
| `style.css` | Light-theme CSS for cards, sidebar, recommendation box, section headers |
| `.streamlit/config.toml` | Forces light theme across all browsers |
