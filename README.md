# ⚡ ChargeAgent — EV Demand & Charging Infrastructure Forecasting Dashboard

**Enterprise-grade Streamlit application for regional EV demand forecasting and
charging infrastructure planning across Indian states.**

Built for IEEE publication research. Reproduces the exact prediction pipeline
from the training notebook using only pre-trained `.pkl` artifacts — no retraining,
and the notebook is not required to run this application.

---

## 📁 Project Files

```
MANDEEP ASSIGNMENT/
├── app.py                  # Main Streamlit application (production-ready)
├── utils.py                # Core inference engine & Plotly chart utilities
├── style.css               # Custom light-theme enterprise dashboard CSS
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Forces light theme, removes dark mode toggle
│
│   — Pre-trained model artifacts —
├── best_model.pkl          # Trained XGBoost model (Train+Val, 2014–2022)
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

Make sure you have the following installed before starting:

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| Python | 3.9 or higher | `python --version` |
| pip | Latest | `pip --version` |

> **Don't have Python?** Download it from [https://www.python.org/downloads/](https://www.python.org/downloads/)  
> ✔ During installation on Windows — check **"Add Python to PATH"**

---

### 📁 Step 1 — Copy the Project Folder

Copy the **MANDEEP ASSIGNMENT** folder to your system. Only the files listed below are needed to run the app — the original training notebook (`mandeep.ipynb`) is **not required** and does not need to be included.

```
MANDEEP ASSIGNMENT/
├── app.py                    ← Main Streamlit application
├── utils.py                  ← Inference engine & chart utilities
├── style.css                 ← Dashboard styling
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── .streamlit/
│   └── config.toml           ← Forces light theme
│
│   — Pre-trained model artifacts (all required) —
├── best_model.pkl
├── feature_columns.pkl
├── forecast_config.pkl
├── label_encoders.pkl
├── metadata.pkl
├── scaler.pkl
├── selected_features.pkl
└── series_history.pkl
```

> ⚠️ All 8 `.pkl` files are required. The app will not start if any are missing.  
> ✅ The notebook `mandeep.ipynb` is **not needed** — the model is already trained and saved.

---

### 💻 Step 2 — Open a Terminal in the Project Folder

**Windows:**
1. Open the `MANDEEP ASSIGNMENT` folder in File Explorer
2. Click the address bar, type `cmd`, press Enter

**macOS / Linux:**
1. Open Terminal
2. Run: `cd "/path/to/MANDEEP ASSIGNMENT"`

---

### 📦 Step 3 — Install Dependencies

Run this single command to install all required packages:

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `plotly`

> ⏳ First-time install may take 2–3 minutes depending on your internet speed.

**If you get a permissions error on Windows, try:**
```bash
pip install -r requirements.txt --user
```

**If you get a permissions error on macOS/Linux, try:**
```bash
pip3 install -r requirements.txt
```

---

### ▶️ Step 4 — Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at:

```
http://localhost:8501
```

> If the browser doesn't open automatically, copy the URL and paste it manually.

---

### 🛑 How to Stop the App

Press `Ctrl + C` in the terminal window.

---

## 🖥️ Platform-Specific Notes

### Windows

```cmd
cd "C:\path\to\MANDEEP ASSIGNMENT"
pip install -r requirements.txt
streamlit run app.py
```

### macOS

```bash
cd "/path/to/MANDEEP ASSIGNMENT"
pip3 install -r requirements.txt
streamlit run app.py
```

### Linux

```bash
cd "/path/to/MANDEEP ASSIGNMENT"
pip3 install -r requirements.txt
streamlit run app.py
```

---

## 🐍 Using a Virtual Environment (Recommended)

A virtual environment keeps the project dependencies isolated from your system Python.

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

To deactivate the virtual environment when done:
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
| 🔮 Forecast Dashboard | Metric cards, AI recommendation, interactive charts |
| 📦 Batch Prediction | Upload CSV → forecast all rows → download results |
| 📊 Model Information | Algorithm details, features, dataset coverage |

---

## 🧠 How It Works

1. **No retraining** — all `.pkl` files are loaded once at startup
2. **Recursive forecast** — predicts one month at a time, feeding each prediction back as input for the next step
3. **Scenario multipliers** — Conservative ×0.85 / Normal ×1.00 / High Growth ×1.20
4. **Model grain** — State × Vehicle Category (34 × 5 = 170 series)

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost |
| Test R² | 0.8644 |
| Test RMSE | 862.04 |
| Test MAE | 201.95 |
| Test MAPE | 68.24% |
| Training Period | 2014 – 2022 |
| Test Period | 2023 (held-out) |

---

## 📂 Dataset

- **Source:** Kaggle — `mafzal19/electric-vehicle-sales-by-state-in-india`
- **Records:** 96,845 rows, 8 columns
- **Period:** January 2014 – January 2024
- **Coverage:** 34 states, 73 vehicle classes, 5 categories, 12 vehicle types
