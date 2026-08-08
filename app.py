"""
app.py — ChargeAgent: Enterprise EV Demand & Charging Infrastructure Forecasting
Streamlit Application

Reproduces the exact prediction pipeline from mandeep-ieee.ipynb using only
saved .pkl artifacts. No retraining occurs at inference time.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    load_artifacts,
    run_forecast,
    run_batch_forecast,
    generate_recommendation,
    make_forecast_line_chart,
    make_monthly_bar_chart,
    make_growth_trend_chart,
    make_charging_demand_gauge,
    make_confidence_gauge,
    make_priority_indicator,
    SCENARIO_MULTIPLIERS,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChargeAgent — EV Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "ChargeAgent — Regional EV Demand & Charging Infrastructure Forecasting",
    },
)


# ─────────────────────────────────────────────────────────────────
# INJECT CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
def _load_css() -> None:
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


_load_css()


# ─────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS  (cached — only once per session)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()
_meta     = artifacts["metadata"]
_config   = artifacts["forecast_config"]

# Lookup lists loaded from metadata (never hardcoded)
STATES          = _meta["states"]
CATEGORIES      = _meta["categories"]
VEHICLE_CLASSES = _meta["vehicle_classes"]
VEHICLE_TYPES   = _meta["vehicle_types"]


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def _metric_card(icon: str, value: str, label: str, css_class: str = "") -> str:
    """Return HTML for a metric card."""
    cls = f"metric-card {css_class}".strip()
    return (
        f'<div class="{cls}">'
        f'<span class="metric-icon">{icon}</span>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def _section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-icon">{icon}</span>'
        f'<h3>{title}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _priority_css(priority: str) -> str:
    return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(priority, "low")


def _format_number(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


# ─────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT CONTROLS
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo & branding
    st.markdown(
        '<div class="sidebar-logo">'
        '<div style="font-size:2.4rem">⚡</div>'
        '<h2>ChargeAgent</h2>'
        '<div style="font-size:0.72rem;color:#5a6a7e;margin-top:2px;">EV Forecasting Dashboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Navigation
    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Page",
        ["🔮 Forecast Dashboard", "📦 Batch Prediction", "📊 Model Information"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Input controls (always visible) ──
    st.markdown('<div class="sidebar-section-label">📍 Location & Segment</div>', unsafe_allow_html=True)

    default_state = "Maharashtra" if "Maharashtra" in STATES else STATES[0]
    selected_state = st.selectbox(
        "State",
        STATES,
        index=STATES.index(default_state),
        help="Select the Indian state for demand forecasting.",
    )

    default_cat = "2-Wheelers" if "2-Wheelers" in CATEGORIES else CATEGORIES[0]
    selected_category = st.selectbox(
        "Vehicle Category",
        CATEGORIES,
        index=CATEGORIES.index(default_cat),
        help="Primary modelling grain. Forecasts are produced at State × Category level.",
    )

    selected_class = st.selectbox(
        "Vehicle Class (Reporting Filter)",
        VEHICLE_CLASSES,
        index=0,
        help="Reporting filter only — does not change the model output, used for display context.",
    )

    selected_type = st.selectbox(
        "Vehicle Type (Reporting Filter)",
        VEHICLE_TYPES,
        index=0,
        help="Reporting filter only — used for display context in recommendation text.",
    )

    st.markdown('<div class="sidebar-section-label">⏱️ Forecast Settings</div>', unsafe_allow_html=True)

    horizon_months = st.radio(
        "Forecast Horizon (Months)",
        [3, 6, 12],
        index=1,
        horizontal=True,
        help="Number of months to forecast ahead from January 2024.",
    )

    scenario = st.radio(
        "Scenario",
        list(SCENARIO_MULTIPLIERS.keys()),
        index=1,
        help=(
            "Conservative: ×0.85 | Normal: ×1.00 | High Growth: ×1.20  "
            "(multiplier applied to raw model output — does not retrain the model)"
        ),
    )

    st.markdown("---")

    # Generate button
    generate_clicked = st.button("⚡  Generate Forecast", use_container_width=True)

    # Status indicator
    st.markdown(
        '<div class="sidebar-info">'
        f'<strong>Model:</strong> {_meta["best_model_name"]}<br>'
        f'<strong>Test R²:</strong> {_meta["test_r2"]:.4f}<br>'
        f'<strong>Live seed:</strong> {_config["live_date"]}'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA — PAGE ROUTING
# ─────────────────────────────────────────────────────────────────

# ── GLOBAL HEADER ──
st.markdown(
    '<div class="header-banner">'
    '<span class="header-badge">IEEE Research</span>'
    '<span class="header-badge">AI-Powered</span>'
    '<span class="header-badge">India EV Market</span>'
    '<div class="header-title">⚡ ChargeAgent</div>'
    '<p class="header-subtitle">'
    'Autonomous Multi-Agent AI Framework for Regional EV Demand Forecasting '
    '&amp; Charging Infrastructure Planning — India EV Market 2014–2024'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — FORECAST DASHBOARD
# ═══════════════════════════════════════════════════════════════
if "🔮 Forecast Dashboard" in page:

    # Persist last forecast result across reruns using session state
    if "forecast_result" not in st.session_state:
        st.session_state["forecast_result"] = None

    # Run forecast when button is clicked
    if generate_clicked:
        with st.spinner(f"Running {horizon_months}-month recursive forecast for "
                        f"{selected_state} / {selected_category}…"):
            try:
                result = run_forecast(
                    state=selected_state,
                    vehicle_category=selected_category,
                    vehicle_class=selected_class,
                    vehicle_type=selected_type,
                    horizon_months=horizon_months,
                    scenario=scenario,
                    artifacts=artifacts,
                )
                st.session_state["forecast_result"] = result
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                st.session_state["forecast_result"] = None

    result = st.session_state.get("forecast_result")

    # ── Placeholder when no forecast yet ──
    if result is None:
        st.markdown(
            '<div style="text-align:center;padding:80px 20px;">'
            '<div style="font-size:4rem;margin-bottom:16px;">🔮</div>'
            '<h3 style="color:#5a6a7e;font-weight:600;">Select parameters and click <em>Generate Forecast</em></h3>'
            '<p style="color:#3d4a5c;font-size:0.95rem;">Configure State, Vehicle Category, Horizon, and Scenario in the left panel.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()


    # ── FORECAST SUMMARY METRICS ──
    _section_header("📊", "Forecast Summary")

    priority_css = _priority_css(result["priority"])
    confidence_pct = f"{result['confidence']:.0%}"
    growth_pct = f"{result['growth_rate']*100:+.1f}%"

    cols = st.columns(5)
    with cols[0]:
        st.markdown(
            _metric_card("🚗", _format_number(result["total_pred"]),
                         f"Forecast EV Sales ({result['horizon_months']}m)"),
            unsafe_allow_html=True,
        )
    with cols[1]:
        g_css = "high" if result["growth_rate"] > 0.15 else ("medium" if result["growth_rate"] > 0 else "low")
        st.markdown(
            _metric_card("📈", growth_pct, "Growth Rate", g_css),
            unsafe_allow_html=True,
        )
    with cols[2]:
        cdi_val = f"{result['cdi_final']:.2f}"
        cdi_css = "high" if result["cdi_final"] > 1.2 else ("medium" if result["cdi_final"] > 0.9 else "low")
        st.markdown(
            _metric_card("⚡", cdi_val, "Charging Demand Index", cdi_css),
            unsafe_allow_html=True,
        )
    with cols[3]:
        p_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}.get(result["priority"], "⚪")
        st.markdown(
            _metric_card(p_icon, result["priority"], "Infrastructure Priority", priority_css),
            unsafe_allow_html=True,
        )
    with cols[4]:
        conf_css = "high" if result["confidence"] < 0.4 else ("medium" if result["confidence"] < 0.7 else "")
        st.markdown(
            _metric_card("🎯", confidence_pct, "Confidence Score", conf_css),
            unsafe_allow_html=True,
        )


    # ── AI RECOMMENDATION ──
    _section_header("🤖", "AI Infrastructure Recommendation")
    recommendation = generate_recommendation(result)
    st.markdown(
        f'<div class="recommendation-box">'
        f'<div class="rec-title">💡 AI Analysis &amp; Strategic Recommendation</div>'
        f'{recommendation}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── MAIN FORECAST VISUALIZATION (Tabs) ──
    _section_header("📉", "Interactive Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast Trend",
        "📊 Monthly Demand",
        "📉 Growth Analysis",
        "🔧 Infrastructure Gauges",
    ])

    with tab1:
        fig_line = make_forecast_line_chart(result)
        # Inject real residual std for confidence band
        residual_std = _meta["residual_std"]
        fig_line.data[0].y = (
            [p + residual_std for p in result["preds"]] +
            [max(0, p - residual_std) for p in result["preds"]][::-1]
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Download forecast data
        df_dl = pd.DataFrame({
            "Month": [d.strftime("%b %Y") for d in result["dates"]],
            "Forecast_EV_Sales": [round(p, 2) for p in result["preds"]],
        })
        csv_bytes = df_dl.to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download Forecast CSV",
            data=csv_bytes,
            file_name=f"forecast_{selected_state}_{selected_category}_{horizon_months}m.csv",
            mime="text/csv",
        )

    with tab2:
        fig_bar = make_monthly_bar_chart(result)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        fig_growth = make_growth_trend_chart(result)
        st.plotly_chart(fig_growth, use_container_width=True)

    with tab4:
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.plotly_chart(make_charging_demand_gauge(result["cdi_final"]),
                            use_container_width=True)
        with g_col2:
            st.plotly_chart(make_priority_indicator(result["priority"]),
                            use_container_width=True)
        with g_col3:
            st.plotly_chart(make_confidence_gauge(result["confidence"]),
                            use_container_width=True)


    # ── MONTHLY DATA TABLE ──
    st.markdown("---")
    with st.expander("📋 Monthly Forecast Data Table", expanded=False):
        df_table = pd.DataFrame({
            "Month":               [d.strftime("%B %Y") for d in result["dates"]],
            "Forecast EV Sales":   [round(p, 0) for p in result["preds"]],
            "MoM Change":          [
                "—" if i == 0 else f"{(result['preds'][i]-result['preds'][i-1])/result['preds'][i-1]*100:+.1f}%"
                if result["preds"][i-1] else "N/A"
                for i in range(len(result["preds"]))
            ],
            "Cumulative Sales":    [round(sum(result["preds"][:i+1]), 0) for i in range(len(result["preds"]))],
        })
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── FORECAST PARAMETERS SUMMARY ──
    with st.expander("⚙️ Forecast Parameters", expanded=False):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"**State:** {result['state']}")
            st.markdown(f"**Vehicle Category:** {result['vehicle_category']}")
            st.markdown(f"**Vehicle Class (filter):** {result['vehicle_class']}")
            st.markdown(f"**Vehicle Type (filter):** {result['vehicle_type']}")
        with p_col2:
            st.markdown(f"**Horizon:** {result['horizon_months']} months")
            st.markdown(f"**Scenario:** {result['scenario']} (×{result['scenario_mult']:.2f})")
            st.markdown(f"**Forecast start:** February 2024")
            st.markdown(f"**Seed date:** {_config['live_date']}")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════
elif "📦 Batch Prediction" in page:

    _section_header("📦", "Batch Prediction")

    st.markdown(
        '<div class="info-box">'
        '<strong>How it works:</strong> Upload a CSV with columns '
        '<code>State</code>, <code>Vehicle_Category</code>. '
        'Optional: <code>Vehicle_Class</code>, <code>Vehicle_Type</code>, '
        '<code>Forecast_Horizon</code>, <code>Scenario</code>. '
        'The same preprocessing pipeline is applied to every row, '
        'and predictions are generated using the saved model artifacts.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### Default Settings for Batch")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        batch_horizon = st.selectbox("Default Forecast Horizon (months)", [3, 6, 12], index=1)
    with b_col2:
        batch_scenario = st.selectbox("Default Scenario", list(SCENARIO_MULTIPLIERS.keys()), index=1)

    # Template download
    template_df = pd.DataFrame({
        "State":            ["Maharashtra", "Delhi", "Karnataka"],
        "Vehicle_Category": ["2-Wheelers", "4-Wheelers", "3-Wheelers"],
        "Vehicle_Class":    ["MOTOR CYCLE / SCOOTER", "MOTOR CAR", "E-RICKSHAW"],
        "Vehicle_Type":     ["2W_Personal", "4W_Personal", "3W_Shared"],
        "Forecast_Horizon": [6, 12, 3],
        "Scenario":         ["Normal", "High Growth", "Conservative"],
    })
    st.download_button(
        label="⬇️  Download CSV Template",
        data=template_df.to_csv(index=False).encode(),
        file_name="batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload CSV for batch forecasting",
        type=["csv"],
        help="Must include at minimum: State and Vehicle_Category columns.",
    )

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**Uploaded:** {len(df_upload)} rows × {len(df_upload.columns)} columns")
            with st.expander("Preview uploaded data"):
                st.dataframe(df_upload.head(10), use_container_width=True, hide_index=True)

            if st.button("🚀 Run Batch Forecast", use_container_width=True):
                with st.spinner(f"Processing {len(df_upload)} records…"):
                    progress = st.progress(0, text="Starting batch forecast…")
                    df_results = run_batch_forecast(
                        df_input=df_upload,
                        horizon_months=batch_horizon,
                        scenario=batch_scenario,
                        artifacts=artifacts,
                    )
                    progress.progress(100, text="Complete!")

                st.success(f"✅ Batch forecast complete — {len(df_results)} records processed.")
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                st.download_button(
                    label="⬇️  Download Batch Results CSV",
                    data=df_results.to_csv(index=False).encode(),
                    file_name="batch_forecast_results.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Error processing file: {exc}")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFORMATION
# ═══════════════════════════════════════════════════════════════
elif "📊 Model Information" in page:

    _section_header("📊", "Model Information")

    # ── Model performance metrics ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            _metric_card("🧠", _meta["best_model_name"], "Algorithm"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            _metric_card("🎯", f"{_meta['test_r2']:.4f}", "Test R² Score"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            _metric_card("📉", f"{_meta['test_rmse']:.1f}", "Test RMSE"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            _metric_card("📊", f"{_meta['test_mae']:.1f}", "Test MAE"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        _section_header("🔬", "Model Details")

        sf = artifacts["selected_features"]
        fc = artifacts["feature_columns"]

        def _info_row(key: str, val: str) -> str:
            return (
                f'<div class="model-info-row">'
                f'<span class="model-info-key">{key}</span>'
                f'<span class="model-info-val">{val}</span>'
                f'</div>'
            )

        info_html = (
            _info_row("Algorithm", _meta["best_model_name"]) +
            _info_row("Model Type", "Ensemble of Decision Trees (RandomForest)") +
            _info_row("Total Candidate Features", str(len(fc))) +
            _info_row("Selected Features", str(len(sf))) +
            _info_row("Scaling Method", "StandardScaler (fit on train only; trees use raw features)") +
            _info_row("Encoding Method", "LabelEncoder (State & Vehicle_Category)") +
            _info_row("Forecast Type", "Recursive multi-step (auto-regressive)") +
            _info_row("Train Period", f"{_config['train_end']} (2014–2021)") +
            _info_row("Validation Period", f"{_config.get('val_start', 'N/A')} → {_config.get('val_end', 'N/A')}") +
            _info_row("Test Period", f"{_config['test_start']} → {_config['test_end']}") +
            _info_row("Live Seed Date", _config["live_date"]) +
            _info_row("Test MAPE", f"{_meta['test_mape']:.2f}%") +
            _info_row("Residual Std", f"{_meta['residual_std']:.2f}")
        )
        st.markdown(f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;">{info_html}</div>',
                    unsafe_allow_html=True)

    with info_col2:
        _section_header("🗂️", "Dataset & Coverage")

        ds_html = (
            _info_row("Dataset", "India EV Market (Kaggle: mafzal19)") +
            _info_row("Total Records", "96,845 rows") +
            _info_row("Columns", "8 (Year, Month, Date, State, Vehicle_Class, Category, Type, Sales)") +
            _info_row("Date Range", "January 2014 – January 2024") +
            _info_row("Unique States", str(len(STATES))) +
            _info_row("Vehicle Categories", ", ".join(CATEGORIES)) +
            _info_row("Unique Vehicle Classes", str(len(VEHICLE_CLASSES))) +
            _info_row("Unique Vehicle Types", str(len(VEHICLE_TYPES))) +
            _info_row("Modelling Grain", "State × Vehicle_Category (170 series)") +
            _info_row("Zero-Sales Rows", "~61.8% (tree models handle this natively)")
        )
        st.markdown(f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;">{ds_html}</div>',
                    unsafe_allow_html=True)

    # ── Selected Features ──
    st.markdown("---")
    _section_header("🔢", "Selected Features")

    sf_data = [{"#": i+1, "Feature": f, "Type": (
        "Lag" if f.startswith("lag_") else
        "Rolling" if f.startswith("roll_") else
        "Temporal" if f in ["Month","Quarter","Year","Week","sin_month","cos_month"] else
        "Categorical" if f in ["state_code","category_code"] else
        "Growth/Trend" if "growth" in f or f in ["momentum","cagr_approx"] else
        "Interaction" if "_x_" in f else
        "Infrastructure"
    )} for i, f in enumerate(sf)]

    st.dataframe(pd.DataFrame(sf_data), use_container_width=True, hide_index=True)

    # ── Forecast Config ──
    st.markdown("---")
    with st.expander("⚙️ Forecast Configuration (from forecast_config.pkl)", expanded=False):
        st.json(_config)

    # ── State & Category Historical Averages ──
    with st.expander("📈 Historical Average EV Sales by State (training window)", expanded=False):
        df_state_avg = pd.DataFrame(
            list(_meta["state_hist_avg"].items()),
            columns=["State", "Avg Monthly EV Sales (Train)"]
        ).sort_values("Avg Monthly EV Sales (Train)", ascending=False).reset_index(drop=True)
        st.dataframe(df_state_avg, use_container_width=True, hide_index=True)

    with st.expander("📈 Historical Average EV Sales by Category (training window)", expanded=False):
        df_cat_avg = pd.DataFrame(
            list(_meta["category_hist_avg"].items()),
            columns=["Vehicle Category", "Avg Monthly EV Sales (Train)"]
        ).sort_values("Avg Monthly EV Sales (Train)", ascending=False).reset_index(drop=True)
        st.dataframe(df_cat_avg, use_container_width=True, hide_index=True)



# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:24px 0 8px 0;'
    'color:#94a3b8;font-size:0.8rem;border-top:1px solid #e2e8f0;'
    'margin-top:40px;">'
    '⚡ ChargeAgent — Regional EV Demand &amp; Charging Infrastructure Planner &nbsp;|&nbsp; '
    'India EV Market 2014–2024 &nbsp;|&nbsp; '
    f'Model: {_meta["best_model_name"]} &nbsp;|&nbsp; Test R² = {_meta["test_r2"]:.4f}'
    '</div>',
    unsafe_allow_html=True,
)
