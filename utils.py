"""
utils.py — Core inference utilities for the ChargeAgent Streamlit dashboard.

Reproduces the exact forecasting pipeline from the training notebook:
  Data → Feature Engineering → Feature Selection → Recursive Forecast → Post-processing

All logic uses only the saved .pkl artifacts — no retraining.
"""

from __future__ import annotations

import pickle
import joblib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1.  ARTIFACT LOADING
# ─────────────────────────────────────────────────────────────────

ARTIFACT_DIR = Path(__file__).parent


def _load_pkl(filename: str) -> Any:
    """Load a pickle file from the artifact directory."""
    path = ARTIFACT_DIR / filename
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_model(filename: str) -> Any:
    """Load a model file using joblib (handles sklearn models saved with joblib/pickle)."""
    path = ARTIFACT_DIR / filename
    return joblib.load(path)


def load_artifacts() -> dict[str, Any]:
    """
    Load every required .pkl artifact exactly once and return them as a
    single dictionary.  Called by Streamlit with @st.cache_resource so the
    heavy model is only unpickled once per session.
    """
    artifacts: dict[str, Any] = {}

    artifacts["model"]            = _load_model("best_model.pkl")
    artifacts["selected_features"] = _load_pkl("selected_features.pkl")
    artifacts["feature_columns"]   = _load_pkl("feature_columns.pkl")
    artifacts["scaler"]            = _load_pkl("scaler.pkl")
    artifacts["label_encoders"]    = _load_pkl("label_encoders.pkl")
    artifacts["metadata"]          = _load_pkl("metadata.pkl")
    artifacts["series_history"]    = _load_pkl("series_history.pkl")
    artifacts["forecast_config"]   = _load_pkl("forecast_config.pkl")

    return artifacts


# ─────────────────────────────────────────────────────────────────
# 2.  SCENARIO MULTIPLIERS  (identical to the Gradio app)
# ─────────────────────────────────────────────────────────────────

SCENARIO_MULTIPLIERS: dict[str, float] = {
    "Conservative": 0.85,
    "Normal":       1.00,
    "High Growth":  1.20,
}


# ─────────────────────────────────────────────────────────────────
# 3.  CORE FORECASTING ENGINE
# ─────────────────────────────────────────────────────────────────

def run_forecast(
    state: str,
    vehicle_category: str,
    vehicle_class: str,   # reporting filter only — used for display
    vehicle_type: str,    # reporting filter only — used for display
    horizon_months: int,
    scenario: str,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """
    Reproduce the exact recursive-forecast logic from the notebook Cell 33.

    Returns a dict with:
      preds           – list[float], monthly predictions for the horizon
      dates           – list[pd.Timestamp], corresponding month-start dates
      total_pred      – float, cumulative sales over the horizon
      growth_rate     – float, (last - first) / first
      cdi_final       – float, charging demand index at the last forecast step
      priority        – str, "HIGH" / "MEDIUM" / "LOW"
      confidence      – float, 0–1
      scenario_mult   – float, multiplier applied
      state           – str (echo)
      vehicle_category – str (echo)
      vehicle_class   – str (echo)
      vehicle_type    – str (echo)
      horizon_months  – int (echo)
      scenario        – str (echo)
    """
    model              = artifacts["model"]
    selected_features  = artifacts["selected_features"]
    meta               = artifacts["metadata"]
    series_history     = artifacts["series_history"]
    config             = artifacts["forecast_config"]

    scenario_mult: float = SCENARIO_MULTIPLIERS.get(scenario, 1.0)

    # --- Seed history for this (state, vehicle_category) series ---
    key = (state, vehicle_category)
    hist: list[float] = list(series_history.get(key, [0.0] * 12))
    # Ensure we always have at least 12 entries (pad with zeros if needed)
    while len(hist) < 12:
        hist.insert(0, 0.0)
    hist = hist[-12:]  # keep last 12

    # --- Index lookups (match notebook: states_list.index / cats_list.index) ---
    states_list     = meta["states"]
    cats_list       = meta["categories"]
    state_code      = states_list.index(state)
    category_code   = cats_list.index(vehicle_category)

    state_hist_avg_demand    = meta["state_hist_avg"].get(state, 0.0)
    category_hist_avg_demand = meta["category_hist_avg"].get(vehicle_category, 0.0)

    # --- Recursive step-by-step forecast ---
    cur_date: pd.Timestamp = pd.Timestamp(config["live_date"])
    preds:  list[float]           = []
    dates:  list[pd.Timestamp]    = []

    for _step in range(1, horizon_months + 1):
        cur_date = cur_date + pd.DateOffset(months=1)

        # Unpack lag / rolling values from current history window
        lag_1  = hist[-1]
        lag_2  = hist[-2]
        lag_3  = hist[-3]
        lag_6  = hist[-6]
        lag_12 = hist[-12]

        roll_mean_3  = float(np.mean(hist[-3:]))
        roll_mean_6  = float(np.mean(hist[-6:]))
        roll_mean_12 = float(np.mean(hist[-12:]))
        roll_std_3   = float(np.std(hist[-3:]))
        roll_std_6   = float(np.std(hist[-6:]))

        monthly_growth = (lag_1 - lag_2) / lag_2 if lag_2 else 0.0
        annual_growth  = (lag_1 - lag_12) / lag_12 if lag_12 else 0.0
        momentum       = lag_1 - lag_3
        cdi            = lag_1 / roll_mean_12 if roll_mean_12 else 0.0

        # Build feature row — matches notebook Cell 33 exactly
        row: dict[str, float] = {
            "state_code":               float(state_code),
            "category_code":            float(category_code),
            "Month":                    float(cur_date.month),
            "Quarter":                  float((cur_date.month - 1) // 3 + 1),
            "Year":                     float(cur_date.year),
            "Week":                     float(int(cur_date.isocalendar().week)),
            "sin_month":                float(np.sin(2 * np.pi * cur_date.month / 12)),
            "cos_month":                float(np.cos(2 * np.pi * cur_date.month / 12)),
            "lag_1":                    lag_1,
            "lag_2":                    lag_2,
            "lag_3":                    lag_3,
            "lag_6":                    lag_6,
            "lag_12":                   lag_12,
            "roll_mean_3":              roll_mean_3,
            "roll_mean_6":              roll_mean_6,
            "roll_mean_12":             roll_mean_12,
            "roll_std_3":               roll_std_3,
            "roll_std_6":               roll_std_6,
            "monthly_growth":           monthly_growth,
            "annual_growth":            annual_growth,
            "cagr_approx":              0.0,
            "momentum":                 momentum,
            "state_x_month":            float(state_code * cur_date.month),
            "cat_x_month":              float(category_code * cur_date.month),
            "charging_demand_index":    cdi,
            "demand_growth_index":      annual_growth,
            "state_hist_avg_demand":    state_hist_avg_demand,
            "category_hist_avg_demand": category_hist_avg_demand,
        }

        x_row = (
            pd.DataFrame([row])[selected_features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        raw_pred = float(model.predict(x_row)[0])
        pred = max(0.0, raw_pred) * scenario_mult

        # Update rolling history window
        hist.append(pred)
        hist = hist[-12:]

        preds.append(pred)
        dates.append(cur_date)

    # --- Post-processing metrics (notebook Cell 33) ---
    total_pred = float(sum(preds))
    mean_pred  = float(np.mean(preds)) if preds else 1.0
    cdi_final  = preds[-1] / mean_pred if mean_pred else 0.0
    growth_rate = (preds[-1] - preds[0]) / preds[0] if preds[0] else 0.0

    if growth_rate > 0.15 or cdi_final > 1.2:
        priority = "HIGH"
    elif growth_rate > 0.0 or cdi_final > 0.9:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    residual_std = meta["residual_std"]
    confidence = float(
        max(0.0, min(1.0, 1.0 - residual_std / (total_pred + residual_std + 1e-6)))
    )

    return {
        "preds":            preds,
        "dates":            dates,
        "total_pred":       total_pred,
        "growth_rate":      growth_rate,
        "cdi_final":        cdi_final,
        "priority":         priority,
        "confidence":       confidence,
        "scenario_mult":    scenario_mult,
        "state":            state,
        "vehicle_category": vehicle_category,
        "vehicle_class":    vehicle_class,
        "vehicle_type":     vehicle_type,
        "horizon_months":   horizon_months,
        "scenario":         scenario,
    }


# ─────────────────────────────────────────────────────────────────
# 4.  RECOMMENDATION GENERATOR  (template — matches notebook Cell 33)
# ─────────────────────────────────────────────────────────────────

def generate_recommendation(result: dict[str, Any]) -> str:
    """
    Build the AI recommendation paragraph from forecast results.
    Matches the template fallback in the notebook, extended with richer detail.
    """
    state         = result["state"]
    cat           = result["vehicle_category"]
    vc_class      = result["vehicle_class"]
    vtype         = result["vehicle_type"]
    horizon       = result["horizon_months"]
    total_pred    = result["total_pred"]
    cdi           = result["cdi_final"]
    growth        = result["growth_rate"]
    priority      = result["priority"]
    confidence    = result["confidence"]
    scenario      = result["scenario"]

    urgency_map = {
        "HIGH":   "immediate and high-priority",
        "MEDIUM": "near-term and phased",
        "LOW":    "monitored and deferred",
    }
    urgency = urgency_map.get(priority, "near-term")

    action_map = {
        "HIGH": (
            "Recommend deploying high-density fast-charging hubs along major urban "
            "corridors, expressways, and commercial zones. Target districts with "
            "highest registered EV density for immediate grid-level capacity upgrades."
        ),
        "MEDIUM": (
            "Recommend phased infrastructure rollout — prioritize city centers and "
            "key intercity highways. Plan quarterly demand reviews to align capex "
            "with actual adoption velocity."
        ),
        "LOW": (
            "Recommend deferring major charging-infrastructure capex. "
            "Monitor demand for 2–3 quarters and re-evaluate once adoption "
            "thresholds are met."
        ),
    }
    action = action_map.get(priority, action_map["MEDIUM"])

    scenario_note = {
        "Conservative": "Under a conservative scenario (−15% adjustment), ",
        "Normal":       "Under a baseline scenario, ",
        "High Growth":  "Under a high-growth scenario (+20% adjustment), ",
    }.get(scenario, "")

    rec = (
        f"<strong>{scenario_note}{state}</strong> exhibits "
        f"<strong>{urgency}</strong> charging-infrastructure "
        f"demand, driven primarily by <strong>{cat}</strong> adoption "
        f"(reporting filters: {vc_class} / {vtype}). "
        f"The {horizon}-month cumulative EV sales forecast is "
        f"<strong>{total_pred:,.0f} units</strong>, "
        f"with a projected growth rate of <strong>{growth * 100:.1f}%</strong> and a "
        f"Charging Demand Index of <strong>{cdi:.2f}</strong> "
        f"(&gt;1.2 = surging, 0.9–1.2 = stable, &lt;0.9 = declining). "
        f"Infrastructure priority is rated <strong>{priority}</strong> with a model confidence "
        f"score of <strong>{confidence:.0%}</strong>. "
        f"{action}"
    )
    return rec


# ─────────────────────────────────────────────────────────────────
# 5.  PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────

_COLORS = {
    "primary":   "#2E86AB",
    "secondary": "#A23B72",
    "accent":    "#F18F01",
    "success":   "#27ae60",
    "warning":   "#f39c12",
    "danger":    "#e74c3c",
    "bg":        "#ffffff",
    "card":      "#f8fafc",
    "grid":      "rgba(0,0,0,0.07)",
    "text":      "#1a202c",
    "subtext":   "#64748b",
}


def make_forecast_line_chart(result: dict[str, Any]) -> go.Figure:
    """
    Interactive Plotly line chart with shaded confidence band, markers,
    hover info, zoom, and download support.
    """
    preds   = result["preds"]
    dates   = result["dates"]
    horizon = result["horizon_months"]
    state   = result["state"]
    cat     = result["vehicle_category"]

    residual_std = 0.0  # Will be filled from meta in app.py if desired
    lower = [max(0.0, p - residual_std) for p in preds]
    upper = [p + residual_std for p in preds]

    date_strs = [d.strftime("%b %Y") for d in dates]

    fig = go.Figure()

    # Shaded confidence band
    fig.add_trace(go.Scatter(
        x=date_strs + date_strs[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Confidence Band",
        showlegend=False,
    ))

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=date_strs,
        y=preds,
        mode="lines+markers",
        name="Forecast EV Sales",
        line=dict(color=_COLORS["primary"], width=3),
        marker=dict(size=9, color=_COLORS["accent"], line=dict(color="white", width=2)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Forecast EV Sales: <b>%{y:,.0f}</b> units<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{horizon}-Month EV Demand Forecast — {state} / {cat}</b>",
            font=dict(size=16, color=_COLORS["text"]),
        ),
        xaxis=dict(
            title="Month",
            showgrid=True,
            gridcolor=_COLORS["grid"],
            tickfont=dict(color=_COLORS["subtext"]),
            title_font=dict(color=_COLORS["subtext"]),
            linecolor="#e2e8f0",
        ),
        yaxis=dict(
            title="Forecast EV Sales (Units)",
            showgrid=True,
            gridcolor=_COLORS["grid"],
            tickfont=dict(color=_COLORS["subtext"]),
            title_font=dict(color=_COLORS["subtext"]),
            linecolor="#e2e8f0",
        ),
        plot_bgcolor=_COLORS["bg"],
        paper_bgcolor=_COLORS["bg"],
        font=dict(color=_COLORS["text"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0.03)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(color=_COLORS["text"]),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_monthly_bar_chart(result: dict[str, Any]) -> go.Figure:
    """Monthly demand bar chart."""
    preds     = result["preds"]
    dates     = result["dates"]
    date_strs = [d.strftime("%b %Y") for d in dates]
    colors    = [_COLORS["accent"] if p == max(preds) else _COLORS["primary"] for p in preds]

    fig = go.Figure(go.Bar(
        x=date_strs,
        y=preds,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>EV Sales: <b>%{y:,.0f}</b> units<extra></extra>",
        name="Monthly EV Sales",
    ))
    fig.update_layout(
        title=dict(text="<b>Monthly Demand Breakdown</b>", font=dict(size=15, color=_COLORS["text"])),
        xaxis=dict(tickfont=dict(color=_COLORS["subtext"]), title_font=dict(color=_COLORS["subtext"]),
                   showgrid=False, linecolor="#e2e8f0"),
        yaxis=dict(
            title="EV Sales (Units)",
            tickfont=dict(color=_COLORS["subtext"]),
            title_font=dict(color=_COLORS["subtext"]),
            gridcolor=_COLORS["grid"],
        ),
        plot_bgcolor=_COLORS["bg"],
        paper_bgcolor=_COLORS["bg"],
        font=dict(color=_COLORS["text"]),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def make_growth_trend_chart(result: dict[str, Any]) -> go.Figure:
    """Month-over-month growth trend chart."""
    preds     = result["preds"]
    dates     = result["dates"]

    if len(preds) < 2:
        fig = go.Figure()
        fig.update_layout(title="<b>Growth Trend</b>", paper_bgcolor="#0e1117")
        return fig

    growth_rates = [
        (preds[i] - preds[i - 1]) / preds[i - 1] * 100 if preds[i - 1] else 0.0
        for i in range(1, len(preds))
    ]
    date_strs = [d.strftime("%b %Y") for d in dates[1:]]
    bar_colors = [
        _COLORS["success"] if g >= 0 else _COLORS["danger"] for g in growth_rates
    ]

    fig = go.Figure(go.Bar(
        x=date_strs,
        y=growth_rates,
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>MoM Growth: <b>%{y:.1f}%</b><extra></extra>",
        name="MoM Growth %",
    ))
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)
    fig.update_layout(
        title=dict(text="<b>Month-over-Month Growth Trend (%)</b>", font=dict(size=15, color=_COLORS["text"])),
        xaxis=dict(tickfont=dict(color=_COLORS["subtext"]), showgrid=False, linecolor="#e2e8f0"),
        yaxis=dict(
            title="MoM Growth (%)",
            tickfont=dict(color=_COLORS["subtext"]),
            title_font=dict(color=_COLORS["subtext"]),
            gridcolor=_COLORS["grid"],
        ),
        plot_bgcolor=_COLORS["bg"],
        paper_bgcolor=_COLORS["bg"],
        font=dict(color=_COLORS["text"]),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def make_charging_demand_gauge(cdi: float) -> go.Figure:
    """Charging Demand Index gauge chart."""
    color = (
        _COLORS["danger"]  if cdi > 1.5 else
        _COLORS["accent"]  if cdi > 1.2 else
        _COLORS["primary"] if cdi > 0.9 else
        "#7f8c8d"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(cdi, 3),
        title=dict(text="<b>Charging Demand Index</b>", font=dict(size=14, color=_COLORS["text"])),
        number=dict(font=dict(color=_COLORS["text"], size=28)),
        delta=dict(reference=1.0, valueformat=".2f"),
        gauge=dict(
            axis=dict(range=[0, 2.5], tickcolor=_COLORS["subtext"],
                      tickfont=dict(color=_COLORS["subtext"])),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0.04)",
            bordercolor="#e2e8f0",
            steps=[
                dict(range=[0, 0.9],   color="rgba(100,116,139,0.12)"),
                dict(range=[0.9, 1.2], color="rgba(46,134,171,0.12)"),
                dict(range=[1.2, 2.5], color="rgba(231,76,60,0.12)"),
            ],
            threshold=dict(
                line=dict(color=_COLORS["text"], width=2),
                thickness=0.75,
                value=1.0,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=_COLORS["bg"],
        font=dict(color=_COLORS["text"]),
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
    )
    return fig


def make_confidence_gauge(confidence: float) -> go.Figure:
    """Confidence score gauge chart."""
    pct = confidence * 100
    color = (
        _COLORS["success"] if pct >= 70 else
        _COLORS["accent"]  if pct >= 40 else
        _COLORS["danger"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pct, 1),
        title=dict(text="<b>Confidence Score (%)</b>", font=dict(size=14, color=_COLORS["text"])),
        number=dict(suffix="%", font=dict(color=_COLORS["text"], size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=_COLORS["subtext"],
                      tickfont=dict(color=_COLORS["subtext"])),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0.04)",
            bordercolor="#e2e8f0",
            steps=[
                dict(range=[0,  40], color="rgba(231,76,60,0.12)"),
                dict(range=[40, 70], color="rgba(241,143,1,0.12)"),
                dict(range=[70,100], color="rgba(39,174,96,0.12)"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor=_COLORS["bg"],
        font=dict(color=_COLORS["text"]),
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
    )
    return fig


def make_priority_indicator(priority: str) -> go.Figure:
    """Traffic-light style infrastructure priority indicator."""
    priority_map = {
        "HIGH":   (0.9, _COLORS["danger"],  "🔴 HIGH PRIORITY"),
        "MEDIUM": (0.6, _COLORS["accent"],  "🟡 MEDIUM PRIORITY"),
        "LOW":    (0.3, "#7f8c8d",          "⚪ LOW PRIORITY"),
    }
    value, color, label = priority_map.get(priority, priority_map["LOW"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title=dict(text=f"<b>Infrastructure Priority</b><br><span style='font-size:13px;color:{color}'>{label}</span>",
                   font=dict(size=13, color=_COLORS["text"])),
        number=dict(suffix="%", font=dict(color=color, size=24)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=_COLORS["subtext"],
                      tickfont=dict(color=_COLORS["subtext"]),
                      tickvals=[0, 30, 60, 90, 100]),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0.04)",
            bordercolor="#e2e8f0",
            steps=[
                dict(range=[0,  30], color="rgba(100,116,139,0.12)"),
                dict(range=[30, 60], color="rgba(241,143,1,0.12)"),
                dict(range=[60,100], color="rgba(231,76,60,0.12)"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor=_COLORS["bg"],
        font=dict(color=_COLORS["text"]),
        margin=dict(l=20, r=20, t=80, b=20),
        height=280,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# 6.  BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────

def run_batch_forecast(
    df_input: pd.DataFrame,
    horizon_months: int,
    scenario: str,
    artifacts: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the exact same single-row forecast pipeline to every row in the
    uploaded CSV.

    Expected CSV columns:
        State, Vehicle_Category, Vehicle_Class, Vehicle_Type
    Optional: Forecast_Horizon, Scenario (override per-row if present)

    Returns the input dataframe augmented with forecast columns.
    """
    meta               = artifacts["metadata"]
    states_list        = meta["states"]
    cats_list          = meta["categories"]

    # Validate & normalise columns
    required_cols = ["State", "Vehicle_Category"]
    missing = [c for c in required_cols if c not in df_input.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    results: list[dict] = []
    for _, row in df_input.iterrows():
        state = str(row["State"]).strip()
        cat   = str(row["Vehicle_Category"]).strip()
        vc    = str(row.get("Vehicle_Class", "N/A")).strip()
        vtype = str(row.get("Vehicle_Type", "N/A")).strip()

        # Validate values
        if state not in states_list:
            results.append({"error": f"Unknown state: {state}"})
            continue
        if cat not in cats_list:
            results.append({"error": f"Unknown vehicle category: {cat}"})
            continue

        h   = int(row["Forecast_Horizon"]) if "Forecast_Horizon" in row and pd.notna(row["Forecast_Horizon"]) else horizon_months
        sc  = str(row["Scenario"]) if "Scenario" in row and pd.notna(row["Scenario"]) else scenario

        try:
            res = run_forecast(state, cat, vc, vtype, h, sc, artifacts)
            monthly_cols = {f"Month_{i+1}_Sales": round(res["preds"][i], 2) for i in range(len(res["preds"]))}
            results.append({
                "State":              state,
                "Vehicle_Category":   cat,
                "Vehicle_Class":      vc,
                "Vehicle_Type":       vtype,
                "Forecast_Horizon":   h,
                "Scenario":           sc,
                "Total_Forecast_EV_Sales": round(res["total_pred"], 2),
                "Growth_Rate_%":      round(res["growth_rate"] * 100, 2),
                "Charging_Demand_Index": round(res["cdi_final"], 4),
                "Infrastructure_Priority": res["priority"],
                "Confidence_Score":   round(res["confidence"], 4),
                **monthly_cols,
            })
        except Exception as exc:
            results.append({"State": state, "Vehicle_Category": cat, "error": str(exc)})

    return pd.DataFrame(results)
