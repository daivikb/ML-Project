'''
This file is the main streamlit application.
It loads the artifacts from /artifacts and displays them in streamlit.
'''

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="MPG Predictor", layout="wide", page_icon="⛽")

# Configuration for our app
ART_DIR = "artifacts"
VEHICLES_CSV = "vehicles.csv"
TARGET = "comb08"

# Only have the three ratios we want here
RATIO_COLS = ["hp_per_liter", "liter_per_cyl", "hp_per_cyl"]

MODEL_PATHS = {
    "ElasticNet (tuned)": os.path.join(ART_DIR, "enet_tuned_pipeline.joblib"),
    "Lasso": os.path.join(ART_DIR, "lasso_pipeline.joblib"),
    "Ridge": os.path.join(ART_DIR, "ridge_pipeline.joblib"),
    "LightGBM": os.path.join(ART_DIR, "lgbm_pipeline.joblib"),
    "XGBoost": os.path.join(ART_DIR, "xgb_pipeline.joblib"),
}

RANGES_PATH = os.path.join(ART_DIR, "numeric_ranges.csv")
CAT_OPTIONS_PATH = os.path.join(ART_DIR, "categorical_options.joblib")
NUM_FEATURES_PATH = os.path.join(ART_DIR, "numeric_features.joblib")
CAT_FEATURES_PATH = os.path.join(ART_DIR, "categorical_features.joblib")
SELECTED_FEATURES_PATH = os.path.join(ART_DIR, "selected_features.joblib")


# Load the artifacts from /artifacts
if not os.path.exists(RANGES_PATH):
    st.error("Missing artifacts/numeric_ranges.csv:- Run train.py first.")
    st.stop()

ranges = pd.read_csv(RANGES_PATH, index_col=0)
cat_options = joblib.load(CAT_OPTIONS_PATH) if os.path.exists(CAT_OPTIONS_PATH) else {"drive": [], "trany": [], "VClass": [], "fuelType1": []}
numeric_features = joblib.load(NUM_FEATURES_PATH) if os.path.exists(NUM_FEATURES_PATH) else []
categorical_features = joblib.load(CAT_FEATURES_PATH) if os.path.exists(CAT_FEATURES_PATH) else ["drive", "trany", "VClass", "fuelType1"]
selected_features = joblib.load(SELECTED_FEATURES_PATH) if os.path.exists(SELECTED_FEATURES_PATH) else []

# It could be that older artifacts still have the extra ratios we don't want. If so, enforce a 3-ratio setup 
numeric_features = [c for c in numeric_features if c not in ["co2_per_hp", "fuelcost_per_hp"]]
for r in RATIO_COLS:
    if r not in numeric_features:
        numeric_features.append(r)

# Load our models in 
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

if len(models) == 0:
    st.error("No model artifacts found in artifacts/. Run train.py first.")
    st.stop()


# All the helpers methods we need for dealing with the data from artifacts
def clamp_to_range(col: str, value: float) -> float:
    if col not in ranges.index or value is None or pd.isna(value):
        return float(value) if value is not None and not pd.isna(value) else 0.0
    mn, mx = float(ranges.loc[col, "min"]), float(ranges.loc[col, "max"])
    return float(min(max(float(value), mn), mx))

def safe_div(a, b):
    return np.nan if (pd.isna(a) or pd.isna(b) or b == 0) else float(a) / float(b)

def compute_ratios_from_row(row):
    hpv, displ, cyl = row.get("hpv", np.nan), row.get("displ", np.nan), row.get("cylinders", np.nan)
    return {"hp_per_liter": safe_div(hpv, displ), "liter_per_cyl": safe_div(displ, cyl), "hp_per_cyl": safe_div(hpv, cyl)}

def get_transformed_feature_names(pipeline):
    pre = pipeline.named_steps["preprocess"]
    num_out = pre.transformers_[0][2]
    ohe = pre.transformers_[1][1].named_steps["onehot"]
    cat_out = ohe.get_feature_names_out(pre.transformers_[1][2])
    return np.concatenate([np.array(num_out, dtype=str), cat_out])

def get_importance_series(pipeline):
    names = get_transformed_feature_names(pipeline)
    reg = pipeline.named_steps["regressor"]
    if hasattr(reg, "coef_"):
        return pd.Series(reg.coef_, index=names).sort_values(key=np.abs, ascending=False), "Coefficient (signed)"
    if hasattr(reg, "feature_importances_"):
        return pd.Series(reg.feature_importances_, index=names).sort_values(ascending=False), "Feature importance"
    return pd.Series(dtype=float), "Importance"

def predict_all_models(X_one):
    preds = []
    for name, pipe in models.items():
        try:
            yhat = float(pipe.predict(X_one)[0])
        except Exception:
            yhat = np.nan
        preds.append({"Model": name, "Predicted MPG": round(yhat, 2)})
    return pd.DataFrame(preds)

def build_X_from_row_dict(row):
    if selected_features:
        for c in selected_features:
            if c not in row:
                row[c] = np.nan
        return pd.DataFrame([[row[c] for c in selected_features]], columns=selected_features)
    return pd.DataFrame([row])

# This is where the UI code for our streamlit app lives. The code above was simply loading in the artifacts
def mpg_color(mpg):
    if mpg < 20: return "#ff4444"
    if mpg < 28: return "#ffaa00"
    return "#00ff9d"

def plot_gauge(pred_mpg, model_name):
    color = mpg_color(pred_mpg)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_mpg,
        gauge={
            "axis": {"range": [0, 120], "tickcolor": "#555", "tickfont": {"color": "#888"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#111",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20],   "color": "#1f0f0f"},
                {"range": [20, 35],  "color": "#1f1a0a"},
                {"range": [35, 120], "color": "#0a1f14"},
            ],
        },
        title={"text": model_name, "font": {"color": "#888", "size": 13}},
        number={"suffix": " MPG", "font": {"size": 52, "color": color}},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=30, r=30),
                      paper_bgcolor="#0d0d0d", font_color="#f0ede6")
    return fig

def plot_importance(imp_series, xlabel, top_k):
    imp = imp_series.head(top_k)
    colors = ["#00ff9d" if v >= 0 else "#ff4444" for v in imp.values[::-1]]
    fig = go.Figure(go.Bar(
        x=imp.values[::-1], y=imp.index[::-1],
        orientation="h", marker_color=colors, marker_line_width=0,
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111", font_color="#f0ede6",
        height=max(300, top_k * 22),
        xaxis=dict(title=xlabel, gridcolor="#222", zeroline=True, zerolinecolor="#444"),
        yaxis=dict(gridcolor="#222"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

def plot_model_comparison(preds_df, actual=None):
    df = preds_df.dropna().copy()
    fig = go.Figure(go.Bar(
        x=df["Model"], y=df["Predicted MPG"],
        marker_color=[mpg_color(v) for v in df["Predicted MPG"]],
        marker_line_width=0,
        text=df["Predicted MPG"].round(1),
        textposition="outside", textfont=dict(color="#f0ede6"),
    ))
    if actual is not None and not pd.isna(actual):
        fig.add_hline(y=actual, line_dash="dash", line_color="white", line_width=1.5,
                      annotation_text=f"  Actual: {actual:.1f}", annotation_font_color="white")
    fig.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111", font_color="#f0ede6", height=300,
        yaxis=dict(title="MPG", gridcolor="#222"),
        xaxis=dict(gridcolor="#222"),
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    return fig

# Here, use custom CSS & HTML to style streamlit app's internal components (theming basically)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }
[data-testid="stMetric"] {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 10px; padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: white !important; }
[data-testid="stMetricLabel"] { color: white !important; }
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
.stRadio > div > label {
    background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
    padding: 0.3rem 0.75rem; font-size: 0.8rem;
    font-family: 'Space Mono', monospace; transition: border-color 0.2s;
}
.stRadio > div > label:hover { border-color: #00ff9d; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='border-bottom:1px solid #2a2a2a; padding-bottom:1rem; margin-bottom:1.5rem;'>
    <span style='font-family:Space Mono; font-size:0.65rem; color:#00ff9d; letter-spacing:4px;'>ECS 171 · MACHINE LEARNING</span>
    <h1 style='font-family:Space Mono; font-size:2.6rem; margin:0.2rem 0 0.1rem; color:#f0ede6;'>MPG Predictor</h1>
    <p style='color:#666; font-size:0.9rem; margin:0;'>Compare 5 regression models across 40,000+ EPA-rated vehicles in real time.</p>
</div>
""", unsafe_allow_html=True)

tab_manual, tab_pick = st.tabs(["✦ Manual Inputs", "✦ Pick from Dataset"])


# The manual inputs tab
with tab_manual:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Vehicle Parameters")

        # This is a slider helper that uses loaded_row defaults, if present
        def slider(col, default=None, key=None):
            if "loaded_row" in st.session_state and col in st.session_state["loaded_row"]:
                default = st.session_state["loaded_row"][col]
            if col not in ranges.index:
                val = float(default) if default is not None and not pd.isna(default) else 0.0
                return st.number_input(col, value=val, key=key)
            mn, mx = float(ranges.loc[col, "min"]), float(ranges.loc[col, "max"])
            if default is None or pd.isna(default):
                default = (mn + mx) / 2.0
            val = st.slider(col, mn, mx, float(clamp_to_range(col, default)), key=key)
            if col in ["year", "cylinders"]:
                return int(val)
            return val

        num_inputs = {col: slider(col, key=f"num_{col}") for col in numeric_features}

        # This is the numeric inputs slider
        num_inputs = {}
        for col in numeric_features:
            num_inputs[col] = slider(col, key=f"num_{col}")

        # Here, we're loading in our categorical inputs for the user to choose from
        cat_inputs = {}
        for col in categorical_features:
            opts = cat_options.get(col, [])
            loaded_val = None
            if "loaded_row" in st.session_state and col in st.session_state["loaded_row"]:
                loaded_val = str(st.session_state["loaded_row"][col]) if st.session_state["loaded_row"][col] is not None else ""
            if len(opts) == 0:
                cat_inputs[col] = st.text_input(col, value=loaded_val or "", key=f"cat_{col}")
            else:
                default_val = loaded_val if loaded_val in opts else opts[0]
                cat_inputs[col] = st.selectbox(col, options=opts, index=opts.index(default_val) if default_val in opts else 0, key=f"cat_{col}")

        # Building X_user
        row = {**num_inputs, **cat_inputs}
        X_user = build_X_from_row_dict(row)

    with right:
        st.subheader("Predictions")
        model_names = list(models.keys())
        selected = st.radio("Primary model", model_names, horizontal=True)

        results_df = predict_all_models(X_user)
        selected_pred = float(results_df.loc[results_df["Model"] == selected, "Predicted MPG"].values[0])

        st.plotly_chart(plot_gauge(selected_pred, selected), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted MPG", f"{selected_pred:.1f}")
        c2.metric("Efficiency", "High ↑" if selected_pred >= 28 else ("Mid →" if selected_pred >= 20 else "Low ↓"))
        c3.metric("Models Loaded", len(models))

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("All Models")
        st.plotly_chart(plot_model_comparison(results_df), use_container_width=True)

        st.subheader("Feature Importance")
        top_k = st.slider("Top K features", 5, 30, 15)
        imp_series, xlabel = get_importance_series(models[selected])
        if imp_series.empty:
            st.info("This model type does not expose feature importances.")
        else:
            st.plotly_chart(plot_importance(imp_series, xlabel, top_k), use_container_width=True)


# UI for if the user wants to pick from the vehicles.csv Tab
with tab_pick:
    st.subheader("Pick a Real Vehicle")

    if not os.path.exists(VEHICLES_CSV):
        st.error("vehicles.csv not found next to app.py.")
        st.stop()

    df = pd.read_csv(VEHICLES_CSV, low_memory=False)

    # keep only valid target rows: we don't really need the rest
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET] > 0]
    df = df.reset_index(drop=True)

    # Choose by row index here
    idx = st.number_input("Row index", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
    sample = df.loc[int(idx)]

    # Here we display/show what car it is (all the necessary stats/details)
    display_cols = [c for c in ["year", "make", "model", "VClass", "trany", "drive", "fuelType1"] if c in df.columns]
    actual = float(sample[TARGET]) if TARGET in sample and not pd.isna(sample[TARGET]) else np.nan

    info_cols = st.columns(len(display_cols))
    for i, col in enumerate(display_cols):
        info_cols[i].metric(col.capitalize(), str(sample.get(col, "—")))

    st.markdown("<br>", unsafe_allow_html=True)

    ratios = compute_ratios_from_row(sample)
    row = {}

    # Numeric and ratio columns
    for col in numeric_features:
        row[col] = ratios.get(col, np.nan) if col in RATIO_COLS else sample.get(col, np.nan)
    # Categorical features: basically strings
    for col in categorical_features:
        v = sample.get(col, "")
        row[col] = "" if pd.isna(v) else str(v)

    X_one = build_X_from_row_dict(row)
    preds_df = predict_all_models(X_one)

    left2, right2 = st.columns([1, 1], gap="large")

    with left2:
        st.subheader("Model Predictions vs Actual")
        st.plotly_chart(plot_model_comparison(preds_df, actual=actual), use_container_width=True)

    with right2:
        st.subheader("Error Breakdown")
        if not pd.isna(actual):
            tmp = preds_df.copy()
            tmp["Actual MPG"] = actual
            tmp["Abs Error"] = (tmp["Predicted MPG"] - actual).abs().round(2)
            tmp["% Error"] = ((tmp["Abs Error"] / actual) * 100).round(1).astype(str) + "%"
            st.dataframe(tmp.sort_values("Abs Error").reset_index(drop=True), use_container_width=True, hide_index=True)
            best = tmp.sort_values("Abs Error").iloc[0]
            st.success(f"Best prediction: **{best['Model']}** with {best['Abs Error']:.2f} MPG error")
        else:
            st.dataframe(preds_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("⟵ Load into Manual Inputs", use_container_width=False):
        st.session_state["loaded_row"] = row
        st.success("Loaded! Switch to the Manual Inputs tab.")
        st.rerun()