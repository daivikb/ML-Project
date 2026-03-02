'''
This file is the main streamlit application.
It loads the artifacts from /artifacts and displays them in streamlit.
'''

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="MPG Predictor", layout="wide")

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
    mn = float(ranges.loc[col, "min"])
    mx = float(ranges.loc[col, "max"])
    return float(min(max(float(value), mn), mx))

def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a) / float(b)

def compute_ratios_from_row(row: pd.Series) -> dict:
    hpv = row.get("hpv", np.nan)
    displ = row.get("displ", np.nan)
    cylinders = row.get("cylinders", np.nan)
    return {
        "hp_per_liter": safe_div(hpv, displ),
        "liter_per_cyl": safe_div(displ, cylinders),
        "hp_per_cyl": safe_div(hpv, cylinders),
    }

def get_transformed_feature_names(pipeline):
    preprocess = pipeline.named_steps["preprocess"]
    num_features_out = preprocess.transformers_[0][2]
    cat_pipe = preprocess.transformers_[1][1]
    cat_features_in = preprocess.transformers_[1][2]
    ohe = cat_pipe.named_steps["onehot"]
    cat_names_out = ohe.get_feature_names_out(cat_features_in)
    return np.concatenate([np.array(num_features_out, dtype=str), cat_names_out])

def get_importance_series(pipeline):
    names = get_transformed_feature_names(pipeline)
    reg = pipeline.named_steps["regressor"]

    if hasattr(reg, "coef_"):
        vals = reg.coef_
        s = pd.Series(vals, index=names).sort_values(key=np.abs, ascending=False)
        return s, "Coefficient (signed)"

    if hasattr(reg, "feature_importances_"):
        vals = reg.feature_importances_
        s = pd.Series(vals, index=names).sort_values(ascending=False)
        return s, "Feature importance"

    return pd.Series(dtype=float), "Importance"

def predict_all_models(X_one: pd.DataFrame) -> pd.DataFrame:
    preds = []
    for name, pipe in models.items():
        try:
            yhat = float(pipe.predict(X_one)[0])
        except Exception:
            yhat = np.nan
        preds.append({"Model": name, "Predicted MPG": yhat})
    return pd.DataFrame(preds)

def build_X_from_row_dict(row: dict) -> pd.DataFrame:
    if selected_features:
        # fill missing
        for c in selected_features:
            if c not in row:
                row[c] = np.nan
        return pd.DataFrame([[row[c] for c in selected_features]], columns=selected_features)
    return pd.DataFrame([row])


# This is where the UI code for our streamlit app lives. The code above was simply loading in the artifacts
# -----------------------------
st.title("MPG Prediction Demo")
st.caption("Manual input + pick a real vehicle row, then compare model predictions + feature importance.")

tab_manual, tab_pick = st.tabs(["Manual Inputs", "Pick from vehicles.csv"])

# The manual inputs tab
with tab_manual:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Inputs")

        # This is a slider helper that uses loaded_row defaults, if present
        def slider(col, default=None, key=None):
            if "loaded_row" in st.session_state and col in st.session_state["loaded_row"]:
                default = st.session_state["loaded_row"][col]

            if col not in ranges.index:
                val = float(default) if default is not None and not pd.isna(default) else 0.0
                return st.number_input(col, value=val, key=key)

            mn = float(ranges.loc[col, "min"])
            mx = float(ranges.loc[col, "max"])

            if default is None or pd.isna(default):
                default = (mn + mx) / 2.0
            default = clamp_to_range(col, default)

            return st.slider(col, mn, mx, float(default), key=key)

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
                idx = opts.index(default_val) if default_val in opts else 0
                cat_inputs[col] = st.selectbox(col, options=opts, index=idx, key=f"cat_{col}")

        # Building X_user
        row = {**num_inputs, **cat_inputs}
        X_user = build_X_from_row_dict(row)

    with right:
        st.subheader("Predictions")

        model_names = list(models.keys())
        selected = st.radio("Primary model", model_names, horizontal=True)

        results_df = predict_all_models(X_user)
        selected_pred = float(results_df.loc[results_df["Model"] == selected, "Predicted MPG"].values[0])

        st.metric(label=f"{selected} predicted MPG", value=f"{selected_pred:.2f}")
        st.write("**All models:**")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Feature importance")
        top_k = st.slider("Top K", 5, 30, 15)

        imp_series, xlabel = get_importance_series(models[selected])
        if imp_series.empty:
            st.info("This model type does not expose coefficients/importances.")
        else:
            imp_series = imp_series.head(top_k)
            fig, ax = plt.subplots()
            ax.barh(imp_series.index[::-1], imp_series.values[::-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Feature")
            st.pyplot(fig)


# UI for if the user wants to pick from the vehicles.csv Tab
with tab_pick:
    st.subheader("Pick a real vehicle row to auto-fill Manual Inputs")

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
    st.write("**Selected vehicle:**")
    st.write({c: sample.get(c, None) for c in display_cols})

    actual = float(sample[TARGET]) if TARGET in sample and not pd.isna(sample[TARGET]) else np.nan
    st.write(f"**Actual MPG (comb08):** {actual:.2f}" if not pd.isna(actual) else "**Actual MPG (comb08):** N/A")

    # Build model input row (compute ratios from raw columns)
    # Building model input row: we're computing ratios from raw columns here
    ratios = compute_ratios_from_row(sample)
    row = {}

    # Numeric and ratio columns
    for col in numeric_features:
        if col in RATIO_COLS:
            row[col] = ratios.get(col, np.nan)
        else:
            row[col] = sample.get(col, np.nan)

    # Categorical features: basically strings
    for col in categorical_features:
        v = sample.get(col, "")
        row[col] = "" if pd.isna(v) else str(v)

    X_one = build_X_from_row_dict(row)

    st.write("### Predictions for this vehicle")
    preds_df = predict_all_models(X_one)
    st.dataframe(preds_df, use_container_width=True)

    if not pd.isna(actual):
        tmp = preds_df.copy()
        tmp["Actual MPG"] = actual
        tmp["Abs Error"] = (tmp["Predicted MPG"] - actual).abs()
        st.write("### Errors vs actual")
        st.dataframe(tmp.sort_values("Abs Error"), use_container_width=True)

    st.write("---")
    st.subheader("Load this example into Manual Inputs")
    if st.button("Load into input grid"):
        # Store the row for defaults then rerun, ensuring we show updated inforamation
        # The manual tab reads st.session_state["loaded_row"] during widget creation
        st.session_state["loaded_row"] = row
        st.success("Loaded! Switching to Manual Inputs (use the tab).")
        st.rerun()