'''
This file trains the models and saves them to /artifacts. 
This file only needs to be ran once.
'''

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# LightGBM and XGBoost are treated as optional here: if you don't have these available, these will not be trained. 
# They also won't be added to the results table and will not be visible on app.py.
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


# Config 
INPUT_CSV = "vehicles.csv"
TARGET = "comb08"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# Helper function to eval model & return metrics (MAE, RMSE, R^2)
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2: {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# Load the input CSV into a dataframe, and do some basic target cleaning
vehicles_df = pd.read_csv(INPUT_CSV, low_memory=False)
vehicles_df = vehicles_df.dropna(subset=[TARGET])
vehicles_df = vehicles_df[vehicles_df[TARGET] > 0]

# Select features, and remove sources of leakage (directly correlated factors)
model_df = vehicles_df.copy()
model_df = model_df.drop(columns=["city08", "highway08"], errors="ignore")

# Avoid division by zero errs only where it matters
if "displ" in model_df.columns:
    model_df.loc[model_df["displ"] == 0, "displ"] = np.nan
if "cylinders" in model_df.columns:
    model_df.loc[model_df["cylinders"] == 0, "cylinders"] = np.nan
if "hpv" in model_df.columns:
    model_df.loc[model_df["hpv"] == 0, "hpv"] = np.nan

# Ratio features (these are only the 3 we want, and no more)
model_df["hp_per_liter"] = model_df["hpv"] / model_df["displ"]
model_df["liter_per_cyl"] = model_df["displ"] / model_df["cylinders"]
model_df["hp_per_cyl"] = model_df["hpv"] / model_df["cylinders"]

ratio_cols = ["hp_per_liter", "liter_per_cyl", "hp_per_cyl"]

# Drop rows where ratios or the target missing
model_df = model_df.dropna(subset=ratio_cols + [TARGET])

selected_features = [
    "year", "cylinders", "displ",
    "drive", "trany", "VClass", "fuelType1",
] + ratio_cols


# Split into X & Y
X = model_df[selected_features]
y = model_df[TARGET]

numeric_features = ["year", "cylinders", "displ"] + ratio_cols
categorical_features = ["drive", "trany", "VClass", "fuelType1"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

out_dir = "artifacts"
os.makedirs(out_dir, exist_ok=True)

# Here, save test set for Streamlit's "Pick from test set"
X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

# OPTIONAL but super useful:
# Save a "display" version that includes human-readable car info + actual mpg
display_cols = [c for c in ["year", "make", "model", "VClass", "trany", "drive", "fuelType1"] if c in vehicles_df.columns]
test_display = model_df.loc[X_test.index, display_cols + [TARGET]].copy()
test_display.to_csv(os.path.join(out_dir, "test_display.csv"), index=False)

# Save slider ranges and also the categorical options for Streamlit

ranges = pd.DataFrame({
    "min": X_train.select_dtypes(include="number").min(),
    "max": X_train.select_dtypes(include="number").max()
})
ranges_path = os.path.join(out_dir, "numeric_ranges.csv")
ranges.to_csv(ranges_path)

cat_options = {}
for col in categorical_features:
    cat_options[col] = sorted(pd.Series(X_train[col].astype(str).unique()).dropna().tolist())
joblib.dump(cat_options, os.path.join(out_dir, "categorical_options.joblib"))

joblib.dump(selected_features, os.path.join(out_dir, "selected_features.joblib"))
joblib.dump(numeric_features, os.path.join(out_dir, "numeric_features.joblib"))
joblib.dump(categorical_features, os.path.join(out_dir, "categorical_features.joblib"))
joblib.dump(ratio_cols, os.path.join(out_dir, "ratio_cols.joblib"))


# Models
print("\n--- Dummy ---")
dummy_model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", DummyRegressor(strategy="mean")),
])
dummy_model.fit(X_train, y_train)
dummy_pred = dummy_model.predict(X_test)
dummy_metrics = evaluate_model(y_test, dummy_pred)

print("\n--- ElasticNet (base) ---")
baseline_enet = Pipeline([
    ("preprocess", preprocess),
    ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)),
])
baseline_enet.fit(X_train, y_train)
enet_base_pred = baseline_enet.predict(X_test)
enet_base_metrics = evaluate_model(y_test, enet_base_pred)

print("\n--- ElasticNet (tuned GridSearchCV) ---")
enet_param_grid = {
    "regressor__alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
    "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}
enet_search = GridSearchCV(
    estimator=Pipeline([
        ("preprocess", preprocess),
        ("regressor", ElasticNet(max_iter=5000, random_state=42)),
    ]),
    param_grid=enet_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
enet_search.fit(X_train, y_train)
best_elastic_net = enet_search.best_estimator_
print("Best params:", enet_search.best_params_)
print(f"CV RMSE: {-enet_search.best_score_:.3f}")

enet_tuned_pred = best_elastic_net.predict(X_test)
enet_tuned_metrics = evaluate_model(y_test, enet_tuned_pred)

print("\n--- Lasso (tuned GridSearchCV) ---")
lasso = Pipeline([
    ("preprocess", preprocess),
    ("regressor", Lasso(max_iter=20000, random_state=42)),
])
lasso_parameters = {"regressor__alpha": np.logspace(-4, 2, 15)}
lasso_search = GridSearchCV(
    lasso,
    param_grid=lasso_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
lasso_search.fit(X_train, y_train)
print("Best Lasso parameters:", lasso_search.best_params_)
lasso_best = lasso_search.best_estimator_
lasso_pred = lasso_best.predict(X_test)
lasso_metrics = evaluate_model(y_test, lasso_pred)

print("\n--- Ridge (tuned GridSearchCV) ---")
ridge = Pipeline([
    ("preprocess", preprocess),
    ("regressor", Ridge(random_state=42)),
])
ridge_parameters = {"regressor__alpha": np.logspace(-4, 4, 15)}
ridge_search = GridSearchCV(
    ridge,
    param_grid=ridge_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
ridge_search.fit(X_train, y_train)
print("Best Ridge parameters:", ridge_search.best_params_)
print("Best CV RMSE:", -ridge_search.best_score_)
ridge_best = ridge_search.best_estimator_
ridge_pred = ridge_best.predict(X_test)
ridge_metrics = evaluate_model(y_test, ridge_pred)


# LightGBM (optional)
lgbm_best = None
lgbm_metrics = None
if lgb is not None:
    print("\n--- LightGBM (GridSearchCV) ---")
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgbm_pipeline = Pipeline([
        ("preprocess", preprocess),
        ("regressor", lgb_model),
    ])

    lgb_param_grid = {
        "regressor__n_estimators": [200, 500],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [-1, 10, 20],
        "regressor__num_leaves": [31, 50],
    }

    grid_lgb = GridSearchCV(
        estimator=lgbm_pipeline,
        param_grid=lgb_param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    grid_lgb.fit(X_train, y_train)
    lgbm_best = grid_lgb.best_estimator_
    print("Best Parameters:", grid_lgb.best_params_)

    lgbm_pred = lgbm_best.predict(X_test)
    lgbm_metrics = evaluate_model(y_test, lgbm_pred)
else:
    print("\n[SKIP] lightgbm not installed; skipping LightGBM training.")


# XGBoost (optional)
xgb_best = None
xgb_metrics = None
if XGBRegressor is not None:
    print("\n--- XGBoost (RandomizedSearchCV) ---")
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    xgb_pipeline = Pipeline([
        ("preprocess", preprocess),
        ("regressor", xgb),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    param_distributions = {
        "regressor__n_estimators": np.arange(300, 2000, 100),
        "regressor__learning_rate": np.linspace(0.01, 0.2, 100),
        "regressor__max_depth": [3, 4, 5, 6, 8, 10],
        "regressor__subsample": np.linspace(0.6, 1.0, 10),
        "regressor__colsample_bytree": np.linspace(0.6, 1.0, 10),
        "regressor__gamma": [0, 0.1, 0.3, 0.5],
        "regressor__reg_alpha": [0, 0.01, 0.1, 1],
        "regressor__reg_lambda": [0.5, 1, 1.5, 2, 3],
    }

    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="r2",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    xgb_best = search.best_estimator_
    print("Best CV R2:", search.best_score_)
    print("Best Params:", search.best_params_)

    xgb_pred = xgb_best.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred)
else:
    print("\n[SKIP] xgboost not installed; skipping XGBoost training.")


# Results table
rows = [
    {"model": "Dummy", **dummy_metrics},
    {"model": "ElasticNet(base)", **enet_base_metrics},
    {"model": "ElasticNet(tuned)", **enet_tuned_metrics},
    {"model": "Lasso", **lasso_metrics},
    {"model": "Ridge", **ridge_metrics},
]
if lgbm_metrics is not None:
    rows.append({"model": "LightGBM", **lgbm_metrics})
if xgb_metrics is not None:
    rows.append({"model": "XGBoost", **xgb_metrics})

results = pd.DataFrame(rows)
print("\n=== RESULTS ===")
print(results)


# Save artifacts for streamlit to use
joblib.dump(dummy_model, os.path.join(out_dir, "dummy_pipeline.joblib"))
joblib.dump(baseline_enet, os.path.join(out_dir, "enet_base_pipeline.joblib"))
joblib.dump(best_elastic_net, os.path.join(out_dir, "enet_tuned_pipeline.joblib"))
joblib.dump(lasso_best, os.path.join(out_dir, "lasso_pipeline.joblib"))
joblib.dump(ridge_best, os.path.join(out_dir, "ridge_pipeline.joblib"))

if lgbm_best is not None:
    joblib.dump(lgbm_best, os.path.join(out_dir, "lgbm_pipeline.joblib"))
if xgb_best is not None:
    joblib.dump(xgb_best, os.path.join(out_dir, "xgb_pipeline.joblib"))

results.to_csv(os.path.join(out_dir, "results.csv"), index=False)

print(f"\nSaved models + metadata to: {out_dir}/")