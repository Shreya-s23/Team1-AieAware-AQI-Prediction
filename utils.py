"""
utils.py — Shared utilities for AirAware
Place this in the same folder as all other files.
"""

import hashlib
import json
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
USERS_FILE  = "users.json"
MODEL_FILE  = "aqi_xgboost_model.pkl"
FEATS_FILE  = "aqi_feature_columns.pkl"
METRICS_FILE= "aqi_model_metrics.json"

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Default users
    users = {
        "admin": {
            "password": _hash("admin123"),
            "role": "admin",
            "name": "Administrator"
        },
        "user": {
            "password": _hash("user123"),
            "role": "user",
            "name": "User"
        }
    }
    save_users(users)
    return users

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def register_user(username: str, password: str, name: str) -> tuple[bool, str]:
    users = load_users()
    if username in users:
        return False, "Username already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    users[username] = {
        "password": _hash(password),
        "role": "user",
        "name": name
    }
    save_users(users)
    return True, "Account created successfully."

def authenticate(username: str, password: str) -> tuple[bool, dict]:
    users = load_users()
    if username not in users:
        return False, {}
    u = users[username]
    if u["password"] == _hash(password):
        return True, {"username": username, "role": u["role"], "name": u["name"]}
    return False, {}

def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)

def current_user() -> dict:
    return st.session_state.get("user_info", {})

def logout():
    for key in ["logged_in", "user_info", "role"]:
        if key in st.session_state:
            del st.session_state[key]

# ─────────────────────────────────────────────────────────────────────────────
#  AQI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",                  "#059669"),
    (51,  100, "Moderate",              "#d97706"),
    (101, 150, "Unhealthy — Sensitive", "#ea580c"),
    (151, 200, "Unhealthy",             "#dc2626"),
    (201, 300, "Very Unhealthy",        "#7c3aed"),
    (301, 999, "Hazardous",             "#9f1239"),
]

def classify_aqi(val):
    val = float(max(0, val))
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= val <= hi:
            return label, color
    return "Hazardous", "#9f1239"

def aqi_advice(label):
    d = {
        "Good": "Air quality is satisfactory. Outdoor activities are safe for all.",
        "Moderate": "Acceptable quality. Unusually sensitive individuals may notice mild effects.",
        "Unhealthy — Sensitive": "Children, elderly and those with respiratory conditions should reduce prolonged outdoor exertion.",
        "Unhealthy": "Everyone may begin to experience adverse health effects. Limit outdoor activity.",
        "Very Unhealthy": "Health alert — everyone should avoid prolonged outdoor exertion.",
        "Hazardous": "Emergency conditions. The entire population is likely to be affected. Stay indoors.",
    }
    return d.get(label, "Monitor air quality conditions.")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def model_exists() -> bool:
    return os.path.exists(MODEL_FILE) and os.path.exists(FEATS_FILE)

def load_model():
    if not model_exists():
        return None, None
    model = joblib.load(MODEL_FILE)
    feats = joblib.load(FEATS_FILE)
    return model, feats

def save_metrics(mae, rmse, r2, target_col, dataset_name=""):
    data = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "target_col": target_col,
        "dataset_name": dataset_name,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_metrics() -> dict:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {}

# ─────────────────────────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    # Try multiple datetime patterns
    for date_col, time_col in [("Date","Time"),("date","time")]:
        if date_col in df.columns and time_col in df.columns:
            df["Datetime"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
            break
    else:
        for col in ["Datetime","datetime","timestamp","Date","date"]:
            if col in df.columns:
                df["Datetime"] = pd.to_datetime(df[col], errors="coerce")
                break
        else:
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return df

def auto_detect_target(df) -> list:
    known = ["AirQualityIndex","aqi_value","AQI","aqi","air_quality_index"]
    candidates = [c for c in known if c in df.columns]
    for col in df.select_dtypes(include=np.number).columns:
        if col not in candidates and col.lower() not in [
            "date","time","hour","dayofweek","month","year","day","minute","second"
        ]:
            vals = df[col].dropna()
            if len(vals) > 0 and vals.min() >= 0:
                candidates.append(col)
    return candidates

def find_target(df):
    for col in ["AirQualityIndex","aqi_value","AQI","aqi","air_quality_index"]:
        if col in df.columns:
            return col
    return None

def engineer_features(df, TARGET):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    keep = numeric_cols + ["Datetime"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.ffill().bfill()
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    df["month"]       = df["Datetime"].dt.month
    df["quarter"]     = df["Datetime"].dt.quarter
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["hour"]        = df["Datetime"].dt.hour
    df["day_of_year"] = df["Datetime"].dt.dayofyear
    if TARGET in df.columns:
        df["target_lag_1"]        = df[TARGET].shift(1)
        df["target_lag_2"]        = df[TARGET].shift(2)
        df["target_lag_24"]       = df[TARGET].shift(24)
        df["target_roll_mean_3"]  = df[TARGET].rolling(3).mean()
        df["target_roll_mean_24"] = df[TARGET].rolling(24).mean()
        df["target_roll_std_3"]   = df[TARGET].rolling(3).std()
    df = df.fillna(df.median(numeric_only=True))
    return df

def train_model(df, TARGET):
    drop_cols = [TARGET, "Datetime", "Date", "Time"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=np.number)
    y = df[TARGET]
    mask = y.notna()
    X, y = X[mask], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.06, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=20, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    return model, X, X_test, y_test, preds, mae, rmse, r2