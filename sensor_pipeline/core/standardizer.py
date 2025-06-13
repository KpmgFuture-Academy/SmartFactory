import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

SAVE_DIR = "data/scaler_objects"

def ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)

def standardize_numeric(df, file_key):
    """
    Applies StandardScaler to numeric columns and saves the scaler for reuse.
    """
    ensure_dir()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    scaler_path = os.path.join(SAVE_DIR, f"{file_key.replace('/', '_')}_scaler.pkl")
    joblib.dump({"columns": numeric_cols, "scaler": scaler}, scaler_path)
    print(f"⚖️ Standardized and saved scaler → {scaler_path}")
    return df

def apply_standardization(df, file_key):
    """
    Loads scaler and applies to test data.
    """
    scaler_path = os.path.join(SAVE_DIR, f"{file_key.replace('/', '_')}_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No scaler saved for {file_key}")

    obj = joblib.load(scaler_path)
    numeric_cols = obj["columns"]
    scaler = obj["scaler"]

    df[numeric_cols] = scaler.transform(df[numeric_cols])
    print("✅ Applied saved scaler to test data")
    return df
