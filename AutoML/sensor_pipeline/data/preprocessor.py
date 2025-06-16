import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from config import CONFIG

def load_csv(filepath):
    """
    Loads CSV in batch mode if large. Applies missing handling strategy.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    batch_size = CONFIG.get("batch_size", 1000)
    row_count = sum(1 for _ in open(filepath)) - 1

    if row_count > batch_size:
        print(f"⚠️ Large file detected ({row_count} rows). Using chunked mode...")
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=batch_size):
            cleaned = fix_missing(chunk)
            chunks.append(cleaned)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(filepath)
        df = fix_missing(df)

    print(f"✅ Final cleaned shape: {df.shape}")
    return df

def fix_missing(df):
    """
    Applies missing value strategy: 'drop' or 'knn'
    """
    strategy = CONFIG.get("inf_handling", "drop")

    # Drop columns > 50% missing
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    if to_drop:
        print(f"🗑 Dropping sparse columns: {to_drop}")
        df = df.drop(columns=to_drop)

    if strategy == "knn":
        print("🧠 Imputing missing values using KNN...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif strategy == "drop":
        print("🧼 Dropping rows with any missing values...")
        df = df.dropna()
    else:
        raise ValueError(f"❌ Unknown missing strategy: {strategy}")

    return df

def extract_features(df):
    exclude = ["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    return df.drop(columns=[col for col in exclude if col in df.columns]).to_dict(orient="records")

def extract_targets(df):
    if "Machine failure" not in df.columns:
        return None, None
    binary = df["Machine failure"].astype(bool).tolist()
    multi_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    multi = df[multi_cols].to_dict(orient="records")
    return binary, multi

def generate_timestamp(n_rows, start="2025-05-20 09:00:00", interval="3min"):
    return pd.date_range(start=start, periods=n_rows, freq=interval).tolist()
