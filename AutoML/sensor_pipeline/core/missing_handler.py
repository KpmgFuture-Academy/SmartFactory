import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def drop_sparse_cols(df, threshold=0.5):
    """
    Drops columns with more than `threshold` % missing.
    """
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    df_cleaned = df.drop(columns=to_drop)
    print(f"🗑 Dropped columns with >{int(threshold*100)}% missing: {to_drop}")
    return df_cleaned

def apply_knn_imputer(df):
    """
    Fills missing values using KNN for numeric columns only.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print(f"🧠 KNN Imputed numeric columns: {list(numeric_cols)}")
    return df

def clean_missing(df):
    """
    Full missing value cleaning: drop sparse → KNN impute rest.
    """
    df = drop_sparse_cols(df)
    df = apply_knn_imputer(df)
    return df
