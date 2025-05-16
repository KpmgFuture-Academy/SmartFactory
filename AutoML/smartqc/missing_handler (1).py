# smartqc/missing_handler.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from smartqc.config import CONFIG

class MissingHandler:
    def __init__(self, df):
        """
        Handles missing values for a pandas DataFrame.
        """
        self.df = df.copy()
        self.original_shape = self.df.shape
        self.report()

    def report(self):
        """
        Prints audit of missing data before cleaning.
        """
        total_cells = np.prod(self.df.shape)
        total_missing = self.df.isnull().sum().sum()
        percent_missing = (total_missing / total_cells) * 100

        print("\nMissing Value Audit:")
        print(f"   Dataset Shape: {self.df.shape}")
        print(f"   Total Missing Cells: {total_missing}")
        print(f"   Missing %: {percent_missing:.2f}%")

    def fix(self):
        """
        Fixes missing values based on strategy in CONFIG.
        """
        strategy = CONFIG["inf_handling"]

        # Drop sparse columns (>50% missing)
        threshold = 0.5
        missing_ratio = self.df.isnull().mean()
        to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        if to_drop:
            print(f"\nDropping {len(to_drop)} columns with >{int(threshold * 100)}% missing values...")
            self.df.drop(columns=to_drop, inplace=True)
        else:
            print("No columns dropped for missing %.")

        # Impute or Drop based on strategy
        if strategy == "knn":
            print("Using KNNImputer (numeric columns only)...")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns

            imputer = KNNImputer(n_neighbors=5)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

            print(f"KNNImputer done. Non-numeric columns untouched: {list(non_numeric_cols)}")

        elif strategy == "drop":
            print("Dropping rows with any missing values...")
            self.df.dropna(inplace=True)

        else:
            raise ValueError(f"Unknown strategy in CONFIG: {strategy}")

        print(f"\nFinal dataset shape after missing handling: {self.df.shape}")
        return self.df
