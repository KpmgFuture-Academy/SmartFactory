# smartqc/missing_handler.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from smartqc.config import CONFIG

class MissingHandler:
    def __init__(self, df):
        """
        Takes a pandas dataframe and checks + handles missing values.
        """
        self.df = df.copy()
        self.report()

    def report(self):
        """
        Prints basic missing value audit report.
        """
        total_cells = np.prod(self.df.shape)
        total_missing = self.df.isnull().sum().sum()
        percent_missing = (total_missing / total_cells) * 100

        print(" Missing Value Audit:")
        print(f"   Dataset Shape: {self.df.shape}")
        print(f"   Total Missing Cells: {total_missing}")
        print(f"   Missing %: {percent_missing:.2f}%")

    def fix(self):
        """
        Applies missing value fixing based on config setting.
        """
        strategy = CONFIG["inf_handling"]

        if strategy == "knn":
            print("Using KNNImputer to fill missing values...")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns

            # Only impute numeric columns
            imputer = KNNImputer(n_neighbors=5)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

            print(f"KNNImputer complete. Non-numeric columns untouched: {list(non_numeric_cols)}")

        elif strategy == "drop":
            print("Dropping rows with any missing values...")
            self.df.dropna(inplace=True)

        else:
            raise ValueError(f"Unknown inf_handling strategy: {strategy}")

        print(f"New dataset shape after missing value handling: {self.df.shape}")
        return self.df
