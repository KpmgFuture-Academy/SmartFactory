# smartqc/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, df, target_columns, corr_threshold=0.95):
        """
        Automated preprocessing pipeline with editable drop list:
        - Drop ID-like and categorical columns
        - Remove multicollinear features
        - Scale numeric columns
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.target_columns = target_columns
        self.corr_threshold = corr_threshold
        self.feature_names = []

    def drop_id_and_categorical_columns(self):
        """
        Auto-detect and allow manual edit of columns to drop.
        """
        print("Auto-detecting ID-like and categorical columns...")

        id_like_cols = [col for col in self.df.columns if 'id' in col.lower() or col.lower() in ['udi', 'product id']]
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        proposed_drop = list(set(id_like_cols + categorical_cols))
        proposed_drop = [col for col in proposed_drop if col not in self.target_columns]

        print(f"\n🧾 Proposed columns to drop: {proposed_drop}")
        resp = input("Would you like to edit this list? (Y/N): ").strip().lower()

        if resp == 'y':
            manual_input = input("Enter final column names to drop (comma-separated): ")
            final_drop = [col.strip() for col in manual_input.split(',') if col.strip() in self.df.columns]
            print(f"Using manual drop list: {final_drop}")
        else:
            final_drop = proposed_drop
            print(f"Using auto drop list.")

        self.df.drop(columns=final_drop, inplace=True, errors='ignore')

    def remove_multicollinear_features(self):
        """
        Drop one feature from each highly correlated pair (above threshold).
        """
        print("\nRemoving multicollinear features...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in self.target_columns]

        corr_matrix = self.df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        self.df.drop(columns=to_drop, inplace=True, errors='ignore')

        print(f"   Dropped {len(to_drop)} highly correlated features.")

    def scale_numeric_features(self):
        """
        Apply StandardScaler to all numeric input features.
        """
        print("\nScaling numeric features...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        input_cols = [col for col in numeric_cols if col not in self.target_columns]

        scaler = StandardScaler()
        self.df[input_cols] = scaler.fit_transform(self.df[input_cols])

        self.feature_names = input_cols
        print(f"   Scaled {len(input_cols)} features.")

    def run(self):
        """
        Run full preprocessing pipeline.
        Returns:
            X: scaled feature DataFrame
            y: target column(s)
            feature_names: list of feature column names
        """
        self.drop_id_and_categorical_columns()
        self.remove_multicollinear_features()
        self.scale_numeric_features()

        X = self.df[self.feature_names]
        y = self.df[self.target_columns]

        print(f"\nFinal shapes → X: {X.shape}, y: {y.shape}")
        return X, y, self.feature_names
