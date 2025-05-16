# smartqc/quality_monitor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from smartqc.config import CONFIG

def get_user_sensor_limits(df):
    """
    Interactive helper to let user decide:
    - manual mode: define min/max for columns
    - automatic mode: skip manual check
    """
    choice = input("\nDo you want to manually define sensor limits? (y/n): ").lower()
    if choice != 'y':
        print("Skipping manual outlier detection. Will proceed with automatic (if enabled).")
        return {}  # empty dictionary skips manual check

    manual_outlier_ranges = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print("\nNumeric columns available for sensor limit definition:")
    for i, col in enumerate(numeric_cols):
        print(f"{i}: {col}")

    print("\nFor each column, type 'y' to define limits, or 'n' to skip.")

    for col in numeric_cols:
        col_choice = input(f"Define limits for '{col}'? (y/n): ").lower()
        if col_choice == 'y':
            while True:
                try:
                    min_val = float(input(f"Enter minimum acceptable value for '{col}': "))
                    max_val = float(input(f"Enter maximum acceptable value for '{col}': "))
                    if min_val >= max_val:
                        print("Min must be less than Max. Please try again.")
                        continue
                    manual_outlier_ranges[col] = (min_val, max_val)
                    break
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
        else:
            continue

    if manual_outlier_ranges:
        print(f"\nManual sensor limits saved for {len(manual_outlier_ranges)} columns.")
    else:
        print("\nNo manual sensor limits defined. Skipping manual check.")

    return manual_outlier_ranges

class QualityMonitor:
    def __init__(self, df, manual_outlier_ranges=None):
        """
        Quality monitor for anomaly detection and inf values.
        manual_outlier_ranges: dict { column_name: (min, max) }
        """
        self.df = df.copy()
        self.manual_outlier_ranges = manual_outlier_ranges
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        self.report_inf()

        if self.manual_outlier_ranges:
            self.detect_manual_outliers()

        if CONFIG["anomaly_detection"]:
            self.detect_auto_outliers()

    def report_inf(self):
        """
        Report any infinite values found in dataset.
        """
        inf_mask = np.isinf(self.df[self.numeric_cols].values)
        total_inf = np.sum(inf_mask)

        print("Data Quality Report:")
        print(f"   Dataset Shape: {self.df.shape}")
        print(f"   Infinite values detected: {total_inf}")

    def detect_manual_outliers(self):
        """
        Apply manual min-max thresholds for key columns.
        """
        print("Checking manual sensor limits...")

        outlier_flags = pd.Series(0, index=self.df.index)

        for col, (min_val, max_val) in self.manual_outlier_ranges.items():
            if col not in self.df.columns:
                print(f"Column '{col}' not found in data. Skipping.")
                continue

            out_of_bounds = (self.df[col] < min_val) | (self.df[col] > max_val)
            outlier_flags = outlier_flags | out_of_bounds.astype(int)

            print(f"   {col}: flagged {out_of_bounds.sum()} rows outside range [{min_val}, {max_val}]")

        self.df["is_manual_outlier"] = outlier_flags
        print("Manual outlier detection complete.")

    def detect_auto_outliers(self):
        """
        Run Isolation Forest to flag additional unknown outliers.
        Adds a new column: 'is_outlier' (1=outlier, 0=normal).
        """
        print("Running automatic anomaly detection (Isolation Forest)...")
        X = self.df[self.numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if X.shape[0] < 10:
            print("Not enough rows for Isolation Forest. Skipping.")
            return

        clf = IsolationForest(contamination=0.01, random_state=CONFIG["random_state"])
        preds = clf.fit_predict(X)
        is_outlier = np.where(preds == -1, 1, 0)

        outlier_mask = pd.Series(0, index=self.df.index)
        outlier_mask[X.index] = is_outlier
        self.df["is_outlier"] = outlier_mask

        print(f"Automatic outliers flagged in 'is_outlier' column (1=outlier, 0=normal).")
        print(self.df[["is_outlier"] + list(self.numeric_cols)[:5]].head())

    def get_data(self):
        """
        Return dataframe with outlier columns.
        """
        return self.df


