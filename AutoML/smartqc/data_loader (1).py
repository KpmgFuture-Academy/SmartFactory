# smartqc/data_loader.py

import os
import pandas as pd
from smartqc.config import CONFIG

class DataLoader:
    def __init__(self, filepath):
        """
        Load a CSV file into pandas dataframe. 
        If large dataset → auto-splits into chunks.
        """
        self.filepath = filepath
        self.df = None
        self.load_data()

    def load_data(self):
        """
        Handles batch reading for large datasets based on config["batch_size"].
        """
        batch_size = CONFIG["batch_size"]

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Get file size in rows (estimate by reading small sample)
        row_sample = pd.read_csv(self.filepath, nrows=5)
        try:
            total_cols = len(row_sample.columns)
            total_rows = sum(1 for _ in open(self.filepath)) - 1  # minus header
        except Exception:
            total_rows = None

        if total_rows and total_rows > batch_size:
            print(f"Large file detected ({total_rows} rows). Using batch chunking mode...")
            chunks = []
            for chunk in pd.read_csv(self.filepath, chunksize=batch_size):
                chunks.append(chunk)
            self.df = pd.concat(chunks, ignore_index=True)
        else:
            self.df = pd.read_csv(self.filepath)

        if CONFIG["verbose"]:
            print(f"Data loaded → shape: {self.df.shape}")

    def preview(self, n=5):
        """
        Show first n rows of dataset.
        """
        if self.df is not None:
            print(self.df.head(n))
        else:
            print("No data loaded yet.")
