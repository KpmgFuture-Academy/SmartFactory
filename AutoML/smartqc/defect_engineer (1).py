# smartqc/defect_engineer.py

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display

class DefectEngineer:
    def __init__(self, df, correlation_threshold=0.2, safe_division_eps=1e-6):
        """
        Universal feature engineering + target selector.
        """
        self.df = df.copy()
        self.correlation_threshold = correlation_threshold
        self.safe_division_eps = safe_division_eps
        self.target_columns = []

    def _select_target_columns(self):
        """
        Interactive multi-select dropdown for target column(s).
        """
        print("Select your target column(s) for prediction:")
        cols = self.df.columns.tolist()

        dropdown = widgets.SelectMultiple(
            options=cols,
            description='Targets:',
            rows=6,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )

        confirm_btn = widgets.Button(description="Confirm Selection")

        output = widgets.Output()

        def on_confirm(b):
            self.target_columns = list(dropdown.value)
            with output:
                output.clear_output()
                print(f"Selected Target(s): {self.target_columns}")

        confirm_btn.on_click(on_confirm)
        display(dropdown, confirm_btn, output)

    def _generate_features(self, input_cols):
        """
        Generate pairwise feature combinations: × / −
        """
        new_features = {}

        for i, col1 in enumerate(input_cols):
            for col2 in input_cols[i+1:]:
                # Multiplication
                new_features[f"{col1}_x_{col2}"] = self.df[col1] * self.df[col2]

                # Safe Division
                new_features[f"{col1}_div_{col2}"] = self.df[col1] / (self.df[col2] + self.safe_division_eps)

                # Subtraction
                new_features[f"{col1}_minus_{col2}"] = self.df[col1] - self.df[col2]

        return pd.DataFrame(new_features)

    def _filter_by_correlation(self, features_df):
        """
        Keep only features with correlation to target(s) above threshold.
        """
        keep_features = []

        for feature in features_df.columns:
            correlations = []
            for target in self.target_columns:
                if target not in self.df.columns:
                    continue
                corr = np.corrcoef(features_df[feature], self.df[target])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            max_corr = max(correlations) if correlations else 0
            if max_corr >= self.correlation_threshold:
                keep_features.append(feature)

        return features_df[keep_features]

    def run(self):
        """
        Full pipeline:
        - Ask for target(s)
        - Generate features
        - Filter by correlation
        - Return updated dataframe + target list
        """
        # Temporary manual input fallback
        print("Dropdown confirm not working. Fallback to manual input.")
        print(f"Columns available: {list(self.df.columns)}")
        raw_input = input("Enter your target column(s), comma-separated: ")
        self.target_columns = [col.strip() for col in raw_input.split(",") if col.strip() in self.df.columns]
        print(f"Selected Target(s): {self.target_columns}")

        print("\nGenerating synthetic features...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in self.target_columns]

        print(f"   Using {len(numeric_cols)} numeric input columns.")
        new_features = self._generate_features(numeric_cols)

        print(f"   Created {new_features.shape[1]} synthetic features.")
        print("Filtering based on correlation to target(s)...")

        filtered = self._filter_by_correlation(new_features)

        if filtered.empty:
            print("No synthetic features passed correlation threshold.")
        else:
            print(f"{filtered.shape[1]} features kept after filtering.")

        # Append to original dataframe
        final_df = pd.concat([self.df, filtered], axis=1)
        print(f"Final dataset shape: {final_df.shape}")

        return final_df, self.target_columns
