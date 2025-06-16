# core/feature_engineer.py

import pandas as pd
import numpy as np
import os
import json
from db.insert_features import log_feature_formula, log_time_batch_cycle  # 🔄 DB logging functions

MEMORY_PATH = "data/feature_memory.json"

class DefectEngineer:
    def __init__(self, df, target_columns, upload_id, correlation_threshold=0.2, top_k=20, safe_division_eps=1e-6):
        self.manual_features = []
        self.df = df.copy()
        self.top_k = top_k
        self.upload_id = upload_id
        self.correlation_threshold = correlation_threshold
        self.safe_division_eps = safe_division_eps
        self.target_columns = target_columns
        self.numeric_cols = []

    def _generate_features(self):
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = [col for col in self.numeric_cols if col not in self.target_columns]

        new_features = {}
        formula_map = {}

        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i+1:]:
                f1 = f"{col1}_x_{col2}"
                f2 = f"{col1}_div_{col2}"
                f3 = f"{col1}_minus_{col2}"

                new_features[f1] = self.df[col1] * self.df[col2]
                new_features[f2] = self.df[col1] / (self.df[col2] + self.safe_division_eps)
                new_features[f3] = self.df[col1] - self.df[col2]

                formula_map[f1] = f"{col1} * {col2}"
                formula_map[f2] = f"{col1} / ({col2} + eps)"
                formula_map[f3] = f"{col1} - {col2}"

        self.auto_formula_map = formula_map
        return pd.DataFrame(new_features)

    def _score_features(self, features_df):
        scores = {}
        for feature in features_df.columns:
            is_collinear = False
            for manual in self.manual_features:
                if manual in self.df.columns:
                    corr = np.corrcoef(features_df[feature], self.df[manual])[0, 1]
                    if abs(corr) > 0.9:
                        is_collinear = True
                        break
            if is_collinear:
                continue

            correlations = []
            for target in self.target_columns:
                if target not in self.df.columns:
                    continue
                corr = np.corrcoef(features_df[feature], self.df[target])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            max_corr = max(correlations) if correlations else 0
            if max_corr >= self.correlation_threshold:
                scores[feature] = max_corr

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _confirm_and_select_features(self, features_df, ranked_features):
        top_features = [name for name, _ in ranked_features[:self.top_k]]

        print(f"\n📊 Top {self.top_k} auto-selected features:")
        for i, name in enumerate(top_features):
            print(f"{i+1:2d}. {name}")

        approve = input("\n✅ Keep all of these features? (y/n): ").strip().lower()
        if approve == "y":
            return features_df[top_features], top_features

        print("\nSelect features by their number (comma-separated):")
        selected = input("Your selection: ").strip()
        selected_indexes = [int(x.strip()) - 1 for x in selected.split(",") if x.strip().isdigit()]
        selected_features = [top_features[i] for i in selected_indexes if 0 <= i < len(top_features)]

        print(f"✅ Keeping {len(selected_features)} selected features.")
        return features_df[selected_features], selected_features

    def _manual_formula_input(self):
        print("\n➕ Manually create new features (optional).")
        print("Use column numbers for math expressions. Type 'done' to finish.\n")

        col_ref = {i: name for i, name in enumerate(self.numeric_cols)}
        for i, col in col_ref.items():
            print(f"{i}: {col}")
        print()

        while True:
            formula = input("Enter formula (or 'done'): ").strip()
            if formula.lower() == 'done':
                break
            try:
                original = formula
                for i, col in col_ref.items():
                    formula = formula.replace(str(i), f"`{col}`")
                new_col_name = original.replace("*", "_mul_").replace("/", "_div_").replace("+", "_plus_").replace("-", "_minus_").replace(" ", "")
                self.df[new_col_name] = self.df.eval(formula)
                self.manual_features.append(new_col_name)
                log_feature_formula(self.upload_id, new_col_name, original, feature_type="manual")
                print(f"✅ Added: {new_col_name}")
            except Exception as e:
                print(f"❌ Error parsing formula: {e}")

    def _filter_multicollinear(self, feature_names):
        corr_matrix = self.df[feature_names].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
        print(f"🧹 Dropping multicollinear features: {drop_cols}")
        return [f for f in feature_names if f not in drop_cols]

    def _analyze_time_series(self):
        print("🧠 Performing time series batch analysis...")

        ts = pd.to_datetime(self.df['timestamp']).sort_values()
        duration = (ts.max() - ts.min()).total_seconds() / 60  # in minutes

        log_time_batch_cycle(
            upload_id=self.upload_id,
            start_timestamp=ts.min(),
            end_timestamp=ts.max(),
            batch_duration_min=int(duration)
        )

        print(f"📦 Detected batch duration: {int(duration)} minutes.")

    def run(self):
        print("\n🔧 Generating synthetic features...")
        synthetic_df = self._generate_features()

        print("\n🔎 Scoring features based on correlation...")
        ranked = self._score_features(synthetic_df)

        selected_df, selected_names = self._confirm_and_select_features(synthetic_df, ranked)

        self.df = pd.concat([self.df, selected_df], axis=1)
        selected_names = self._filter_multicollinear(selected_names)

        to_drop = [col for col in selected_df.columns if col not in selected_names]
        self.df.drop(columns=to_drop, inplace=True)

        for feat in selected_names:
            log_feature_formula(self.upload_id, feat, self.auto_formula_map.get(feat, "n/a"), feature_type="auto")

        if "timestamp" in self.df.columns:
            self._analyze_time_series()

        self._manual_formula_input()

        return self.df, selected_names + self.manual_features

# === Integration wrapper ===

def save_feature_selection(file_key, features):
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            memory = json.load(f)
    else:
        memory = {}

    memory[file_key] = features
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)
    print(f"💾 Saved selected features to {MEMORY_PATH}")

def run_feature_engineering(df, file_key, target_columns, upload_id):
    engineer = DefectEngineer(df, target_columns, upload_id)
    engineered_df, feature_names = engineer.run()
    save_feature_selection(file_key, feature_names)
    return engineered_df
