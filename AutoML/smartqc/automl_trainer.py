# smartqc/automl_trainer.py

import os
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

class AutoMLTrainer:
    def __init__(self, X, y, task_type=None, max_runtime_secs=300, max_models=10, save_dir="models"):
        """
        Automatically handles classification and regression tasks using H2O AutoML.

        Parameters:
        - X: pandas DataFrame (features)
        - y: pandas Series or DataFrame (target)
        - task_type: Optional[str] ("classification" or "regression"); inferred if None
        - max_runtime_secs: Total time budget
        - max_models: Max number of models to train
        - save_dir: Directory to save model
        """
        self.X = X
        self.y = y if isinstance(y, pd.Series) else y.iloc[:, 0]
        self.target_col = self.y.name
        self.task_type = task_type or self._infer_task_type()
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.save_dir = save_dir

        self.h2o_df = None
        self.model = None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        h2o.init(max_mem_size="4G", nthreads=-1)

    def _infer_task_type(self):
        """Infers task type from y"""
        unique_vals = self.y.dropna().unique()
        if self.y.dtype == "object" or len(unique_vals) <= 20 and all(np.equal(np.mod(unique_vals, 1), 0)):
            return "classification"
        return "regression"

    def prepare_data(self):
        """
        Converts X and y to H2OFrame, handles factor conversion.
        """
        print(f"\n[INFO] Preparing data for {self.task_type.upper()} task...")

        # Cast y depending on task
        y_cleaned = self.y.astype("int" if self.task_type == "classification" else "float")

        # Combine for H2O
        combined_df = pd.concat([self.X.reset_index(drop=True), y_cleaned.reset_index(drop=True)], axis=1)
        self.h2o_df = h2o.H2OFrame(combined_df)

        if self.task_type == "classification":
            self.h2o_df[self.target_col] = self.h2o_df[self.target_col].asfactor()

        print(f"[INFO] H2OFrame ready: {self.h2o_df.nrows} rows, {self.h2o_df.ncols} columns.")

    def run_automl(self):
        """
        Trains H2O AutoML model.
        """
        sort_metric = "AUC" if self.task_type == "classification" else "RMSE"

        print(f"\n[INFO] Running H2O AutoML ({self.task_type}) with sort_metric='{sort_metric}'...")
        aml = H2OAutoML(
            max_runtime_secs=self.max_runtime_secs,
            max_models=self.max_models,
            seed=42,
            sort_metric=sort_metric,
            verbosity="info"
        )

        aml.train(y=self.target_col, training_frame=self.h2o_df)
        self.model = aml.leader
        print(f"[SUCCESS] Training complete. Best model: {self.model.algo}")
        return aml

    def save_model(self):
        """
        Saves model to disk.
        """
        model_path = h2o.save_model(model=self.model, path=self.save_dir, force=True)
        print(f"[INFO] Model saved at: {model_path}")
        return model_path

    def run(self):
        """
        Full pipeline: Prepare → Train → Save → Print Metrics → Return model + leaderboard + path
        """
        self.prepare_data()
        aml = self.run_automl()
        model_path = self.save_model()

        # Print full H2O leaderboard
        print("\n[📊] Full H2O Leaderboard:")
        print(aml.leaderboard)

        # Print evaluation metrics
        print(f"\n[✅] Evaluating best model performance on training data...")
        perf = self.model.model_performance(self.h2o_df)

        if self.task_type == "classification":
            try:
                acc = perf.accuracy()[0][1]
            except:
                acc = "N/A"
            print(f"→ Accuracy: {acc}")
            print(f"→ AUC: {perf.auc():.4f}")
            print(f"→ Logloss: {perf.logloss():.4f}")
        else:  # regression
            print(f"→ RMSE: {perf.rmse():.4f}")
            print(f"→ MAE: {perf.mae():.4f}")
            print(f"→ R²: {perf.r2():.4f}")

        leaderboard = aml.leaderboard.as_data_frame()
        return self.model, leaderboard, model_path

    def plot_shap_summary(self):
        """
        Plots variable importance if available.
        """
        try:
            print(f"[INFO] Showing variable importance...")
            self.model.varimp_plot()
        except Exception as e:
            print(f"[WARNING] Could not plot variable importance: {e}")
