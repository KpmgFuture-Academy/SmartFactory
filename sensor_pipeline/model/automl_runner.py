import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
import os

# Initialize H2O and disable progress bars
h2o.init(max_mem_size="4G")
h2o.no_progress()  # ✅ Hides training progress bar

def train_and_log_model(df, target_col, upload_id, max_runtime=300):
    """
    Trains H2O AutoML model, logs to MLflow, and returns the best model info.
    """
    print("📊 Starting AutoML training...")

    # Split train/val (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert to H2OFrame
    train_h2o = h2o.H2OFrame(train_df)
    val_h2o = h2o.H2OFrame(val_df)

    # Set target + predictors
    y = target_col
    x = [col for col in train_h2o.columns if col != y]

    train_h2o[y] = train_h2o[y].asfactor()
    val_h2o[y] = val_h2o[y].asfactor()

    # Train AutoML with less verbose output
    aml = H2OAutoML(
        max_runtime_secs=max_runtime,
        seed=42,
        nfolds=5,
        verbosity="warn"  # ✅ Show only warnings
    )
    aml.train(x=x, y=y, training_frame=train_h2o, validation_frame=val_h2o)

    best_model = aml.leader
    leaderboard = aml.leaderboard.as_data_frame()
    model_metrics = best_model.model_performance(val_h2o)

    acc = model_metrics.accuracy()[0][1]
    f1 = model_metrics.F1()[0][1]
    auc = model_metrics.auc()

    print(f"✅ Model trained — Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    model_id = leaderboard.loc[0, "model_id"]
    shap_df = best_model.varimp(use_pandas=True)

    mlflow.set_experiment(f"smartqc_sensor_upload_{upload_id}")
    with mlflow.start_run() as run:
        mlflow.log_param("target", y)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_score", auc)

        run_id = run.info.run_id
        print(f"📦 Logged to MLflow → Run ID: {run_id}")

    return {
        "run_id": run_id,
        "accuracy": acc,
        "f1_score": f1,
        "auc_score": auc,
        "leaderboard": leaderboard,
        "shap_values": shap_df,
        "model_id": model_id,
    }


    