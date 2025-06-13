from core.s3_handler import download_file
from core.missing_handler import clean_missing
from core.target_selector import ask_target_column
from core.feature_engineer import run_feature_engineering
from core.sensor_limiter import define_sensor_limits
from core.standardizer import standardize_numeric
from core.time_batch import (
    get_timestamp_column,
    convert_and_validate_timestamp,
    get_retrain_interval,
    log_time_series_result
)

import pandas as pd
from db.upload_log import insert_upload
from db.insert_features import insert_sensor_features
from db.insert_targets import insert_target_labels
from db.insert_model import insert_model_metadata
from model.automl_runner import train_and_log_model

if __name__ == "__main__":
    # 📥 Step 1: Download from S3
    s3_key = "ai4i2020/Timestamped_Dataset.csv"
    path = download_file(s3_key)
    df = pd.read_csv(path)

    # 🧹 Step 2: Clean missing values
    df = clean_missing(df)

    # 🎯 Step 3: Select target column
    target_col = ask_target_column(df, file_key=s3_key)

    # ✅ 🧾 Register the upload EARLY to get upload_id
    upload_id = insert_upload(
        file_key=s3_key,
        has_target=(target_col is not None),
        task_type="binary_classification"
    )

    # 🛠️ Step 4: Feature engineering
    if target_col is not None:
        df = run_feature_engineering(df, file_key=s3_key, target_columns=[target_col], upload_id=upload_id)
    else:
        print("⚠️ Skipping feature engineering due to no target selection.")

    # 📊 Step 5: Define sensor limits
    sensor_limits = define_sensor_limits(df, file_key=s3_key)

    # ⚙️ Step 6: Standardize features
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_scaled = standardize_numeric(X, file_key=s3_key)
    df = pd.concat([X_scaled, y], axis=1)

    # 📈 Step 7: Time Series Analysis (only if target exists)
    if target_col:
        try:
            from core.time_batch import (
                get_timestamp_column,
                convert_and_validate_timestamp,
                predict_next_failure_timestamp
            )
            from db.insert_features import log_time_batch_cycle

            timestamp_col = get_timestamp_column(df)
            df = convert_and_validate_timestamp(df, timestamp_col)

            # Predict next failure timestamp
            predicted_ts, interval_minutes = predict_next_failure_timestamp(
                df=df,
                timestamp_col=timestamp_col,
                target_col=target_col,
                method="arima"  # internally tagged
            )

            print(f"🧠 Next failure predicted at: {predicted_ts}")
            print(f"⏱️ Time to next failure: {interval_minutes} minutes")

            # Format predicted date into YYYYMMDD
            detected_from = predicted_ts.strftime("%Y%m%d")

            # Log into DB
            log_time_batch_cycle(
                upload_id=upload_id,
                batch_interval_minutes=interval_minutes,
                detected_from=detected_from
            )

        except Exception as e:
            print(f"❌ Time series analysis failed: {e}")
    else:
        print("⏭️ Time series analysis skipped (no target column).")

    # 🤖 Step 8: Train AutoML and log to MLflow
    result = train_and_log_model(df, target_col=target_col, upload_id=upload_id)

    print("\n🏆 MLflow Run ID:", result["run_id"])
    print("✅ Accuracy:", result["accuracy"])
    print("✅ F1 Score:", result["f1_score"])
    print("✅ AUC Score:", result["auc_score"])
    print("\n📊 Leaderboard (Top 5):")
    print(result["leaderboard"].head(5))

    # 💾 Step 9: Save model metadata to DB
    model_id = insert_model_metadata(
        run_id=result["run_id"],
        upload_id=upload_id,
        accuracy=result["accuracy"],
        f1_score=result["f1_score"],
        auc_score=result["auc_score"]
    )
    print("🧠 Model ID saved to DB:", model_id)

    # 📡 Step 10: Insert features
    features_only = df.drop(columns=[target_col])
    insert_sensor_features(features_only, upload_id=upload_id)

    # 🎯 Step 11: Insert targets
    insert_target_labels(df, upload_id=upload_id, target_col=target_col)


# 📥 Step 12: Insert production logs (fake or real)
from db.insert_production import insert_production_log
insert_production_log(df=features_only, upload_id=upload_id)

# 🛠️ Step 13: Insert MTTR logs (if available)
from db.insert_mttr import insert_mttr_log
insert_mttr_log(file_key=s3_key)

# 📊 Step 14: Generate and insert dashboard metrics
from db.insert_dashboard import generate_dashboard_metrics
generate_dashboard_metrics(upload_id=upload_id)
