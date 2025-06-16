# core/time_batch.py

import os
import json
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from db.insert_features import log_time_batch_cycle

def get_timestamp_column(df):
    path = "data/ts_column_memory.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            col_name = json.load(f).get("timestamp")
            if col_name in df.columns:
                print(f"⏰ Reusing saved timestamp column: '{col_name}'")
                return col_name

    # Ask user
    print("\n📅 Available columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    choice = input("⏱ Select timestamp column by number or name: ").strip()
    col_name = df.columns[int(choice)] if choice.isdigit() else choice

    # Save to memory
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"timestamp": col_name}, f)

    return col_name

def convert_and_validate_timestamp(df, timestamp_col):
    print(f"⏰ Using '{timestamp_col}' as timestamp column.")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df

def get_retrain_interval(df, timestamp_col, target_col):
    ts = df.set_index(timestamp_col)[target_col].dropna()

    if len(ts) < 20:
        print("⚠️ Not enough data points for time series analysis.")
        return 10

    model = ARIMA(ts, order=(2, 1, 2)).fit()
    forecast = model.forecast(steps=10)

    threshold = ts.quantile(0.9)
    for i, val in enumerate(forecast):
        if val > threshold:
            return i + 1  # steps until anomaly
    return 10

from datetime import datetime
from db.insert_features import log_time_batch_cycle

def log_time_series_result(upload_id, interval, detected_from="arima"):
    timestamp = datetime.now()
    log_time_batch_cycle(
        interval_minutes=interval,
        detected_from=detected_from,
        timestamp=timestamp,
        upload_id=upload_id
    )



def predict_next_failure_timestamp(df, timestamp_col, target_col, method="arima"):
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA

    df = df[[timestamp_col, target_col]].copy()
    df = df.sort_values(timestamp_col)
    df = df.dropna()

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)

    y = df[target_col]

    # Fit ARIMA (simple default model for now)
    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast the next 100 future steps
    forecast = model_fit.forecast(steps=100)

    # Find first future step with likely failure
    for i, val in enumerate(forecast):
        if val >= 0.5:  # Adjust threshold as needed
            predicted_timedelta = pd.Timedelta(minutes=i)
            break
    else:
        predicted_timedelta = pd.Timedelta(minutes=100)  # fallback if no failure predicted

    last_timestamp = df.index[-1]
    predicted_ts = last_timestamp + predicted_timedelta
    interval_minutes = int(predicted_timedelta.total_seconds() // 60)

    return predicted_ts, interval_minutes
