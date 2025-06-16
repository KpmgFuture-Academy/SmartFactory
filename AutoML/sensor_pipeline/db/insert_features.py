import pandas as pd
import json
from db.connection import get_connection
from datetime import datetime

# ✅ Insert preprocessed sensor features into input_sensor_data
def insert_sensor_features(df, upload_id):
    conn, cur = get_connection()

    for _, row in df.iterrows():
        # Extract JSON-serializable features, excluding timestamp
        feature_json = {
            col: (val.isoformat() if isinstance(val, pd.Timestamp) else val)
            for col, val in row.items()
            if col.lower() != "timestamp"
        }

        # Parse timestamp
        timestamp = row.get("Timestamp", None)
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        cur.execute("""
            INSERT INTO input_sensor_data (upload_id, timestamp, features)
            VALUES (%s, %s, %s)
        """, (upload_id, timestamp, json.dumps(feature_json)))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into input_sensor_data.")


# ✅ Log feature formulas to feat_formula (auto-generated or manual)
def log_feature_formula(upload_id, feature_name, formula, feature_type=None):
    conn, cur = get_connection()

    cur.execute("""
        INSERT INTO feat_formula (source_upload_id, feature_name, formula)
        VALUES (%s, %s, %s)
    """, (upload_id, feature_name, formula))

    conn.commit()
    cur.close()
    conn.close()
    print(f"📌 Logged formula for feature: {feature_name}")


# ✅ Log predicted time batch cycle (forecasted interval) into DB
def log_time_batch_cycle(upload_id, batch_interval_minutes, detected_from):
    conn, cur = get_connection()

    cur.execute("""
        INSERT INTO time_batch_cycle (upload_id, batch_interval_minutes, detected_from)
        VALUES (%s, %s, %s)
    """, (upload_id, batch_interval_minutes, detected_from))

    conn.commit()
    cur.close()
    conn.close()
    print(f"🕒 Logged time batch: {batch_interval_minutes} mins from {detected_from}")
