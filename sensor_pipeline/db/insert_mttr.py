import pandas as pd
from db.connection import get_connection
from datetime import datetime
import os

def insert_mttr_log(file_key=None, source="S3"):
    """
    Insert MTTR (mean time to repair) logs into mttr_log table.
    Currently uses synthetic data from a local CSV.
    
    Args:
        file_key (str): Used for logging/traceability
        source (str): 'S3', 'manual', etc.
    """
    # You can replace this with actual S3 download logic later
    csv_path = os.path.join("data", "mttr_log.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["failure_timestamp", "resolved_timestamp"])

    conn, cur = get_connection()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO mttr_log (sensor_id, failure_timestamp, resolved_timestamp, source)
            VALUES (%s, %s, %s, %s)
        """, (
            row["sensor_id"],
            row["failure_timestamp"],
            row["resolved_timestamp"],
            source
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into mttr_log from {source}")
