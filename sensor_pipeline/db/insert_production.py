import psycopg2
from config import DB_CONFIG
from datetime import datetime

def insert_production_log(df, upload_id: str):
    """
    Insert rows into the production_log table based on input df.
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    for idx, row in df.iterrows():
        # Try to get a real timestamp, else use now()
        timestamp = (
            row.get("timestamp")
            or row.get("Timestamp")
            or datetime.now()
        )

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                timestamp = datetime.now()

        product_id = f"{upload_id}_P{idx}"
        sensor_id = "S1"  # Or replace with logic later
        anomaly = bool(row.get("machine_failure", False))
        is_auto = True  # You can update logic later

        cursor.execute("""
            INSERT INTO production_log (
                timestamp, product_id, sensor_id, anomaly_detected, is_automated
            ) VALUES (%s, %s, %s, %s, %s)
        """, (timestamp, product_id, sensor_id, anomaly, is_auto))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into production_log with timestamps.")
