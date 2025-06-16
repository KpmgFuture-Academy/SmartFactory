# db/insert_targets.py

import psycopg2
import pandas as pd
import json
from config import DB_CONFIG

def insert_target_labels(df: pd.DataFrame, upload_id: str, target_col: str):
    """
    Insert machine failure boolean and JSON multi-failure labels into input_target.
    Assumes df contains the main binary target and sub-failure mode columns.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Identify failure mode columns (multi-labels)
    sub_targets = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    available = [col for col in sub_targets if col in df.columns]

    for _, row in df.iterrows():
        machine_failure = bool(row[target_col])
        failure_modes = {col: int(row[col]) for col in available}
        cursor.execute("""
            INSERT INTO input_target (upload_id, machine_failure, failure_modes)
            VALUES (%s, %s, %s)
        """, (upload_id, machine_failure, json.dumps(failure_modes)))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into input_target.")
