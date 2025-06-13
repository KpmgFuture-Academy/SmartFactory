# db/insert_model.py

import psycopg2
from config import DB_CONFIG

def insert_model_metadata(upload_id, run_id, accuracy, f1_score, auc_score):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_registry (
            mlflow_run_id,
            trained_on_upload_id,
            accuracy,
            f1_score,
            auc_score,
            is_active
        ) VALUES (%s, %s, %s, %s, %s, TRUE)
        RETURNING id;
    """, (run_id, upload_id, accuracy, f1_score, auc_score))

    model_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Registered model → model_id = {model_id}")
    return model_id
