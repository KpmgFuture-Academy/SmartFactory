# db/insert_shap.py

import pandas as pd
import json
from core.db_connector import get_connection

def insert_shap_values(df: pd.DataFrame, shap_values: pd.DataFrame, upload_id: str, model_id: str):
    """
    Inserts SHAP values into prediction_result_sensor table.

    Parameters:
    - df: original DataFrame that includes timestamps
    - shap_values: SHAP values (DataFrame with same index as df)
    - upload_id: str, the upload_id used for prediction
    - model_id: str, the model_id used
    """
    if "timestamp" not in df.columns:
        raise ValueError("The dataframe must contain a 'timestamp' column.")

    conn = get_connection()
    cursor = conn.cursor()

    for idx, row in df.iterrows():
        timestamp = row["timestamp"]
        shap_row = shap_values.loc[idx].to_dict()
        shap_json = json.dumps(shap_row)

        cursor.execute("""
            INSERT INTO prediction_result_sensor (upload_id, model_id, timestamp, shap_summary)
            VALUES (%s, %s, %s, %s)
        """, (upload_id, model_id, timestamp, shap_json))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Inserted {len(df)} SHAP rows into prediction_result_sensor.")
