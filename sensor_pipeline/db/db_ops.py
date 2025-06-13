import psycopg2
from config import DB_CONFIG
from datetime import datetime

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def insert_upload_metadata(file_key, data_type, has_target, task_type, company_id, uploader_id):
    """
    Inserts a new upload into the `uploads` table.
    Returns the generated upload_id.
    """
    conn = get_connection()
    cur = conn.cursor()

    sql = """
        INSERT INTO uploads (
            company_id,
            uploader_id,
            file_key,
            data_type,
            has_target,
            task_type,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING upload_id;
    """

    now = datetime.now()
    cur.execute(sql, (
        company_id,
        uploader_id,
        file_key,
        data_type,
        has_target,
        task_type,
        now
    ))

    upload_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Upload metadata inserted (upload_id={upload_id})")
    return upload_id




import json

def insert_sensor_data(upload_id, timestamps, feature_dicts):
    """
    Inserts sensor data rows into the input_sensor_data table.
    """
    assert len(timestamps) == len(feature_dicts), "Timestamps and features must match in length."

    conn = get_connection()
    cur = conn.cursor()

    sql = """
        INSERT INTO input_sensor_data (
            upload_id,
            timestamp,
            features,
            created_at
        ) VALUES (%s, %s, %s, %s);
    """

    now = datetime.now()
    rows = [
        (upload_id, timestamps[i], json.dumps(feature_dicts[i]), now)
        for i in range(len(timestamps))
    ]

    cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted {len(rows)} sensor data rows.")









def insert_target_data(upload_id, binary_targets, multi_label_targets):
    """
    Inserts target values into input_target table.
    """
    assert len(binary_targets) == len(multi_label_targets), "Mismatch in binary and multi-label rows."

    conn = get_connection()
    cur = conn.cursor()

    sql = """
        INSERT INTO input_target (
            upload_id,
            machine_failure,
            failure_modes,
            created_at
        ) VALUES (%s, %s, %s, %s);
    """

    now = datetime.now()
    rows = [
        (upload_id, binary_targets[i], json.dumps(multi_label_targets[i]), now)
        for i in range(len(binary_targets))
    ]

    cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted {len(rows)} target rows.")
