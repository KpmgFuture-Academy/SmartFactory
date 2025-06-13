import psycopg2
from config import DB_CONFIG

def insert_upload(file_key: str, has_target: bool, task_type: str = "binary_classification", user_id=None, company_id=None):
    """
    Inserts a new upload record and returns the generated upload_id (e.g., 's1').

    Parameters:
    - file_key: path like 'ai4i2020/ai4i2020.csv'
    - has_target: True/False
    - task_type: sensor_task_type_enum
    - user_id/company_id: default test IDs will be used if None
    """

    company_id = company_id or '11111111-1111-1111-1111-111111111111'
    user_id = user_id or '22222222-2222-2222-2222-222222222222'

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO uploads (company_id, uploader_id, file_key, has_target, task_type)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING upload_id;
""", (company_id, user_id, file_key, has_target, task_type))



    upload_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()

    print(f"📁 Created upload entry → upload_id = {upload_id}")
    return upload_id
