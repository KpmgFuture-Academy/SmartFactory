import time
import io
import uuid
import boto3
import psycopg2
from PIL import Image
from datetime import datetime

# ----------------------- 설정 -----------------------
DB_INFO = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

BUCKET_NAME = "smart-factory-datalake"
S3_INPUT_PREFIX = "input/"
COMPANY_ID = '11111111-1111-1111-1111-111111111111'
UPLOADER_ID = '22222222-2222-2222-2222-222222222222'
processed_keys = set()
upload_id = None

# -------------------- DB 연결 ----------------------
def connect_db():
    return psycopg2.connect(**DB_INFO)

# ------------- uploads 테이블에 기록 ---------------
def create_upload_entry(conn, folder_path):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO uploads (company_id, uploader_id, file_key, label_type)
            VALUES (%s, %s, %s, %s)
            RETURNING upload_id
        """, (COMPANY_ID, UPLOADER_ID, folder_path, 'binary'))
        result = cur.fetchone()
        conn.commit()
        return result[0]

# ---------- S3에서 이미지 메타데이터 추출 ----------
def fetch_s3_image_metadata(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    img_bytes = obj['Body'].read()
    img = Image.open(io.BytesIO(img_bytes))
    width, height = img.size
    fmt = img.format.lower()
    return width, height, fmt

# ---------- input_image 테이블에 기록 -------------
def insert_image_metadata(conn, upload_id, key, width, height, fmt):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO input_image (upload_id, image_path, width, height, format, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (upload_id, key, width, height, fmt))
    conn.commit()

# ------------------- 실행 루프 ----------------------
def run():
    global upload_id
    s3 = boto3.client('s3')
    conn = connect_db()

    while True:
        print(f"[🔍] S3 '{S3_INPUT_PREFIX}' 폴더 감시 중...")
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_INPUT_PREFIX)

        if 'Contents' in response:
            new_keys = []

            for obj in response['Contents']:
                key = obj['Key']
                if not key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if key in processed_keys:
                    continue

                new_keys.append(key)

            if new_keys:
                print(f"[🆕] 새 이미지 {len(new_keys)}장 감지됨")
                upload_id = create_upload_entry(conn, S3_INPUT_PREFIX)
                print(f"[📦] upload_id 생성됨: {upload_id}")

                for key in new_keys:
                    width, height, fmt = fetch_s3_image_metadata(s3, BUCKET_NAME, key)
                    insert_image_metadata(conn, upload_id, key, width, height, fmt)
                    print(f"[✅] {key} 메타데이터 저장 완료")
                    processed_keys.add(key)

        time.sleep(10)

# -------------------- 메인 -------------------------
if __name__ == '__main__':
    run()