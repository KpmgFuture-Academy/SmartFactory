import boto3
import time
import os
from pathlib import Path
from predict_batch import predict_and_store

BUCKET = "smart-factory-datalake"
PREFIX = "input/"  # 예: input/ok/ or 그냥 input/
LOCAL_TEMP = r"C:\Users\Admin\Desktop\smart_factory_qa\s3_temp"
CHECK_INTERVAL = 10  # 초

s3 = boto3.client("s3")
downloaded = set()

def list_s3_images():
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    return [item["Key"] for item in response.get("Contents", []) if item["Key"].lower().endswith((".jpg", ".jpeg", ".png"))]

def download_and_process(key):
    filename = os.path.basename(key)
    local_path = os.path.join(LOCAL_TEMP, filename)
    s3.download_file(BUCKET, key, local_path)
    print(f"[⬇️] 다운로드 완료: {filename}")
    predict_and_store(Path(local_path))

if __name__ == "__main__":
    os.makedirs(LOCAL_TEMP, exist_ok=True)
    while True:
        try:
            all_keys = list_s3_images()
            new_keys = [k for k in all_keys if k not in downloaded]

            for key in new_keys:
                download_and_process(key)
                downloaded.add(key)

            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n[🛑] 감지 중지됨")
            break
        except Exception as e:
            print(f"[⚠️] 오류 발생: {e}")