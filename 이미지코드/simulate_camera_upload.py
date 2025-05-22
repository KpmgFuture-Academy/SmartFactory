import boto3
import os
import time
from pathlib import Path

# 🛠️ 설정
LOCAL_IMAGE_DIR = r"C:\Users\Admin\Desktop\smart_factory_qa\camera"
BUCKET_NAME = "smart-factory-datalake"
S3_FOLDER = "input/"  # S3 업로드 경로
BATCH_SIZE = 3
INTERVAL_SEC = 10  # 업로드 간격(초)

# S3 클라이언트
s3 = boto3.client("s3")

# 이미지 리스트 준비
all_images = list(Path(LOCAL_IMAGE_DIR).glob("*.jpg")) + list(Path(LOCAL_IMAGE_DIR).glob("*.jpeg")) + list(Path(LOCAL_IMAGE_DIR).glob("*.png"))
uploaded = set()

print(f"[📸] 총 이미지 수: {len(all_images)}")

# 업로드 루프
while True:
    # 아직 업로드 안 된 이미지들만 선별
    pending_images = [img for img in all_images if img.name not in uploaded]

    if not pending_images:
        print("[✅] 모든 이미지를 업로드 완료했습니다.")
        break

    # 업로드할 이미지 배치 선택
    batch = pending_images[:BATCH_SIZE]

    for img_path in batch:
        s3_key = os.path.join(S3_FOLDER, img_path.name)
        with open(img_path, "rb") as f:
            s3.upload_fileobj(f, BUCKET_NAME, s3_key)
        print(f"[⬆️] 업로드 완료: {img_path.name}")
        uploaded.add(img_path.name)

    print(f"[⏱️] {INTERVAL_SEC}초 대기 후 다음 배치 업로드...")
    time.sleep(INTERVAL_SEC)