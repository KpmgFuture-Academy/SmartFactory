from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import autokeras as ak
from PIL import Image
import uuid
import tensorflow as tf
from gradcam_utils import generate_gradcam
from s3_utils import upload_to_s3

# 설정
MODEL_PATH = r"C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5"
INPUT_DIR = r"C:\Users\Admin\Desktop\smart_factory_qa\input_images"
SORTED_DIR = r"C:\Users\Admin\Desktop\smart_factory_qa\sorted"
RESULT_PATH = r"C:\Users\Admin\Desktop\smart_factory_qa\results\results.xlsx"
IMAGE_SIZE = (300, 300)

# 모델 로드
model = load_model(MODEL_PATH, custom_objects=ak.CUSTOM_OBJECTS)

# Grad-CAM 유틸


# 예측 및 저장
def predict_and_store(img_path):
    # 🔹 이미지 로드 및 전처리
    image = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # 🔹 예측
    pred = model.predict(arr)[0][0]
    label = "정상" if pred >= 0.5 else "불량"
    prob = round(float(pred), 4)

    # 🔹 고유 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    ext = os.path.splitext(img_path.name)[-1]
    base_label = "ok" if label == "정상" else "defect"
    filename = f"{timestamp}_{unique_id}{ext}"

    # 🔹 저장 경로 구성
    save_dir = Path(SORTED_DIR) / base_label / "original"
    cam_dir = Path(SORTED_DIR) / base_label / "gradcam"
    save_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    # 🔹 원본 이미지 저장
    save_path = save_dir / filename
    image.save(save_path)
    print(f"[✅] 원본 저장 완료: {save_path}")

    # 🔹 Grad-CAM 생성 및 저장
    heatmap = generate_gradcam(model, arr)
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.array(image), 0.6, heatmap_color, 0.4, 0)
    cam_filename = f"cam_{timestamp}_{unique_id}.jpeg"
    cam_save_path = cam_dir / cam_filename
    cv2.imwrite(str(cam_save_path), cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    print(f"[✅] Grad-CAM 저장 완료: {cam_save_path}")

    # 🔹 결과 엑셀 저장
    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": img_path.name,
        "saved_filename": filename,
        "prediction": label,
        "probability": prob
    }])

    if os.path.exists(RESULT_PATH):
        df = pd.read_excel(RESULT_PATH, engine="openpyxl")
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_excel(RESULT_PATH, index=False, engine="openpyxl")
    print(f"[📝] 엑셀 저장 완료: {RESULT_PATH}")

# 실행
input_folder = Path(INPUT_DIR)
images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.jpeg")) + list(input_folder.glob("*.png"))

for img_file in images:
    predict_and_store(img_file)
    # img_file.unlink()  # ← 처리 후 삭제하고 싶으면 주석 해제
    
BUCKET_NAME = "smart-factory-datalake"

# 🔼 모든 이미지 예측 후 결과 업로드
upload_to_s3(RESULT_PATH, BUCKET_NAME, "classification/results")
upload_to_s3(SORTED_DIR, BUCKET_NAME, "classification/sorted")