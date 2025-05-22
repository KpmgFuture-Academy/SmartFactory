from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
from PIL import Image
import autokeras as ak
from datetime import datetime
import csv
import os
import pandas as pd
import shutil
import uuid
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from s3_utils import upload_to_s3

app = FastAPI()

def generate_gradcam(model, img_array, last_conv_layer_name="conv2d"):
    # 모델의 마지막 Conv layer로부터 Grad-CAM 계산
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

# 🔧 모델 로드
MODEL_PATH = r"C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5"
model = load_model(MODEL_PATH, custom_objects=ak.CUSTOM_OBJECTS)
image_size = (300, 300)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기 및 전처리
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(image_size)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # (1, 300, 300, 3)

        # 예측
        prediction = model.predict(image)[0][0]
        label = "정상" if prediction >= 0.5 else "불량"
        prob = round(float(prediction), 4)
        
        # 🔄 이미지 분류 저장
        save_root = r"C:\Users\Admin\Desktop\smart_factory_qa\sorted"
        base_label = "ok" if label == "정상" else "defect"
        original_dir = os.path.join(save_root, base_label, "original")
        gradcam_dir = os.path.join(save_root, base_label, "gradcam")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(gradcam_dir, exist_ok=True)

        # 이미지 파일 이름 timestamp와 동일:
        ext = os.path.splitext(file.filename)[-1]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(original_dir, f"{timestamp_str}{ext}")

        # 이미지 내용을 저장
        with open(save_path, "wb") as out_file:
            out_file.write(contents)
            
        # 🔥 Grad-CAM 히트맵 생성 (이건 따로!)
        heatmap = generate_gradcam(model, image)
        heatmap = cv2.resize(heatmap, image_size)
        heatmap = np.uint8(255 * heatmap)

        original = np.array(Image.open(io.BytesIO(contents)).resize(image_size).convert("RGB"))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

        # 저장
        cam_path = os.path.join(gradcam_dir, f"cam_{timestamp_str}.jpeg")
        cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        
        # 🔄 결과 엑셀 저장 (엑셀 열려있을 경우 log.txt에 기록)
        excel_path = r"C:\Users\Admin\Desktop\smart_factory_qa\results\results.xlsx"
        timestamp_now = datetime.now()

        new_row = pd.DataFrame([{
            "timestamp": timestamp_now.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "prediction": label,
            "probability": prob
        }])

        try:
            if os.path.exists(excel_path):
                existing_df = pd.read_excel(excel_path, engine='openpyxl')
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row

            updated_df.to_excel(excel_path, index=False, engine='openpyxl')

        except PermissionError:
            log_path = r"C:\Users\Admin\Desktop\smart_factory_qa\log.txt"
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(f"[{timestamp_now}] ❌ 엑셀 기록 실패 - 파일 열려 있음: {file.filename}\n")

        

        # (예측과 저장 끝난 후에)
        BUCKET_NAME = "smart-factory-datalake"

        # 🔼 Excel 업로드
        upload_to_s3(excel_path, BUCKET_NAME, "classification/results")

        # 🔼 이미지 업로드 (original + shap 하위 폴더 포함)
        upload_to_s3(original_dir, BUCKET_NAME, f"classification/sorted/{base_label}/original")
        upload_to_s3(gradcam_dir, BUCKET_NAME, f"classification/sorted/{base_label}/gradcam")
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": label,
            "probability": f"{prob:.2%}"
        })
        
        
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
