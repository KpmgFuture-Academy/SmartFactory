#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ✅ S3 이미지 데이터를 PostgreSQL에 등록하는 스크립트
# 목적: def_front/, ok_front/ 이미지 → uploads + input_image 테이블 자동 적재

import boto3
import psycopg2
import io
from PIL import Image
from datetime import datetime

# ---------------------- 설정 ----------------------
AWS_ACCESS_KEY = 'AKIA2UMF2NRTTGTVHHUD'
AWS_SECRET_KEY = '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'
BUCKET_NAME = 'smart-factory-datalake'

DB_INFO = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "0523",
    "port": 5432,
}

S3_PREFIXES = {
    "def_front/": True,   # 불량 이미지 (is_defect=True)
    "ok_front/": False    # 정상 이미지 (is_defect=False)
}

# ---------------------- S3 및 DB 연결 ----------------------
s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
conn = psycopg2.connect(**DB_INFO)
cur = conn.cursor()

# ---------------------- uploads 테이블 등록 ----------------------
cur.execute("""
    INSERT INTO uploads (company_id, uploader_id, file_key, has_target, label_type)
    VALUES (%s, %s, %s, TRUE, 'binary')
    RETURNING upload_id
""", (
    '11111111-1111-1111-1111-111111111111',
    '22222222-2222-2222-2222-222222222222',
    'manual_upload/def_ok_front/'
))
upload_id = cur.fetchone()[0]
print(f"\n🟢 Upload ID 생성됨: {upload_id}\n")

# ---------------------- 이미지 등록 ----------------------
for prefix, is_defect in S3_PREFIXES.items():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"): continue  # 디렉토리 무시

            try:
                response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                image = Image.open(io.BytesIO(response["Body"].read()))
                width, height = image.size
                fmt = image.format

                cur.execute("""
                    INSERT INTO input_image (upload_id, image_path, width, height, format, is_defect)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (upload_id, key, width, height, fmt, is_defect))

                print(f"✅ 등록 완료: {key} ({'불량' if is_defect else '정상'})")

            except Exception as e:
                print(f"⚠️ 실패: {key} → {e}")

# ---------------------- 완료 처리 ----------------------
conn.commit()
cur.close()
conn.close()
print("\n🎉 모든 이미지 등록이 완료되었습니다.")


# In[ ]:


# ✅ 처음 input? -> yes 인 경우의 전체 파이프라인 코드
# S3에서 이미지 선택 -> DB 저장 -> 모델 탐색/등록까지 포함 (샘플링 포함)

import os
import io
import boto3
import psycopg2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
import autokeras as ak
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# -------------------- 설정 --------------------

AWS_ACCESS_KEY = 'AKIA2UMF2NRTTGTVHHUD'
AWS_SECRET_KEY = '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'
BUCKET_NAME = 'smart-factory-datalake'

DB_INFO = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

image_size = (300, 300)
sample_size = 3000
s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

def fetch_latest_upload_id():
    conn = psycopg2.connect(**DB_INFO)
    cur = conn.cursor()
    cur.execute("""
        SELECT upload_id FROM uploads
        WHERE label_type = 'binary'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    result = cur.fetchone()
    cur.close()
    conn.close()
    if not result:
        raise Exception("❌ binary 타입의 upload_id가 없습니다.")
    return result[0]

def fetch_sampled_images(upload_id, sample_size):
    print(f"\n🟢 DB에서 upload_id={upload_id} 데이터 로딩 중")
    conn = psycopg2.connect(**DB_INFO)
    df = pd.read_sql("""
        SELECT id AS image_id, image_path, is_defect
        FROM input_image
        WHERE upload_id = %s
    """, conn, params=(upload_id,))
    present_classes = set(df["is_defect"].unique())
    if len(present_classes) < 2:
        missing_class = 1 if 0 in present_classes else 0
        print(f"⚠️ 클래스 {missing_class} 보완을 위해 DB에서 추가 조회")
        df_add = pd.read_sql(f"""
            SELECT id AS image_id, image_path, is_defect
            FROM input_image
            WHERE is_defect = {missing_class} AND upload_id != %s
            ORDER BY RANDOM()
            LIMIT 10
        """, conn, params=(upload_id,))
        df = pd.concat([df, df_add], ignore_index=True)
    conn.close()
    if df.empty:
        raise ValueError(f"❌ upload_id {upload_id}에 해당하는 이미지가 없습니다.")
    df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    print(f"✅ 샘플링 완료: {len(df_sampled)}개 (두 클래스 포함됨)")
    x, y = [], []
    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="📦 이미지 로딩"):
        try:
            key = row["image_path"]
            label = int(row["is_defect"])
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB").resize(image_size)
            x.append(np.array(img) / 255.0)
            y.append(label)
        except Exception as e:
            print(f"⚠️ {key} → {e}")
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32)

def train_and_register_model(upload_id):
    x, y = fetch_sampled_images(upload_id, sample_size)
    if len(np.unique(y)) < 2:
        print("❌ 클래스가 하나입니다. 학습 생략")
        return
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    try:
        y_pred_probs = clf.predict(x_test).flatten()
    except:
        y_pred_probs = clf.predict(x_test)
    y_pred = np.round(y_pred_probs).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("SmartFactory_Image_Models")

    with mlflow.start_run() as run:
        model = clf.export_model()
        mlflow.keras.log_model(model, "model")
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1, "auc_score": auc})

        conn = psycopg2.connect(**DB_INFO)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_registry (
                mlflow_run_id, trained_on_upload_id,
                accuracy, f1_score, auc_score, is_active, created_at
            ) VALUES (%s, %s, %s, %s, %s, TRUE, NOW())
        """, (run.info.run_id, upload_id, accuracy, f1, auc))
        conn.commit()
        conn.close()
    print("✅ 모델 등록 완료")

if __name__ == "__main__":
    upload_id = fetch_latest_upload_id()
    train_and_register_model(upload_id)

