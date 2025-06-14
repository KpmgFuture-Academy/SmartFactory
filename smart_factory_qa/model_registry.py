import os
import io
import boto3
import psycopg2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import autokeras as ak
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# ------------------------- 설정 -------------------------
image_size = (300, 300)
sample_size = 15

AWS_ACCESS_KEY = 'AKIA2UMF2NRT6ZYSUMMF'
AWS_SECRET_KEY = 's/6bAvd206bInerzTT2us97JPUEF+CHVAv7gU+0W'
BUCKET_NAME = 'smart-factory-datalake'

DB_INFO = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# ------------------------ 이미지 로딩 ------------------------
def fetch_sampled_images(upload_id, sample_size):
    print(f"\n🟢 DB에서 upload_id={upload_id} 데이터 로딩 중")
    conn = psycopg2.connect(**DB_INFO)

    # 기본적으로 upload_id에 해당하는 이미지 불러오기
    df = pd.read_sql("""
        SELECT id AS image_id, image_path, is_defect
        FROM input_image
        WHERE upload_id = %s
    """, conn, params=(upload_id,))

    # 클래스 종류 확인
    present_classes = set(df["is_defect"].unique())

    # 클래스가 하나뿐일 경우 → 다른 클래스 보완
    if len(present_classes) < 2:
        missing_class = 1 if 0 in present_classes else 0
        print(f"⚠️ 클래스 {missing_class} 보완을 위해 DB에서 추가 조회")
        df_add = pd.read_sql(f"""
            SELECT id AS image_id, image_path, is_defect
            FROM input_image
            WHERE is_defect = {missing_class} AND upload_id != %s
            ORDER BY RANDOM()
            LIMIT 5
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

# ------------------------ 학습 및 저장 ------------------------
def train_and_log_model(upload_id):
    x, y = fetch_sampled_images(upload_id, sample_size)

    if len(np.unique(y)) < 2:
        raise ValueError("❌ 클래스가 하나만 존재합니다. 최소 두 클래스가 필요합니다.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    y_pred_probs = clf.predict(x_test)
    if y_pred_probs.shape[1] == 1:
        y_pred = np.round(y_pred_probs).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    try:
        accuracy = np.mean(y_pred == y_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0
    except Exception as e:
        print(f"⚠️ 지표 계산 실패: {e}")
        accuracy, f1, auc = 0.0, 0.0, 0.0

    model = clf.export_model()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("SmartFactory_Image_Models")

    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, "model")
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "auc_score": auc
        })

        mlflow_run_id = run.info.run_id
        model_name = run.info.run_name
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        conn = psycopg2.connect(**DB_INFO)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO model_registry (
                model_name, version, mlflow_run_id,
                trained_on_upload_id, data_type,
                accuracy, f1_score, auc_score,
                is_active, created_at
            ) VALUES (%s, %s, %s, %s, 'image', %s, %s, %s, TRUE, NOW())
        ''', (model_name, version, mlflow_run_id, upload_id, accuracy, f1, auc))
        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ 모델 등록 완료: {model_name}, version={version}, Run ID={mlflow_run_id}")

# ------------------------ 실행 ------------------------
if __name__ == "__main__":
    upload_id = 1  # 원하는 upload_id 지정
    train_and_log_model(upload_id)