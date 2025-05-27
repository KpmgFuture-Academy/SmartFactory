#!/usr/bin/env python
# coding: utf-8

# !pip install protobuf==3.20.*

# In[1]:


import tensorflow as tf
import autokeras as ak

print("✅ TensorFlow:", tf.__version__)
print("✅ AutoKeras:", ak.__version__)
print("✅ GPU:", tf.config.list_physical_devices('GPU'))


# In[2]:


# 샘플 데이터로 
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
sample_size = 3000

AWS_ACCESS_KEY = 'AKIA2UMF2NRTTGTVHHUD'
AWS_SECRET_KEY = 'Z/90YNZ1tiH2e8QEeXObw+incc99QIDWsrfN2bTb'
BUCKET_NAME = 'smart-factory-datalake'

DB_INFO = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

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

def train_and_log_model(upload_id):
    x, y = fetch_sampled_images(upload_id, sample_size)
    if len(np.unique(y)) < 2:
        raise ValueError("❌ 클래스가 하나만 존재합니다. 최소 두 클래스가 필요합니다.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    try:
        y_pred_probs = clf.predict_proba(x_test)
    except AttributeError:
        y_pred_probs = clf.predict(x_test)
        print("⚠️ predict_proba가 없음. predict 사용. 확률 출력으로 변환:")
        y_pred_probs = y_pred_probs.flatten()  # (n_samples, 1) → (n_samples,)
    y_pred = np.round(y_pred_probs).astype(int).flatten()
    try:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        n_classes = len(np.unique(y_test))
        if n_classes < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(y_test, y_pred_probs)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    except Exception as e:
        print(f"⚠️ 지표 계산 실패: {e}")
        print(f"디버깅 정보 - y_test: {np.unique(y_test, return_counts=True)}")
        print(f"디버깅 정보 - y_pred_probs shape: {y_pred_probs.shape}")
        accuracy, f1, auc = 0.0, 0.0, 0.0
    model = clf.export_model()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("SmartFactory_Image_Models")
    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, "model")
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1, "auc_score": auc})
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

if __name__ == "__main__":
    upload_id = 1
    train_and_log_model(upload_id)


# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 사용 안 함
# 전체 데이터로
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

# TensorFlow GPU 비활성화 명시
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices([], 'GPU')
    print("✅ GPU 비활성화 완료")
else:
    print("✅ GPU 없음, CPU로 진행")

DB_INFO = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

AWS_ACCESS_KEY = 'AKIA2UMF2NRTTGTVHHUD'
AWS_SECRET_KEY = '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'
BUCKET_NAME = 'smart-factory-datalake'

image_size = (300, 300)

s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

def fetch_sampled_images(upload_id):
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
    df_full = df.reset_index(drop=True)  # 샘플링 제거, 전체 데이터 사용
    print(f"✅ 전체 데이터 로드 완료: {len(df_full)}개 (두 클래스 포함됨)")
    x, y = [], []
    for _, row in tqdm(df_full.iterrows(), total=len(df_full), desc="📦 이미지 로딩"):
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

def train_and_log_model(upload_id):
    x, y = fetch_sampled_images(upload_id)
    if len(np.unique(y)) < 2:
        raise ValueError("❌ 클래스가 하나만 존재합니다. 최소 두 클래스가 필요합니다.")
    
    # 데이터 크기 확인
    print(f"📏 데이터 크기 - x: {x.shape}, y: {y.shape}")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    
    # 학습 데이터 크기 확인
    print(f"📏 학습 데이터 크기 - x_train: {x_train.shape}, y_train: {y_train.shape}")
    
    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    try:
        clf.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        print(f"디버깅 정보 - x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        raise
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    try:
        y_pred_probs = clf.predict_proba(x_test)
    except AttributeError:
        y_pred_probs = clf.predict(x_test)
        print("⚠️ predict_proba가 없음. predict 사용. 확률 출력으로 변환:")
        y_pred_probs = y_pred_probs.flatten()  # (n_samples, 1) → (n_samples,)
    y_pred = np.round(y_pred_probs).astype(int).flatten()
    try:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        n_classes = len(np.unique(y_test))
        if n_classes < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(y_test, y_pred_probs)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    except Exception as e:
        print(f"⚠️ 지표 계산 실패: {e}")
        print(f"디버깅 정보 - y_test: {np.unique(y_test, return_counts=True)}")
        print(f"디버깅 정보 - y_pred_probs shape: {y_pred_probs.shape}")
        accuracy, f1, auc = 0.0, 0.0, 0.0
    model = clf.export_model()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("SmartFactory_Image_Models")
    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, "model")
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1, "auc_score": auc})
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

if __name__ == "__main__":
    upload_id = 1
    train_and_log_model(upload_id)


# In[ ]:


# 1. Test data를 DB 적재 후 
# 2. 기존 데이터 적재 수(input_image의 수 count)와 새로 들어온 Test data 수 비교 
# 


# In[ ]:


# 3. 모델 재학습(autokeras)


# In[ ]:


# 4. 기존 모델 vs 새 모델


# In[ ]:


# 5. 사용자 선택


# In[ ]:


# 6. 새 모델 registry 등록


# In[ ]:


# 7. 운영 배포 파이프라인 연동


# In[3]:


# test 위한 S3 test 데이터 추출 후 DB 적재 
import os
import boto3
import psycopg2
import random
from datetime import datetime
from PIL import Image
import io

# 환경 설정
AWS_ACCESS_KEY = 'AKIA2UMF2NRTTGTVHHUD'
AWS_SECRET_KEY = 'Z/90YNZ1tiH2e8QEeXObw+incc99QIDWsrfN2bTb'
BUCKET_NAME = 'smart-factory-datalake'
PREFIX = 'test/'  # ✅ test 데이터 위치
SAMPLE_COUNT = 10          # ✅ 고정된 개수 샘플링

# 테스트용 고정 ID
COMPANY_ID = '11111111-1111-1111-1111-111111111111'
UPLOADER_ID = '22222222-2222-2222-2222-222222222222'

DB_INFO = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Step 1. S3 키 리스트 불러오기
def list_image_keys(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))]

# Step 2. 10개만 샘플링
def sample_image_keys(all_keys, count):
    return random.sample(all_keys, min(count, len(all_keys)))

# Step 3. uploads 테이블에 업로드 메타 생성
def create_upload_entry(file_key_prefix):
    conn = psycopg2.connect(**DB_INFO)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO uploads (company_id, uploader_id, file_key, data_type, has_target, label_type, task_type, created_at)
        VALUES (%s, %s, %s, 'image', FALSE, 'binary', NULL, NOW())
        RETURNING upload_id
    """, (COMPANY_ID, UPLOADER_ID, file_key_prefix))
    upload_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    print(f"📥 uploads 테이블에 등록 완료 → upload_id: {upload_id}")
    return upload_id

# Step 4. S3 이미지 메타 추출
def get_image_metadata(key):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    img = Image.open(io.BytesIO(obj['Body'].read()))
    width, height = img.size
    img_format = img.format.lower()
    return width, height, img_format

# Step 5. input_image에 삽입
def insert_images_to_db(image_keys, upload_id):
    conn = psycopg2.connect(**DB_INFO)
    cur = conn.cursor()

    for key in image_keys:
        try:
            width, height, img_format = get_image_metadata(key)
            is_defect = random.choice([True, False])  # 임의로 라벨 부여
            cur.execute("""
                INSERT INTO input_image (upload_id, image_path, width, height, format, is_defect, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (upload_id, key, width, height, img_format, is_defect))
            print(f"✅ 등록 완료: {key}")
        except Exception as e:
            print(f"⚠️ 등록 실패: {key} → {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"📦 총 {len(image_keys)}개 이미지 등록 완료 (upload_id: {upload_id})")

# 전체 실행 흐름
if __name__ == "__main__":
    all_keys = list_image_keys(BUCKET_NAME, PREFIX)
    sampled_keys = sample_image_keys(all_keys, SAMPLE_COUNT)
    print(f"🔍 S3에서 {len(all_keys)}개 중 {len(sampled_keys)}개 샘플링")

    upload_id = create_upload_entry(PREFIX)
    insert_images_to_db(sampled_keys, upload_id)


# In[4]:


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
from sklearn.metrics import f1_score, roc_auc_score

# -------------------- 설정 --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 사용 안 함

DB_INFO = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'
}

AWS_ACCESS_KEY = 'AKIA2UMF2NRTTGTVHHUD'
AWS_SECRET_KEY = 'Z/90YNZ1tiH2e8QEeXObw+incc99QIDWsrfN2bTb'
BUCKET_NAME = 'smart-factory-datalake'

image_size = (300, 300)
total_sample = 15  # 학습에 사용할 전체 샘플 수
sample_threshold = 0.000000001  # 증분 기준 비율
s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# -------------------- 증분 판단 --------------------
def is_incremental_significant(upload_id, threshold=sample_threshold):
    with psycopg2.connect(**DB_INFO) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM input_image")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM input_image WHERE upload_id = %s", (upload_id,))
            new = cur.fetchone()[0]
    ratio = new / total if total > 0 else 1.0
    print(f"🔍 기존: {total - new}, 신규: {new}, 비율: {ratio:.2%}")
    return ratio >= threshold

# -------------------- 최신 모델 조회 --------------------
def get_latest_model_accuracy_and_id():
    with psycopg2.connect(**DB_INFO) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, accuracy FROM model_registry
                WHERE data_type = 'image' AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 50
            """)
            row = cur.fetchone()
            return (row[0], row[1]) if row else (None, 0.0)

# -------------------- 이미지 병합 샘플링 --------------------
def fetch_merged_sample_images(upload_id, total_sample=15):
    with psycopg2.connect(**DB_INFO) as conn:
        # 신규 데이터 전체
        df_new = pd.read_sql("""
            SELECT image_path, is_defect
            FROM input_image
            WHERE upload_id = %s
        """, conn, params=(upload_id,))

        remain = max(0, total_sample - len(df_new))
        if remain > 0:
            df_old = pd.read_sql(f"""
                SELECT image_path, is_defect
                FROM input_image
                WHERE upload_id != %s
                ORDER BY RANDOM()
                LIMIT {remain}
            """, conn, params=(upload_id,))
        else:
            df_old = pd.DataFrame(columns=['image_path', 'is_defect'])

    df = pd.concat([df_new, df_old]).reset_index(drop=True)

    x, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="📦 이미지 로딩"):
        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=row["image_path"])
            img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB").resize(image_size)
            x.append(np.array(img, dtype=np.float32) / 255.0)
            y.append(int(row["is_defect"]))
        except Exception as e:
            print(f"⚠️ {row['image_path']} 로딩 실패: {e}")

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32)

# -------------------- 학습 및 등록 --------------------
def train_and_register_model(upload_id, total_sample=15, threshold=sample_threshold):
    if not is_incremental_significant(upload_id, threshold):
        print("📌 증분 부족. 학습 생략.")
        return

    try:
        x, y = fetch_merged_sample_images(upload_id, total_sample)
    except Exception as e:
        print(f"❌ 이미지 로딩 실패: {e}")
        return

    if len(np.unique(y)) < 2:
        print("📌 클래스 불균형으로 학습 불가")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    clf.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test)  # ✅ 이 부분을 명시적으로 지정
    )

    y_pred = np.argmax(clf.predict(x_test), axis=1)
    new_accuracy = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    old_model_id, old_accuracy = get_latest_model_accuracy_and_id()
    print(f"📈 이전 acc={old_accuracy:.4f}, 새 acc={new_accuracy:.4f} | f1={f1:.4f}, auc={auc:.4f}")

    was_accepted = False
    if new_accuracy > old_accuracy:
        decision = input("📌 새 모델을 운영에 반영할까요? (y/n): ").strip().lower()
        was_accepted = decision == 'y'

    model = clf.export_model()
    mlflow.set_experiment("SmartFactory_Image_Models")
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, "model")
        mlflow.log_metrics({
            "accuracy": new_accuracy,
            "f1_score": f1,
            "auc_score": auc
        })

        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        mlflow_run_id = run.info.run_id
        model_name = run.info.run_name

        with psycopg2.connect(**DB_INFO) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_registry (
                        model_name, version, mlflow_run_id,
                        trained_on_upload_id, data_type,
                        accuracy, f1_score, auc_score,
                        is_active, created_at
                    )
                    VALUES (%s, %s, %s, %s, 'image', %s, %s, %s, FALSE, NOW())
                """, (
                    'autokeras_image_model',
                    version,
                    mlflow_run_id,
                    upload_id,
                    new_accuracy,
                    f1,
                    auc
                ))

                cur.execute("SELECT id FROM model_registry WHERE mlflow_run_id = %s", (mlflow_run_id,))
                new_model_id = cur.fetchone()[0]

                if was_accepted:
                    cur.execute("UPDATE model_registry SET is_active = FALSE WHERE data_type = 'image' AND is_active = TRUE")
                    cur.execute("UPDATE model_registry SET is_active = TRUE WHERE id = %s", (new_model_id,))

                cur.execute("""
                    INSERT INTO model_selection_log (
                        data_type, old_model_id, new_model_id,
                        accuracy_diff, selected_by_user_id,
                        was_accepted, reason, selected_at
                    )
                    VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW())
                """, (
                    'image',
                    old_model_id,
                    new_model_id,
                    round(new_accuracy - old_accuracy, 6),
                    was_accepted,
                    "User approved via prompt" if was_accepted else "User rejected via prompt"
                ))

            conn.commit()

    print("✅ 모델 학습 및 기록 완료" if was_accepted else "📌 학습 완료 (운영 미반영)")

# -------------------- 실행 --------------------
if __name__ == "__main__":
    train_and_register_model(upload_id=2)



# In[ ]:




