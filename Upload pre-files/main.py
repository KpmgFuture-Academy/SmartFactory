#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ✅ main.py (센서 + 이미지 업로드, binary 이미지 구분, 업로드 테이블 포함 미리보기)

from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import boto3
import psycopg2
import io, json
from PIL import Image
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME, DB_INFO

app = FastAPI()

# ------------------- 모델 정의 --------------------
class FileKeyRequest(BaseModel):
    file_key: str

class UploadIDRequest(BaseModel):
    upload_id: int

class ImageIngestRequest(BaseModel):
    company_id: str
    uploader_id: str
    file_keys: list[str]
    label_type: str = None  # binary, bbox, mask
    binary_labels: dict[str, str] = {}  # {"img_path": "ok" | "defect"}

class IngestRequest(BaseModel):
    company_id: str
    uploader_id: str
    file_key: str
    file_type: str
    task_type: str
    binary_target_column: Optional[str] = None
    multilabel_target_columns: List[str] = []

# ------------------- S3 탐색 --------------------
@app.get("/list_folders")
def list_folders():
    try:
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        result = s3.list_objects_v2(Bucket=BUCKET_NAME, Delimiter="/")
        folders = [prefix["Prefix"].rstrip("/") for prefix in result.get("CommonPrefixes", [])]
        return {"folders": folders}
    except Exception as e:
        print("❌ list_folders 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_files_in_folder")
def list_files_in_folder(prefix: str):
    try:
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        files = [obj["Key"] for obj in result.get("Contents", []) if not obj["Key"].endswith("/")]
        return {"files": files}
    except Exception as e:
        print("❌ list_files_in_folder 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- CSV 처리 --------------------
@app.post("/load_columns")
def load_columns(req: FileKeyRequest):
    try:
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=req.file_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        return {"columns": df.columns.tolist()}
    except Exception as e:
        print("❌ load_columns 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_csv_to_db")
def ingest_csv_to_db(req: IngestRequest):
    try:
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=req.file_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        conn = psycopg2.connect(**DB_INFO)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO uploads (company_id, uploader_id, file_key, file_type, has_target, task_type)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING upload_id
        """, (
            req.company_id,
            req.uploader_id,
            req.file_key,
            req.file_type,
            bool(req.binary_target_column or req.multilabel_target_columns),
            req.task_type
        ))
        upload_id = cur.fetchone()[0]

        target_cols = []
        if req.binary_target_column and req.binary_target_column in df.columns:
            target_cols.append(req.binary_target_column)
        target_cols += [col for col in req.multilabel_target_columns if col in df.columns]

        input_df = df.drop(columns=target_cols) if target_cols else df

        for _, row in input_df.iterrows():
            features_dict = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in row.to_dict().items()
            }
            cur.execute("""
                INSERT INTO input_sensor_data (upload_id, timestamp, features)
                VALUES (%s, NOW(), %s)
            """, (upload_id, json.dumps(features_dict)))

        for i in range(len(df)):
            binary_val = df.iloc[i].get(req.binary_target_column)
            if hasattr(binary_val, "item"):
                binary_val = binary_val.item()

            multi_dict = {
                col: (df.iloc[i][col].item() if hasattr(df.iloc[i][col], "item") else df.iloc[i][col])
                for col in req.multilabel_target_columns if col in df.columns
            }

            cur.execute("""
                INSERT INTO input_target (upload_id, machine_failure, failure_modes)
                VALUES (%s, %s, %s)
            """, (
                upload_id,
                bool(binary_val) if binary_val is not None else None,
                json.dumps(multi_dict) if multi_dict else None
            ))

        conn.commit()
        conn.close()

        return {"status": "success", "upload_id": upload_id}

    except Exception as e:
        print("❌ ingest_csv_to_db 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- 이미지 처리 --------------------
@app.post("/ingest_images")
def ingest_images(req: ImageIngestRequest):
    try:
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        conn = psycopg2.connect(**DB_INFO)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO uploads (company_id, uploader_id, file_key, file_type, has_target, label_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING upload_id
        """, (
            req.company_id,
            req.uploader_id,
            "MULTI_FOLDER_UPLOAD",
            "image",
            True,
            req.label_type
        ))
        upload_id = cur.fetchone()[0]

        for key in req.file_keys:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            img = Image.open(io.BytesIO(obj['Body'].read()))
            width, height = img.size
            fmt = img.format
            is_defect = req.binary_labels.get(key, "ok") == "ok"  # ✅ ok일 때 True, defect일 때 False

            cur.execute("""
                INSERT INTO input_image (upload_id, image_path, width, height, format, is_defect)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (upload_id, key, width, height, fmt, is_defect))

        conn.commit()
        conn.close()
        return {"status": "success", "upload_id": upload_id}
    except Exception as e:
        print("❌ ingest_images 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- 미리보기 --------------------
@app.post("/preview_csv")
def preview_csv(req: UploadIDRequest):
    try:
        conn = psycopg2.connect(**DB_INFO)
        df = pd.read_sql("SELECT features FROM input_sensor_data WHERE upload_id = %s LIMIT 5", conn, params=(req.upload_id,))

        def safe_parse(x):
            if isinstance(x, str):
                return json.loads(x)
            elif isinstance(x, dict):
                return x
            else:
                return {}

        parsed = pd.json_normalize(df["features"].apply(safe_parse)) if "features" in df.columns else pd.DataFrame()

        target_df = pd.read_sql("SELECT machine_failure, failure_modes FROM input_target WHERE upload_id = %s LIMIT 5", conn, params=(req.upload_id,))
        upload_df = pd.read_sql("SELECT * FROM uploads WHERE upload_id = %s", conn, params=(req.upload_id,))

        return {
            "preview": parsed.to_dict(orient="records"),
            "target": target_df.to_dict(orient="records"),
            "upload": upload_df.to_dict(orient="records")
        }

    except Exception as e:
        print("❌ preview_csv 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preview_image")
def preview_image(req: UploadIDRequest):
    try:
        conn = psycopg2.connect(**DB_INFO)
        df = pd.read_sql("""
            SELECT image_path, width, height, format, is_defect, created_at
            FROM input_image
            WHERE upload_id = %s
            LIMIT 5
        """, conn, params=(req.upload_id,))

        # ✅ Boolean → 문자열 변환
        df['is_defect'] = df['is_defect'].map({True: "ok", False: "defect"})

        upload_df = pd.read_sql("SELECT * FROM uploads WHERE upload_id = %s", conn, params=(req.upload_id,))

        return {
            "images": df.to_dict(orient="records"),
            "upload": upload_df.to_dict(orient="records")
        }
    except Exception as e:
        print("❌ preview_image 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))


