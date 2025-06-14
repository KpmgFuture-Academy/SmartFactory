import mlflow
import mlflow.keras
import autokeras as ak
from tensorflow.keras.models import load_model
from datetime import datetime

# 🟢 설정
MODEL_PATH = r"C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5"
UPLOAD_ID = "i1"  # 해당 모델이 학습된 upload_id
ACCURACY = 0.9860   # 예시: 성능지표 직접 입력
F1 = 0.9811
AUC = 0.9986

# 🧠 모델 로드
model = load_model(MODEL_PATH, custom_objects=ak.CUSTOM_OBJECTS)

# 🌐 MLflow 연결
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SmartFactory_Image_Models")

# 🚀 기록 시작
with mlflow.start_run() as run:
    mlflow.keras.log_model(model, "model")  # 모델 등록
    mlflow.log_metrics({
        "accuracy": ACCURACY,
        "f1_score": F1,
        "auc_score": AUC
    })

    run_id = run.info.run_id
    model_name = run.info.run_name
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    # 📦 DB에 등록
    import psycopg2
    DB_INFO = {
        'host': 'localhost',
        'port': '5432',
        'dbname': 'postgres',
        'user': 'postgres',
        'password': '0523'
    }
    conn = psycopg2.connect(**DB_INFO)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO model_registry (
            mlflow_run_id, trained_on_upload_id,
            accuracy, f1_score, auc_score,
            is_active, created_at
        ) VALUES (%s, %s, %s, %s, %s, TRUE, NOW())
    ''', (run_id, UPLOAD_ID, ACCURACY, F1, AUC))
    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ 모델 기록 완료: Run ID = {run_id}, upload_id = {UPLOAD_ID}")