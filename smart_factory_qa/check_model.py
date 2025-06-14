from tensorflow.keras.models import load_model
import autokeras as ak  # AutoKeras 모델이면 반드시 함께 임포트
from tensorflow.keras.utils import plot_model

# 모델 경로
model_path = r"C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5"

# 모델 로드
model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)

# 요약 출력
model.summary()