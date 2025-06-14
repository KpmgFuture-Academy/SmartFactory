import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🔧 설정
model_path = r"C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5"
test_dir = r"C:\Users\Admin\Desktop\smart_factory_qa\data\casting_data\test"
image_size = (300, 300)
batch_size = 32

# 🔄 데이터 로딩
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # 예측-정답 매칭을 위해 shuffle은 False
)

# 📦 데이터 전체 로딩
x_test = []
y_test = []
for i in range(len(test_gen)):
    x_batch, y_batch = test_gen[i]
    x_test.append(x_batch)
    y_test.append(y_batch)
    if (i + 1) * batch_size >= test_gen.n:
        break

x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# 🤖 모델 로드
model = load_model(
    r"C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5",
    custom_objects=ak.CUSTOM_OBJECTS
)

# 🔮 예측
y_probs = model.predict(x_test).flatten()
y_pred = (y_probs >= 0.5).astype(int)

# 📈 지표 계산
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_probs)

# 📢 출력
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ AUC Score: {auc:.4f}")