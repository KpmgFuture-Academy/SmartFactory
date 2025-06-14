import os
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🔧 경로 설정
base_dir = r'C:\Users\Admin\Desktop\smart_factory_qa\data\casting_data'
train_path = os.path.join(base_dir, 'train')
test_path = os.path.join(base_dir, 'test')
model_save_path = r'C:\Users\Admin\Desktop\smart_factory_qa\models\best_model.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# ⚙️ 전처리: 이미지 로딩 및 정규화
image_size = (300, 300)
batch_size = 16
datagen = ImageDataGenerator(rescale=1.0/255)

def load_data(generator):
    x_data, y_data = [], []
    for i in range(len(generator)):
        x, y = generator[i]
        x_data.append(x)
        y_data.append(y)
    return tf.concat(x_data, axis=0), tf.concat(y_data, axis=0)

train_gen = datagen.flow_from_directory(
    train_path, target_size=image_size, color_mode="rgb",
    class_mode="binary", batch_size=batch_size, shuffle=True
)

test_gen = datagen.flow_from_directory(
    test_path, target_size=image_size, color_mode="rgb",
    class_mode="binary", batch_size=batch_size, shuffle=False
)

x_train, y_train = load_data(train_gen)
x_test, y_test = load_data(test_gen)

# AutoKeras가 요구하는 numpy 배열로 변환
x_train = x_train.numpy()
y_train = y_train.numpy()
x_test = x_test.numpy()
y_test = y_test.numpy()

# 🤖 AutoML로 모델 찾고 학습
clf = ak.ImageClassifier(overwrite=True, max_trials=5)  # 5개의 다른 모델 탐색
clf.fit(x_train, y_train, epochs=10)

# 📈 평가
clf.evaluate(x_test, y_test)

# 💾 모델 저장
try:
    model = clf.export_model()
    model.save(model_save_path)
    print(f"✅ 모델 저장 완료: {model_save_path}")
except Exception as e:
    print(f"❌ 모델 저장 실패: {e}")