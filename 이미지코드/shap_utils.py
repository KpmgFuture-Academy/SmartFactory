# shap_utils.py
import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def get_background_data(folder_path, image_size=(300, 300), sample_count=50):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=image_size)
            arr = img_to_array(img) / 255.0
            images.append(arr)
            if len(images) >= sample_count:
                break
    return np.array(images)

def explain_prediction(model, background_data, image_array):
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(image_array)
    return shap_values