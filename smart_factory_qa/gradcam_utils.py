import tensorflow as tf
import numpy as np


# def find_last_conv_layer(model):
#     """
#     AutoKeras 등 복잡한 모델에서도 작동하는 Conv2D 레이어 탐색 함수
#     """
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#         # AutoKeras는 nested Functional 모델을 포함할 수 있음
#         if isinstance(layer, tf.keras.Model):
#             sublayer = find_last_conv_layer(layer)
#             if sublayer:
#                 return sublayer
#     return None


def generate_gradcam(model, image_array, layer_name="conv2d"):
    import tensorflow as tf
    import numpy as np

    # 자동탐색: 마지막 Conv2D 레이어 이름 찾기
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        else:
            # 실패했을 경우 fallback 레이어 이름 수동 지정
            layer_name = "conv2d"  # 필요 시 직접 수정

    # Grad-CAM 모델 구성
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # 정규화
    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return heatmap