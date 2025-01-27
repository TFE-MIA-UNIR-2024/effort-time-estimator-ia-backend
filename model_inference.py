# model_inference.py

import tensorflow as tf
import numpy as np
import os

def load_model():
    """
    Carga el modelo 'model_estimation.h5' si existe.
    Retorna el modelo o None si no está disponible.
    """
    if not os.path.exists("model_estimation.h5"):
        return None
    model = tf.keras.models.load_model("model_estimation.h5")
    return model

def predict_effort(model, input_vector):
    """
    input_vector: lista/array con los features, e.g. [cantidad_objeto, factor_ia].
    Devuelve el esfuerzo estimado (float).
    """
    arr = np.array(input_vector, dtype=float).reshape(1, -1)
    pred = model.predict(arr)
    return float(pred[0][0])
