# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
regularizers = keras.regularizers
from supabase_client import get_supabase_client

def fetch_historical_data():
    """
    Ejemplo: lee de la tabla 'estimacion_esfuerzo_construccion'
    en Supabase para obtener datos históricos (cantidad_objeto_estimado, esfuerzo_real, etc.).
    Ajusta si necesitas JOINS u otras tablas.
    """
    supabase = get_supabase_client()
    response = supabase.table("estimacion_esfuerzo_construccion").select("*").execute()
    data = response.data
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """
    Transforma columnas en X, y para entrenamiento de la RNA.
    Ejemplo:
      - 'cantidad_objeto_estimado' y 'factor_ia' (si existiese) son features (X).
      - 'esfuerzo_real' es la etiqueta (y).
    """
    df["cantidad_objeto_estimado"] = df["cantidad_objeto_estimado"].fillna(0)

    if "esfuerzo_real" not in df.columns:
        df["esfuerzo_real"] = 10  # fallback si no existe

    X_cols = ["cantidad_objeto_estimado"]
    if "factor_ia" in df.columns:
        X_cols.append("factor_ia")

    X = df[X_cols].values.astype(float)
    y = df["esfuerzo_real"].values.astype(float)

    return X, y, X_cols

def create_model(input_dim):
    """
    Construye la MLP con regularización L2 y Dropout
    para evitar sobreajuste.
    """
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')  # salida = esfuerzo estimado
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_and_update_factors():
    # 1. Leer datos
    df = fetch_historical_data()
    X, y, X_cols = preprocess_data(df)

    # 2. Crear / entrenar el modelo
    model = create_model(input_dim=X.shape[1])
    history = model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)

    # 3. Guardar el modelo a disco
    model.save("model_estimation.h5")

    # 4. Ejemplo de actualización de un factor en la BD (e.g. factor_ia)
    supabase = get_supabase_client()
    final_mae = history.history['mae'][-1]
    correction_factor = 1.0 + (final_mae / 100.0)

    # Actualiza Parametro_Estimacion como ejemplo
    supabase.table("parametro_estimacion").update({"factor_ia": correction_factor}).eq("parametro_estimacionid", 1).execute()

    return True