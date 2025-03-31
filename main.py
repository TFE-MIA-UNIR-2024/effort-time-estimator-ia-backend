from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import traceback
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from train_model import train_and_update_factors  # Tu lógica de entrenamiento

MODEL_FILE = "effort_model.keras"
SCALER_FILE = "effort_scaler.joblib"
COLUMNS_FILE = "effort_columns.joblib"

app = FastAPI()
model = None
scaler = None
train_columns = None


class PredictionRequest(BaseModel):
    tipo_elemento_afectado_ids: List[int]


@app.on_event("startup")
def load_resources():
    global model, scaler, train_columns
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        train_columns = joblib.load(COLUMNS_FILE)
        print("✅ Modelo y recursos cargados correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar modelo o recursos: {e}")


@app.post("/predict")
def predict(request: PredictionRequest):
    global model, scaler, train_columns

    if model is None or scaler is None or train_columns is None or len(train_columns) == 0:
        return {"error": "Modelo o recursos no cargados.", "status_code": 500}

    try:
        # Construir DataFrame con cantidad_estimada = 1
        data = [{"cantidad_estimada": 1, "tipo_elemento_afectado_id": tipo_id}
                for tipo_id in request.tipo_elemento_afectado_ids]
        input_df = pd.DataFrame(data)

        # One-hot
        one_hot = pd.get_dummies(input_df["tipo_elemento_afectado_id"], prefix="tipo")
        input_df = pd.concat([input_df, one_hot], axis=1)
        input_df = input_df.reindex(columns=train_columns, fill_value=0)
        input_df = input_df[["cantidad_estimada"] + [col for col in input_df.columns if col != "cantidad_estimada"]]

        # Escalar
        input_scaled = scaler.transform(input_df)

        # Predicción
        predictions = model.predict(input_scaled).flatten()
        predictions = np.round(predictions).astype(int)

        # Formar respuesta
        results = [
            {
                "tipo_elemento_afectado_id": tipo_id,
                "cantidad_estimada_predicha": int(pred)
            }
            for tipo_id, pred in zip(request.tipo_elemento_afectado_ids, predictions)
        ]

        return {"predicciones": results}

    except Exception as e:
        trace = traceback.format_exc()
        return {"error": str(e), "traceback": trace, "status_code": 500}


@app.post("/train")
def api_train():
    """
    Reentrena el modelo y actualiza factores en la base de datos.
    """
    try:
        success = train_and_update_factors()
        return {"message": "Entrenamiento finalizado", "success": success}
    except Exception as e:
        trace = traceback.format_exc()
        return {"error": str(e), "traceback": trace, "status_code": 500}


@app.get("/")
def root():
    return {"message": "API de estimación de cantidad estimada (por tipo de elemento afectado) y entrenamiento manual"}
