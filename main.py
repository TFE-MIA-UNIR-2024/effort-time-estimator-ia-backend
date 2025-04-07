from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import traceback
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from train_model_complete import train_and_adjust_factors
from train_model_quantity import train_quantity_model  # También se entrena el modelo de cantidades

MODEL_FILE = "effort_model.keras"
SCALER_FILE = "effort_scaler.joblib"
COLUMNS_FILE = "effort_columns.joblib"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
train_columns = None


class PredictionRequest(BaseModel):
    parametro_estimacion_ids: List[int]
    tipo_elemento_afectado_ids: List[int]


@app.on_event("startup")
def load_resources():
    global model, scaler, train_columns
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        train_columns = joblib.load(COLUMNS_FILE)
        print("Modelo y recursos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar modelo o recursos: {e}")


@app.post("/predict")
def predict(request: PredictionRequest):
    global model, scaler, train_columns

    if model is None or scaler is None or train_columns is None or len(train_columns) == 0:
        return {"error": "Modelo o recursos no cargados.", "status_code": 500}

    try:
        inputs = []

        for tipo_id in request.tipo_elemento_afectado_ids:
            input_data = {col: 0 for col in train_columns}

            col_name = f"elem_afectado_{tipo_id}"
            if col_name in input_data:
                input_data[col_name] = 1

            for param_id in request.parametro_estimacion_ids:
                for col in train_columns:
                    if col.startswith(f"{param_id}_"):
                        input_data[col] = 1

            inputs.append(input_data)

        input_df = pd.DataFrame(inputs)
        input_scaled = scaler.transform(input_df)
        predictions = model.predict(input_scaled).flatten()

        results = []
        for tipo_id, pred in zip(request.tipo_elemento_afectado_ids, predictions):
            results.append({
                "tipo_elemento_afectado_id": tipo_id,
                "cantidad_estimada_predicha": int(round(float(pred)))
            })

        return {"predicciones": results}

    except Exception as e:
        trace = traceback.format_exc()
        return {"error": str(e), "traceback": trace, "status_code": 500}


@app.post("/train")
def api_train():
    try:
        import io
        import sys

        output = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = output

        print("Entrenando modelo de factores IA (parámetros y elementos afectados)...")
        train_and_adjust_factors()

        print("Entrenando modelo de estimación de cantidad de elementos por requerimiento...")
        cantidad_resultado = train_quantity_model()

        sys.stdout = sys_stdout
        factores_logs = output.getvalue().splitlines()

        return {
            "message": "Entrenamiento finalizado con éxito",
            "factores_logs": factores_logs,
            "cantidad_logs": cantidad_resultado.get("logs", []),
            "cantidad_metrics": cantidad_resultado.get("metrics", {})
        }
    except Exception as e:
        sys.stdout = sys_stdout
        trace = traceback.format_exc()
        return {"error": str(e), "traceback": trace, "status_code": 500}


@app.get("/")
def root():
    return {"message": "API de estimación por requerimiento con elementos afectados y parámetros"}