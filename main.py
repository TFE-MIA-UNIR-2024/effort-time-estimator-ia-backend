# main.py

from fastapi import FastAPI, Body
import os

from train_model import train_and_update_factors
from model_inference import load_model, predict_effort

app = FastAPI()
model = None

@app.on_event("startup")
def on_startup():
    global model
    model = load_model()

@app.post("/train")
def api_train():
    """
    Entrena la RNA con datos históricos y actualiza factores en la BD.
    """
    success = train_and_update_factors()
    return {"message": "Entrenamiento finalizado", "success": success}

@app.post("/predict")
def api_predict(cantidad_objeto: float = Body(...), factor_ia: float = Body(1.0)):
    """
    Retorna el esfuerzo estimado basándose en la cantidad de objetos y factor_ia (si aplica).
    """
    global model
    if not model:
        return {"error": "Modelo no cargado. Ejecuta /train o reinicia la app."}

    input_vec = [cantidad_objeto, factor_ia]
    est = predict_effort(model, input_vec)
    return {"esfuerzo_estimado": est}

@app.get("/")
def root():
    return {"message": "API de estimación de esfuerzo funcionando - RNA MLP"}
