from fastapi import FastAPI, Body
from pydantic import BaseModel
import os
import traceback  # Importa la biblioteca traceback para ver la traza de los errores

from train_model import train_and_update_factors
from model_inference import load_model, predict_effort

app = FastAPI()
model = None

class PredictionRequest(BaseModel):
    input_data: list

@app.on_event("startup")
def on_startup():
    global model
    model = load_model()
    if model:
        print("Modelo cargado exitosamente en el inicio.")
    else:
       print("Error al cargar el modelo en el inicio.")


@app.post("/train")
def api_train():
    """
    Entrena la RNA con datos históricos y actualiza factores en la BD.
    """
    success = train_and_update_factors()
    return {"message": "Entrenamiento finalizado", "success": success}

@app.post("/predict")
def api_predict(request: PredictionRequest):
    """
    Retorna el esfuerzo estimado basándose en la cantidad de objetos y factor_ia (si aplica).
    """
    global model
    if not model:
        return {"error": "Modelo no cargado. Ejecuta /train o reinicia la app.", "status_code": 500}

    try:
        input_vec = request.input_data
        print(f"Datos de entrada a predict_effort: {input_vec}")
        est = predict_effort(model, input_vec)
        print(f"Resultado de predict_effort: {est}")
        return {"esfuerzo_estimado": est}
    except Exception as e:
        trace = traceback.format_exc()  # Obtiene la traza del error
        print(f"Error en la función predict: {e}\nTraceback:\n{trace}")
        return {"error": f"Error en la predicción: {e}", "status_code": 500, "traceback": trace}

@app.get("/")
def root():
    return {"message": "API de estimación de esfuerzo funcionando - RNA MLP"}