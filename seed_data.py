# seed_data.py

from supabase_client import get_supabase_client
import datetime

def seed_data():
    supabase = get_supabase_client()

    # 1. Parametro_Estimacion
    supabase.table("Parametro_Estimacion").insert([
        {
            "nombre": "Factor IA inicial",
            "descripcion": "Factor automático de la IA",
            "factor": 1.0,
            "factor_ia": 1.0,
            "fecha_de_creacion": datetime.datetime.now().isoformat(),
            "pesoFactor": 1.0
        },
        {
            "nombre": "Factor IA secundario",
            "descripcion": "Otro factor",
            "factor": 1.2,
            "factor_ia": 1.2,
            "fecha_de_creacion": datetime.datetime.now().isoformat(),
            "pesoFactor": 1.1
        }
    ]).execute()

    # 2. Estimacion_Esfuerzo_Construccion
    supabase.table("Estimacion_Esfuerzo_Construccion").insert([
        {
            "objeto_afectado": "FormularioX",
            "cantidad_objeto_estimado": 4,
            "cantidad_objeto_real": 5,
            "esfuerzo_adicional": 1,
            "justificacion_EsfuerzoAdicional": "Ajuste de diseño",
            "esfuerzo_real": 12,
            "fechaCreacion": datetime.datetime.now().isoformat()
        },
        {
            "objeto_afectado": "ReporteZ",
            "cantidad_objeto_estimado": 2,
            "cantidad_objeto_real": 2,
            "esfuerzo_adicional": 0,
            "justificacion_EsfuerzoAdicional": "N/A",
            "esfuerzo_real": 8,
            "fechaCreacion": datetime.datetime.now().isoformat()
        },
        {
            "objeto_afectado": "API_A",
            "cantidad_objeto_estimado": 6,
            "cantidad_objeto_real": 7,
            "esfuerzo_adicional": 2,
            "justificacion_EsfuerzoAdicional": "Cambios de última hora",
            "esfuerzo_real": 15,
            "fechaCreacion": datetime.datetime.now().isoformat()
        }
    ]).execute()

    return True

if __name__ == "__main__":
    seed_data()
    print("Data de prueba insertada.")
