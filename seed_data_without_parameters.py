import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from supabase_client import get_supabase_client

def fetch_historical_data():
    """
    Lee la tabla histórica con datos por requerimiento desde Supabase.
    """
    supabase = get_supabase_client()
    query = """
        SELECT 
            eec.estimacion_esfuerzo_construccionid,
            eec.objeto_afectado,
            eec.cantidad_objeto_estimado,
            eec.cantidad_objeto_real,
            eec.esfuerzo_real,
            eec.fechacreacion,
            pf.parametro_estimacionid,
            pf.cantidad_real as cantidad_real_punto_funcion,
            pe.factor as factor_inicial
        FROM estimacion_esfuerzo_construccion eec
        LEFT JOIN punto_funcion pf ON eec.punto_funcionid = pf.punto_funcionid
        LEFT JOIN parametro_estimacion pe ON pf.parametro_estimacionid = pe.parametro_estimacionid;
    """
    response = supabase.from_sql(query).execute()
    data = response.data
    df = pd.DataFrame(data)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    print("Primeras filas del dataset:")
    print(df.head())
    
    # Imprimir las columnas del DataFrame para depuración
    print("Columnas del DataFrame:")
    print(df.columns)

    return df

def preprocess_data_for_requirement(df, parametro_estimacionid):
    """
    Filtra los datos para un requerimiento específico.
    """
    # Filtrar datos por parametro_estimacionid
    df_req = df[df["parametro_estimacionid"] == parametro_estimacionid]

    if df_req.empty:
        print(f"No se encontraron datos para parametro_estimacionid: {parametro_estimacionid}")
        return None, None

    # Crear X (features) y y (etiquetas)
    X = df_req[["cantidad_objeto_estimado", "cantidad_objeto_real", "cantidad_real_punto_funcion"]].values
    y = df_req["esfuerzo_real"].values

    print(f"Datos para parametro_estimacionid {parametro_estimacionid}: {X.shape[0]} filas.")
    print(f"Primeras filas de X para parametro_estimacionid {parametro_estimacionid}:")
    print(X[:5])
    print(f"Primeros valores de y:")
    print(y[:5])

    return X, y

def create_model():
    """
    Define la red neuronal para ajustar el factor_ia.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(3,)),  # Entrada: 3 features
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')  # Salida: factor_ia ajustado
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()  # Mostrar resumen del modelo
    return model

def train_for_all_requirements():
    """
    Entrena el modelo para cada requerimiento y actualiza los factores_ia en la base de datos.
    """
    # Obtener datos históricos
    df = fetch_historical_data()
    
    # Obtener todos los parametro_estimacionid únicos
    unique_param_ids = df["parametro_estimacionid"].unique()
    print(f"Encontrados {len(unique_param_ids)} parametro_estimacionid únicos.")

    # Estructura para almacenar los nuevos factores
    factor_ia_updates = []

     # Entrenamiento por cada requerimiento
    for param_id in unique_param_ids:
        print(f"\nEntrenando para parametro_estimacionid: {param_id}")
        
        # Filtrar datos para este requerimiento
        X, y = preprocess_data_for_requirement(df, param_id)

        # Verificar si X o y están vacíos
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            print(f"Saltando parametro_estimacionid {param_id} por falta de datos.")
            continue

        # Crear y entrenar el modelo
        model = create_model()

        try:
            history = model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)
        except Exception as e:
            print(f"Error durante el entrenamiento para parametro_estimacionid {param_id}: {e}")
            continue

        # Predecir el esfuerzo con el modelo entrenado
        predictions = model.predict(X).flatten()

        # Manejo de NaN en predicciones
        if np.isnan(predictions).any():
            print(f"ADVERTENCIA: Predicciones contienen NaN para parametro_estimacionid {param_id}. Saltando el cálculo del factor_ia.")
            continue

        # Calcular el nuevo factor_ia para este requerimiento
        mean_cantidad_real = X[:, 1].mean()
        if mean_cantidad_real == 0:
           print(f"ADVERTENCIA: cantidad_objeto_real media es 0 para parametro_estimacionid {param_id}. Usando factor_ia predeterminado.")
           factor_ia_new = 1  # O algún otro valor predeterminado
        else:    
            factor_ia_new = predictions.mean() / X[:,1].mean()  # Ajustar según lógica específica
            print(f"Nuevo factor_ia calculado para parametro_estimacionid {param_id}: {factor_ia_new}")


        # Almacenar el nuevo factor_ia en la lista temporal
        factor_ia_updates.append({"parametro_estimacionid": param_id, "factor_ia": factor_ia_new})

    # Actualizar todos los factores en la base de datos al final
    supabase = get_supabase_client()
    for update in factor_ia_updates:
        try:
            supabase.table("parametro_estimacion").update(
                {"factor_ia": update["factor_ia"]}
            ).eq("parametro_estimacionid", update["parametro_estimacionid"]).execute()
            print(f"Actualizado factor_ia en la base de datos para parametro_estimacionid {update['parametro_estimacionid']}.")
        except Exception as e:
            print(f"Error al actualizar la base de datos para parametro_estimacionid {update['parametro_estimacionid']}: {e}")

    print("\nEntrenamiento completado para todos los requerimientos.")


if __name__ == '__main__':
    train_for_all_requirements()