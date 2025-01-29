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
    
    response = supabase.from_("estimacion_esfuerzo_construccion").select(
        """
            estimacion_esfuerzo_construccionid,
            objeto_afectado,
            cantidad_objeto_estimado,
            cantidad_objeto_real,
            esfuerzo_real,
            fechacreacion,
            punto_funcion(parametro_estimacionid, cantidad_real, parametro_estimacion(factor))
        """
    ).execute()
    
    data = response.data

    if data and isinstance(data, list) and len(data) > 0:
        
        # Procesar los datos para obtener el formato deseado
        processed_data = []
        for row in data:
          
          processed_row = {
              'estimacion_esfuerzo_construccionid': row.get('estimacion_esfuerzo_construccionid'),
              'objeto_afectado': row.get('objeto_afectado'),
              'cantidad_objeto_estimado': row.get('cantidad_objeto_estimado'),
              'cantidad_objeto_real': row.get('cantidad_objeto_real'),
              'esfuerzo_real': row.get('esfuerzo_real'),
              'fechacreacion': row.get('fechacreacion'),
          }
          if row.get('punto_funcion') is not None:
             processed_row['parametro_estimacionid'] = row['punto_funcion'].get('parametro_estimacionid')
             processed_row['cantidad_real_punto_funcion'] = row['punto_funcion'].get('cantidad_real')
             if row['punto_funcion'].get('parametro_estimacion') is not None:
                processed_row['factor_inicial'] = row['punto_funcion']['parametro_estimacion'].get('factor')
             else:
                processed_row['factor_inicial'] = None
          else:
            processed_row['parametro_estimacionid'] = None
            processed_row['cantidad_real_punto_funcion'] = None
            processed_row['factor_inicial'] = None


          processed_data.append(processed_row)
        
        df = pd.DataFrame(processed_data)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        print("Primeras filas del dataset:")
        print(df.head())
        # Imprimir las columnas del DataFrame para depuración
        print("Columnas del DataFrame:")
        print(df.columns)
        return df
    else:
        print("Error: No se recibieron datos de la base de datos")
        return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocesa todos los datos para el entrenamiento.
    """
    # Crear X (features) y y (etiquetas)
    X = df[["cantidad_objeto_estimado", "cantidad_objeto_real", "cantidad_real_punto_funcion"]].values
    y = df["esfuerzo_real"].values

    print(f"Datos para entrenamiento global: {X.shape[0]} filas.")
    print(f"Primeras filas de X para entrenamiento global:")
    print(X[:5])
    print(f"Primeros valores de y para entrenamiento global:")
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
    Entrena el modelo con todos los requerimientos y actualiza los factores_ia en la base de datos.
    """
    # Obtener datos históricos
    df = fetch_historical_data()

    # Verificar que el DataFrame no esté vacío
    if df.empty:
      print("No se recibieron datos para entrenar el modelo. Saliendo...")
      return

    # Preprocesar todos los datos
    X, y = preprocess_data(df)

    # Crear y entrenar el modelo
    model = create_model()

    try:
        history = model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)
    except Exception as e:
        print(f"Error durante el entrenamiento global: {e}")
        return

    # Predecir el esfuerzo con el modelo entrenado
    predictions = model.predict(X).flatten()

    # Manejo de NaN en predicciones
    if np.isnan(predictions).any():
        print(f"ADVERTENCIA: Predicciones contienen NaN. Saltando el cálculo del factor_ia.")
        return

    # Estructura para almacenar los nuevos factores
    factor_ia_updates = []

    # Obtener todos los parametro_estimacionid únicos
    unique_param_ids = df["parametro_estimacionid"].unique()
    print(f"Encontrados {len(unique_param_ids)} parametro_estimacionid únicos.")
    
    # Calcular y almacenar el nuevo factor_ia para cada requerimiento
    for param_id in unique_param_ids:
        df_req = df[df["parametro_estimacionid"] == param_id]
        X_req = df_req[["cantidad_objeto_estimado", "cantidad_objeto_real", "cantidad_real_punto_funcion"]].values

        mean_cantidad_real = X_req[:, 1].mean()
        if mean_cantidad_real == 0:
           print(f"ADVERTENCIA: cantidad_objeto_real media es 0 para parametro_estimacionid {param_id}. Usando factor_ia predeterminado.")
           factor_ia_new = 1  # O algún otro valor predeterminado
        else:
            # Obtener predicciones correspondientes al parámetro de estimación actual
            indices = df[df["parametro_estimacionid"] == param_id].index
            predictions_param = predictions[indices]
            factor_ia_new = predictions_param.mean() / mean_cantidad_real  # Ajustar según lógica específica
            print(f"Nuevo factor_ia calculado para parametro_estimacionid {param_id}: {factor_ia_new}")
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