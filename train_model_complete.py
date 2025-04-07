import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend as K
from supabase_client import get_supabase_client
from tabulate import tabulate

# Función de pérdida basada en el Error Relativo Absoluto
def relative_error_loss(y_true, y_pred):
    return K.mean(K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None)))

# ------------------------------------------------------------------------------
# Funciones para el modelo de Parametro de Estimación
# ------------------------------------------------------------------------------

def fetch_historical_data_parametro():
    supabase = get_supabase_client()
    response = supabase.from_("punto_funcion").select(
        """
            punto_funcionid,
            jornada_real,
            jornada_estimada,
            parametro_estimacionid,
            parametro_estimacion(factor, factor_ia)
        """
    ).filter("jornada_real", "not.is", "null").filter("jornada_estimada", "not.is", "null").execute()  # Condición

    data = response.data
    processed_data = []

    if data:
        for row in data:
            parametro_estimacion_data = row.get('parametro_estimacion')

            processed_row = {
                'punto_funcionid': row.get('punto_funcionid'),  # Agregado punto_funcionid
                'jornada_real': row.get('jornada_real'),
                'jornada_estimada': row.get('jornada_estimada'),
                # No convertir aquí, dejar como está
                'parametro_estimacionid': row.get('parametro_estimacionid'),
                'factor_parametro': parametro_estimacion_data.get('factor') if parametro_estimacion_data else None,
                'factor_ia_parametro': parametro_estimacion_data.get('factor_ia') if parametro_estimacion_data else None,
            }
            processed_data.append(processed_row)

    df = pd.DataFrame(processed_data)
    print("\nDataFrame Parametro:")
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # Verifica los tipos de datos
    print("\nTipos de datos del DataFrame Parametro:")
    print(df.dtypes)

    # Convierte las columnas a numérico y maneja los errores
    df["factor_parametro"] = pd.to_numeric(df["factor_parametro"], errors='coerce')
    df["factor_ia_parametro"] = pd.to_numeric(df["factor_ia_parametro"], errors='coerce')

    # Imputa los valores faltantes (NaN) con 0
    df["factor_parametro"] = df["factor_parametro"].fillna(0)
    df["factor_ia_parametro"] = df["factor_ia_parametro"].fillna(0)

    # Forzar que parametro_estimacionid sea entero o NaN
    df["parametro_estimacionid"] = pd.to_numeric(df["parametro_estimacionid"], errors='coerce', downcast='integer')

    return df


def preprocess_data_parametro(df):
    X = df[["jornada_real", "jornada_estimada", "factor_parametro", "factor_ia_parametro"]].values
    y = df["jornada_real"].values
    return X, y


def create_model_parametro():
    model = tf.keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=relative_error_loss, metrics=['mae'])
    return model

def train_and_adjust_factors_parametro():
    df = fetch_historical_data_parametro()
    if df.empty:
        print("No hay datos para entrenar el modelo de Parametro.")
        return

    print("Datos de entrada (Parametro):")
    print(tabulate(df, headers='keys', tablefmt='psql'))

    X, y = preprocess_data_parametro(df)
    model = create_model_parametro()
    model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)

    predictions = model.predict(X).flatten()
    supabase = get_supabase_client()

    for param_id in df["parametro_estimacionid"].unique():
        # Verificar si param_id es NaN o no es un entero válido
        if pd.isna(param_id) or not isinstance(param_id, (int, np.integer)):
            print(f"Advertencia: parametro_estimacionid inválido ({param_id}). Omitiendo actualización.")
            continue

        param_id = int(param_id)  # Asegurar que param_id sea un entero
        df_param = df[df["parametro_estimacionid"] == param_id]
        indices = df_param.index
        esfuerzo_estimado = predictions[indices]
        esfuerzo_real = df_param["jornada_real"].values

        # Verifica si hay NaN en esfuerzo_estimado o esfuerzo_real
        if np.isnan(esfuerzo_estimado).any() or np.isnan(esfuerzo_real).any():
            print(f"Advertencia: NaN encontrado en esfuerzo_estimado o esfuerzo_real para parametro_estimacionid {param_id}. Omitiendo actualización.")
            continue

        mean_esfuerzo_real = np.mean(esfuerzo_real)

        if mean_esfuerzo_real == 0:
            factor_ia_new = 1
        else:
            factor_ia_new = np.mean(esfuerzo_estimado) / mean_esfuerzo_real

        factor_ia_new = max(factor_ia_new, 0)

        if factor_ia_new > 10:
            factor_ia_new = 10

        # Verifica si factor_ia_new es NaN después de los cálculos
        if np.isnan(factor_ia_new):
            print(f"Advertencia: factor_ia_new es NaN para parametro_estimacionid {param_id}. Omitiendo actualización.")
            continue

        supabase.table("parametro_estimacion").update(
            {"factor_ia": factor_ia_new}
        ).eq("parametro_estimacionid", int(param_id)).execute()  # Asegurar que param_id sea entero

        print(f"Factor_ia (Parametro) actualizado para parametro_estimacionid {param_id}: {factor_ia_new}")

# ------------------------------------------------------------------------------
# Funciones para el modelo de Elemento Afectado
# ------------------------------------------------------------------------------

def fetch_historical_data_elemento():
    supabase = get_supabase_client()
    response = supabase.from_("punto_funcion").select(
        """
            punto_funcionid,
            jornada_real,
            jornada_estimada,
            tipo_elemento_afectado_id,
            tipo_elemento_afectado(elemento_afectado(factor, factor_ia))
        """
    ).filter("jornada_real", "not.is", "null").filter("jornada_estimada", "not.is", "null").execute()  # Condición

    data = response.data
    processed_data = []

    if data:
        for row in data:
            tipo_elemento_afectado_data = row.get('tipo_elemento_afectado')
            if tipo_elemento_afectado_data is not None:
                elemento_afectado_list = tipo_elemento_afectado_data.get('elemento_afectado')
                if isinstance(elemento_afectado_list, list) and elemento_afectado_list:
                    elemento_afectado_data = elemento_afectado_list[0]
                else:
                    elemento_afectado_data = None
            else:
                elemento_afectado_data = None

            processed_row = {
                'punto_funcionid': row.get('punto_funcionid'),  # Agregado punto_funcionid
                'jornada_real': row.get('jornada_real'),
                'jornada_estimada': row.get('jornada_estimada'),
                'tipo_elemento_afectado_id': row.get('tipo_elemento_afectado_id'),
                'factor_elemento': elemento_afectado_data.get('factor') if elemento_afectado_data else None,
                'factor_ia_elemento': elemento_afectado_data.get('factor_ia') if elemento_afectado_data else None,
            }
            processed_data.append(processed_row)

    df = pd.DataFrame(processed_data)
    print("\nDataFrame Elemento:")
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # Verifica los tipos de datos
    print("\nTipos de datos del DataFrame Elemento:")
    print(df.dtypes)

    # Convierte las columnas a numérico y maneja los errores
    df["factor_elemento"] = pd.to_numeric(df["factor_elemento"], errors='coerce')
    df["factor_ia_elemento"] = pd.to_numeric(df["factor_ia_elemento"], errors='coerce')

    # Imputa los valores faltantes (NaN) con 0
    df["factor_elemento"] = df["factor_elemento"].fillna(0)
    df["factor_ia_elemento"] = df["factor_ia_elemento"].fillna(0)

    # Forzar que tipo_elemento_afectado_id sea entero o NaN
    df["tipo_elemento_afectado_id"] = pd.to_numeric(df["tipo_elemento_afectado_id"], errors='coerce', downcast='integer')

    return df


def preprocess_data_elemento(df):
    X = df[["jornada_real", "jornada_estimada", "factor_elemento", "factor_ia_elemento"]].values
    y = df["jornada_real"].values
    return X, y


def create_model_elemento():
    model = tf.keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=relative_error_loss, metrics=['mae'])
    return model

def train_and_adjust_factors_elemento():
    df = fetch_historical_data_elemento()
    if df.empty:
        print("No hay datos para entrenar el modelo de Elemento.")
        return

    print("Datos de entrada (Elemento):")
    print(tabulate(df, headers='keys', tablefmt='psql'))

    X, y = preprocess_data_elemento(df)
    model = create_model_elemento()
    model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)

    predictions = model.predict(X).flatten()
    supabase = get_supabase_client()

    # Supongamos que deseas actualizar el factor_ia en la tabla elemento_afectado
    # Necesitas obtener el elemento_afectadoid correspondiente al tipo_elemento_afectado_id
    for tipo_elemento_afectado_id in df["tipo_elemento_afectado_id"].unique():  # Usar tipo_elemento_afectado_id
        # Verificar si tipo_elemento_afectado_id es NaN o no es un entero válido
        if pd.isna(tipo_elemento_afectado_id) or not isinstance(tipo_elemento_afectado_id, (int, np.integer)):
            print(f"Advertencia: tipo_elemento_afectado_id inválido ({tipo_elemento_afectado_id}). Omitiendo actualización.")
            continue

        tipo_elemento_afectado_id = int(tipo_elemento_afectado_id)  # Asegurar que tipo_elemento_afectado_id sea un entero
        df_elemento = df[df["tipo_elemento_afectado_id"] == tipo_elemento_afectado_id] # Filtrar por tipo_elemento_afectado_id
        indices = df_elemento.index
        esfuerzo_estimado = predictions[indices]
        esfuerzo_real = df_elemento["jornada_real"].values

        # Verifica si hay NaN en esfuerzo_estimado o esfuerzo_real
        if np.isnan(esfuerzo_estimado).any() or np.isnan(esfuerzo_real).any():
            print(f"Advertencia: NaN encontrado en esfuerzo_estimado o esfuerzo_real para tipo_elemento_afectado_id {tipo_elemento_afectado_id}.  Omitiendo actualización.")
            continue  # Salta a la siguiente iteración del bucle

        mean_esfuerzo_real = np.mean(esfuerzo_real)

        if mean_esfuerzo_real == 0:
            factor_ia_new = 1
        else:
            factor_ia_new = np.mean(esfuerzo_estimado) / mean_esfuerzo_real

        factor_ia_new = max(factor_ia_new, 0)

        if factor_ia_new > 10:
            factor_ia_new = 10

         # Verifica si factor_ia_new es NaN después de los cálculos
        if np.isnan(factor_ia_new):
            print(f"Advertencia: factor_ia_new es NaN para tipo_elemento_afectado_id {tipo_elemento_afectado_id}.  Omitiendo actualización.")
            continue #Salto a la siguiente iteración del ciclo


        # Ahora busca el elemento_afectado asociado con este tipo_elemento_afectado_id
        response = supabase.from_("elemento_afectado").select("elemento_afectadoid").eq("tipo_elemento_afectadoid", tipo_elemento_afectado_id).limit(1).execute()
        elemento_afectado_data = response.data

        if elemento_afectado_data:
            elemento_afectadoid = elemento_afectado_data[0].get("elemento_afectadoid")

            supabase.table("elemento_afectado").update(
                {"factor_ia": factor_ia_new}
            ).eq("elemento_afectadoid", elemento_afectadoid).execute()

            print(f"Factor_ia (Elemento) actualizado para elemento_afectadoid {elemento_afectadoid}: {factor_ia_new}")
        else:
            print(f"No se encontró elemento_afectado para tipo_elemento_afectado_id {tipo_elemento_afectado_id}")


# ------------------------------------------------------------------------------
# Función principal
# ------------------------------------------------------------------------------

def train_and_adjust_factors():
    train_and_adjust_factors_parametro()
    train_and_adjust_factors_elemento()

if __name__ == '__main__':
    train_and_adjust_factors()