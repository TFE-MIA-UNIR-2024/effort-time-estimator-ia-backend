import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from supabase_client import get_supabase_client
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tabulate import tabulate

# Nombre del archivo para guardar el modelo y el StandardScaler
MODEL_FILE = "effort_model.keras"
SCALER_FILE = "effort_scaler.joblib"
COLUMNS_FILE = "effort_columns.joblib"  # Nuevo archivo para guardar los nombres de las columnas

def fetch_data():
    supabase = get_supabase_client()

    tipo_elementos = supabase.from_("tipo_elemento_afectado").select("tipo_elemento_afectadoid, nombre, activo, fase_proyectoid").execute().data

    response = supabase.from_("punto_funcion").select(
        "punto_funcionid, cantidad_estimada, cantidad_real, \
        requerimientoid, tipo_elemento_afectado_id"
    ).not_.is_("tipo_elemento_afectado_id", None).execute()

    data = response.data
    df = pd.DataFrame(data)

    return df, tipo_elementos

def preprocess_data(df, num_tipos_elemento):
    df['tipo_elemento_afectado_id'] = pd.to_numeric(df['tipo_elemento_afectado_id'], errors='coerce').fillna(0).astype(int)

    # One-Hot Encoding para tipo_elemento_afectado_id
    one_hot = pd.get_dummies(df['tipo_elemento_afectado_id'], prefix='tipo')

    # Agregar columnas one-hot al dataframe original
    df = pd.concat([df, one_hot], axis=1)

    # Seleccionar las características para el modelo
    feature_columns = ['cantidad_estimada'] + [col for col in df.columns if col.startswith('tipo_')]
    X = df[feature_columns].fillna(0)  # Usar el DataFrame directamente

    # Dividir en conjuntos de entrenamiento y prueba (antes de escalar)
    X_train, X_test, y_train, y_test = train_test_split(X, df["cantidad_real"].fillna(0), test_size=0.2, random_state=42)

    # Escalar solo las características de entrenamiento y luego aplicar la misma transformación a las de prueba
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, len(feature_columns), scaler, X.columns # Retornar el scaler

def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='relu') # ReLU para predicciones no negativas
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_and_save_model():
    df, tipo_elementos = fetch_data()
    num_tipos_elemento = len(tipo_elementos)

    if df.empty:
        print("No se recibieron datos para entrenar el modelo. Saliendo...")
        return

    # Preprocesar y dividir los datos
    X_train, X_test, y_train, y_test, input_shape, scaler, train_columns = preprocess_data(df, num_tipos_elemento)

    model = create_model(input_shape)
    print(f"Input shape del modelo: {input_shape}")

    try:
        model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=1)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return

    # Guardar el modelo
    model.save(MODEL_FILE)
    print(f"Modelo guardado en {MODEL_FILE}")

    # Guardar el StandardScaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"StandardScaler guardado en {SCALER_FILE}")

    # Guardar los nombres de las columnas de entrenamiento
    joblib.dump(train_columns, COLUMNS_FILE)
    print(f"Columnas de entrenamiento guardadas en {COLUMNS_FILE}")

    return df, tipo_elementos

def load_model_and_predict(df, data):
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        train_columns = joblib.load(COLUMNS_FILE) # Cargar los nombres de las columnas
    except Exception as e:
        print(f"Error al cargar el modelo, el StandardScaler o las columnas: {e}")
        return None

    # Crear DataFrame a partir de los datos de entrada
    input_df = pd.DataFrame([data])

    # One-Hot Encode la entrada
    one_hot = pd.get_dummies(input_df['tipo_elemento_afectado_id'], prefix='tipo')
    input_df = pd.concat([input_df, one_hot], axis=1)

    # Reindexar el input_df para que tenga las mismas columnas que el conjunto de entrenamiento
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # Asegurarse de que 'cantidad_estimada' esté presente (por si acaso)
    if 'cantidad_estimada' not in input_df.columns:
        input_df['cantidad_estimada'] = 0  # o algún valor predeterminado

    # Mover 'cantidad_estimada' al principio del dataframe
    columns = ['cantidad_estimada'] + [col for col in input_df.columns if col != 'cantidad_estimada']
    input_df = input_df[columns]


    # Seleccionar las características (en el mismo orden que en el entrenamiento)
    X = input_df.values


    # Escalar la entrada
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled).flatten()
    prediction = np.round(prediction).astype(int)

    return prediction[0], input_df

def main():
  # Entrenar y guardar el modelo
  df, tipo_elementos = train_and_save_model()

  # Ejemplo de uso del modelo cargado para hacer una predicción:
  input_data = {'cantidad_estimada': 2, 'tipo_elemento_afectado_id': 3}

  prediction, input_df = load_model_and_predict(df, input_data)

  if prediction is not None:
      # Crear DataFrame para la entrada y la predicción
      output_data = {'cantidad_estimada': [input_data['cantidad_estimada']],
                     'tipo_elemento_afectado_id': [input_data['tipo_elemento_afectado_id']],
                     'Predicción': [prediction]}

      output_df = pd.DataFrame(output_data)
      print("Predicción:")
      print(tabulate(output_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))

  else:
    print("No se pudo realizar la predicción.")

if __name__ == '__main__':
    main()