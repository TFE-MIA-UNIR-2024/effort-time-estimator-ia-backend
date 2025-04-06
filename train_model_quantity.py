import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from supabase_client import get_supabase_client
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
from tabulate import tabulate

MODEL_FILE = "effort_model.keras"
SCALER_FILE = "effort_scaler.joblib"
COLUMNS_FILE = "effort_columns.joblib"


def fetch_data():
    supabase = get_supabase_client()

    response_pf = supabase.from_("punto_funcion").select(
        "punto_funcionid, cantidad_estimada, cantidad_real, requerimientoid, tipo_elemento_afectado_id, parametro_estimacionid"
    ).execute()
    df_pf = pd.DataFrame(response_pf.data)

    response_param = supabase.from_("parametro_estimacion").select(
        "parametro_estimacionid, tipo_parametro_estimacionid"
    ).execute()
    df_param = pd.DataFrame(response_param.data)

    df = df_pf.merge(df_param, how='left', on='parametro_estimacionid')

    if df.empty:
        print("No se recibieron datos para entrenar el modelo. Saliendo...")
        return None

    return df


def preprocess_data(df):
    df = df.dropna(subset=['cantidad_estimada', 'cantidad_real'])

    # Separar los registros de parámetros y elementos afectados
    df_param = df[df['tipo_parametro_estimacionid'].notnull()].copy()
    df_elem = df[df['tipo_elemento_afectado_id'].notnull()].copy()

    # Pivotear los elementos afectados para que cada tipo sea una columna
    df_elem_pivot = df_elem.pivot_table(
        index='requerimientoid',
        columns='tipo_elemento_afectado_id',
        values='cantidad_estimada',
        aggfunc='sum',
        fill_value=0
    )
    df_elem_pivot.columns = [f'elem_afectado_{int(col)}' for col in df_elem_pivot.columns]

    # One-hot encoding de los parámetros de estimación
    df_param['cantidad_estimada'] = df_param['cantidad_estimada'].fillna(0)
    df_param['tipo_parametro_estimacionid'] = df_param['tipo_parametro_estimacionid'].astype(str)
    df_param['param_type_value'] = df_param['tipo_parametro_estimacionid'] + '_' + df_param['cantidad_estimada'].astype(int).astype(str)

    df_param['dummy'] = 1
    df_param_pivot = df_param.pivot_table(
        index='requerimientoid',
        columns='param_type_value',
        values='dummy',
        aggfunc='sum',
        fill_value=0
    )

    # Calcular cantidad_real total por requerimiento como target
    df_target = df.groupby('requerimientoid')['cantidad_real'].sum().to_frame(name='cantidad_real_total')

    # Unir todo en un solo dataframe
    df_final = df_target.join(df_elem_pivot, how='left').join(df_param_pivot, how='left').fillna(0)

    X = df_final.drop(columns=['cantidad_real_total'])
    y = df_final['cantidad_real_total']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X.shape[1], scaler, X.columns


def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_and_save_model():
    df = fetch_data()
    if df is None:
        return None

    print("DataFrame original:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))

    X_train, X_test, y_train, y_test, input_shape, scaler, train_columns = preprocess_data(df)

    model = create_model(input_shape)
    print(f"Input shape del modelo: {input_shape}")

    try:
        model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=1)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return None

    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(train_columns, COLUMNS_FILE)

    print("Modelo, scaler y columnas guardados correctamente.")
    return df


def main():
    df = train_and_save_model()
    if df is None:
        return


if __name__ == '__main__':
    main()
