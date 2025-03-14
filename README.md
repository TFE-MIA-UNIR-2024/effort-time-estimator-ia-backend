# Effort Time Estimator IA Backend

## Requisitos

- Python 3.12
- pip
- virtualenv

## Instalación

1. Clona el repositorio:

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd effort-time-estimator-ia-backend
    ```

2. Crea y activa un entorno virtual:

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

4. Crea un archivo `.env` en la raíz del proyecto con las siguientes variables de entorno:

    ```plaintext
    SUPABASE_URL=<tu_supabase_url>
    SUPABASE_KEY=<tu_supabase_key>
    ```

## Ejecución

1. Inicia el servidor de desarrollo:

    ```bash
    uvicorn main:app --reload
    ```

2. Prueba los endpoints:

    - Para predecir: `http://127.0.0.1:8000/predict`
    - Para entrenar: `http://127.0.0.1:8000/train`

## Pruebas

1. Ejecuta el script de pruebas:

    ```bash
    python test.py
    ```

    Ingresa el endpoint que deseas probar (`predict` o `train`).

## Despliegue en Vercel

1. Asegúrate de tener el archivo `vercel.json` configurado correctamente:

    ```json
    {
        "version": 2,
        "builds": [
            {
                "src": "main.py",
                "use": "@vercel/python"
            }
        ],
        "routes": [
            {
                "src": "/(.*)",
                "dest": "main.py"
            }
        ]
    }
    ```

2. Despliega el proyecto en Vercel:

    ```bash
    vercel
    ```

    Sigue las instrucciones en pantalla para completar el despliegue.

## Notas

- Asegúrate de tener configurado correctamente el cliente de Supabase en `supabase_client.py`.
- Ajusta las rutas y nombres de columnas en `train_model.py` según tu esquema de base de datos.
