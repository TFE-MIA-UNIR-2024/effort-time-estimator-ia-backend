# supabase_client.py

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()  # Carga variables de entorno desde .env

def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Faltan SUPABASE_URL o SUPABASE_KEY en el entorno.")
    supabase: Client = create_client(url, key)
    return supabase