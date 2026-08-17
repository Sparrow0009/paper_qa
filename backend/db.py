import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/paperqa")

def get_connection():
    conn = psycopg.connect(DB_URL)
    register_vector(conn)
    return conn
