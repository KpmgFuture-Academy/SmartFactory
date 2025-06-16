# core/db_connector.py

import psycopg2

def get_connection():
    """
    Returns a connection to the local PostgreSQL DB.
    Adjust parameters as needed.
    """
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="0523"
    )
