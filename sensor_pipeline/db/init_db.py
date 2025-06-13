import psycopg2
from config import DB_CONFIG

def test_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print("✅ PostgreSQL connected successfully:", version)
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ PostgreSQL connection failed:", e)

if __name__ == "__main__":
    test_connection()
