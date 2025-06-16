import psycopg2

# Replace with your actual DB credentials
DB_INFO = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0523'  # Consider using environment variables for security
}

# Global connection and cursor (if needed elsewhere)
conn = psycopg2.connect(**DB_INFO)
cur = conn.cursor()

# Connection getter that returns both conn and cur
def get_connection():
    conn = psycopg2.connect(**DB_INFO)
    cur = conn.cursor()
    return conn, cur
