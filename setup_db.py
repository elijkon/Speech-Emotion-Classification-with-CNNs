import sqlite3

def init_db():
    conn = sqlite3.connect('ravdess_metadata.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_audio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        emotion TEXT,
        intensity TEXT,
        actor_id INTEGER,
        gender TEXT,
        s3_image_url TEXT
    )
    ''')
    conn.commit()
    conn.close()
    print("Database ready.")

if __name__ == "__main__":
    init_db()