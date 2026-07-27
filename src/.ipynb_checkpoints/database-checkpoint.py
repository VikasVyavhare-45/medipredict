import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "../database/medipredict.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, email, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()


def verify_login(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        return None
    if user["password_hash"] == hash_password(password):
        return dict(user)
    return None


def save_prediction(user_id, disease, input_data, prediction, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (user_id, disease, input_data, prediction, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, disease, str(input_data), prediction, confidence, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_user_history(user_id, disease=None):
    conn = get_connection()
    cursor = conn.cursor()
    if disease:
        cursor.execute(
            "SELECT * FROM history WHERE user_id = ? AND disease = ? ORDER BY created_at DESC",
            (user_id, disease)
        )
    else:
        cursor.execute(
            "SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
