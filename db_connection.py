import os, pymysql
from dotenv import load_dotenv
load_dotenv()

_db = None

def get_connection():
    global _db
    if _db is None or not _db.open:
        _db = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),  # type: ignore
            database=os.getenv("DB_NAME"),
            charset="utf8mb4",          # type: ignore
            cursorclass=pymysql.cursors.DictCursor  # type: ignore
        )
    return _db

def get_cursor():
    return get_connection().cursor()

def commit():
    get_connection().commit()
