import sqlite3
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

import schedule

API_URL    = "https://jsonplaceholder.typicode.com/posts"
DB_PATH    = "posts.db"
MAX_RETRY  = 3
BASE_DELAY = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def fetch_posts(url: str = API_URL) -> list[dict]:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            wait = BASE_DELAY ** attempt
            if attempt == MAX_RETRY:
                raise
            time.sleep(wait)


def transform(posts: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    return [{
        "userId":    p["userId"],
        "id":        p["id"],
        "title":     p["title"].upper(),
        "body":      p["body"],
        "timestamp": now,
    } for p in posts]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id        INTEGER PRIMARY KEY,
            userId    INTEGER NOT NULL,
            title     TEXT    NOT NULL,
            body      TEXT    NOT NULL,
            timestamp TEXT    NOT NULL
        )
    """)
    conn.commit()


def get_last_loaded_id(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM posts").fetchone()[0]


def load(posts: list[dict], db_path: str = DB_PATH) -> int:
    with sqlite3.connect(db_path) as conn:
        init_db(conn)
        last_id = get_last_loaded_id(conn)
        new_posts = [p for p in posts if p["id"] > last_id]
        if new_posts:
            conn.executemany(
                "INSERT INTO posts (id, userId, title, body, timestamp) "
                "VALUES (:id, :userId, :title, :body, :timestamp)",
                new_posts,
            )
            conn.commit()
        return len(new_posts)


def run_pipeline() -> Optional[int]:
    try:
        return load(transform(fetch_posts()))
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return None


if __name__ == "__main__":
    run_pipeline()
    schedule.every(5).minutes.do(run_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(10)