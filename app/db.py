import sqlite3
from contextlib import contextmanager


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id INTEGER PRIMARY KEY,
    api_base_url TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    api_key_encrypted TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'url',
    source_url TEXT NOT NULL,
    uploaded_video_path TEXT NOT NULL DEFAULT '',
    task_name TEXT NOT NULL DEFAULT '',
    normalized_source_url TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    output_mode TEXT NOT NULL DEFAULT 'full_notes',
    fetch_strategy TEXT NOT NULL DEFAULT 'auto',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    processing_stage TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    warning_message TEXT NOT NULL DEFAULT '',
    log_text TEXT NOT NULL DEFAULT '',
    work_dir TEXT NOT NULL,
    transcript_path TEXT NOT NULL DEFAULT '',
    notes_html_path TEXT NOT NULL DEFAULT '',
    notes_pdf_path TEXT NOT NULL DEFAULT '',
    metadata_json_path TEXT NOT NULL DEFAULT '',
    reused_from_job_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized_source_url TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    metadata_json_path TEXT NOT NULL DEFAULT '',
    transcript_path TEXT NOT NULL DEFAULT '',
    cover_path TEXT NOT NULL DEFAULT '',
    frames_dir TEXT NOT NULL DEFAULT '',
    transcript_chars INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


JOB_MIGRATIONS = {
    "source_kind": "ALTER TABLE jobs ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'url'",
    "uploaded_video_path": "ALTER TABLE jobs ADD COLUMN uploaded_video_path TEXT NOT NULL DEFAULT ''",
    "task_name": "ALTER TABLE jobs ADD COLUMN task_name TEXT NOT NULL DEFAULT ''",
    "normalized_source_url": "ALTER TABLE jobs ADD COLUMN normalized_source_url TEXT NOT NULL DEFAULT ''",
    "warning_message": "ALTER TABLE jobs ADD COLUMN warning_message TEXT NOT NULL DEFAULT ''",
    "reused_from_job_id": "ALTER TABLE jobs ADD COLUMN reused_from_job_id INTEGER",
    "output_mode": "ALTER TABLE jobs ADD COLUMN output_mode TEXT NOT NULL DEFAULT 'full_notes'",
    "fetch_strategy": "ALTER TABLE jobs ADD COLUMN fetch_strategy TEXT NOT NULL DEFAULT 'auto'",
    "cancel_requested": "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0",
    "processing_stage": "ALTER TABLE jobs ADD COLUMN processing_stage TEXT NOT NULL DEFAULT ''",
}


def dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def init_db(database_path: str) -> None:
    conn = sqlite3.connect(database_path)
    conn.executescript(SCHEMA)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    for column_name, statement in JOB_MIGRATIONS.items():
        if column_name not in columns:
            conn.execute(statement)

    asset_columns = {row[1] for row in conn.execute("PRAGMA table_info(source_assets)").fetchall()}
    if "frames_dir" not in asset_columns:
        conn.execute("ALTER TABLE source_assets ADD COLUMN frames_dir TEXT NOT NULL DEFAULT ''")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_user_normalized_url ON jobs(user_id, normalized_source_url)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_request_logs_user_provider_created ON api_request_logs(user_id, provider, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_assets_url ON source_assets(normalized_source_url)"
    )
    conn.commit()
    conn.close()


@contextmanager
def get_conn(database_path: str):
    conn = sqlite3.connect(database_path)
    conn.row_factory = dict_factory
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
