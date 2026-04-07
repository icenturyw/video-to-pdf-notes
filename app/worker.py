import threading
import time
from datetime import datetime

from .db import get_conn
from .pipeline import process_job


_worker_lock = threading.Lock()
_worker_started = False


def recover_incomplete_jobs(app) -> None:
    database_path = app.config["DATABASE"]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    recovery_line = f"[{timestamp}] [系统] 服务重启后检测到未完成任务，已自动重新入队\n"
    with get_conn(database_path) as conn:
        stale_running = conn.execute(
            """
            SELECT id, log_text
            FROM jobs
            WHERE status = 'running'
            """
        ).fetchall()
        for job in stale_running:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    processing_stage = 'recovered',
                    error_message = '',
                    updated_at = CURRENT_TIMESTAMP,
                    log_text = ?
                WHERE id = ?
                """,
                ((job.get("log_text") or "") + recovery_line, job["id"]),
            )


def start_worker(app) -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_worker_loop, args=(app,), daemon=True, name="job-worker")
        thread.start()
        _worker_started = True


def _worker_loop(app) -> None:
    database_path = app.config["DATABASE"]
    while True:
        try:
            with get_conn(database_path) as conn:
                job = conn.execute(
                    """
                    SELECT id
                    FROM jobs
                    WHERE status = 'queued'
                    ORDER BY id ASC
                    LIMIT 1
                    """
                ).fetchone()
            if not job:
                time.sleep(1.0)
                continue
            process_job(app, job["id"])
        except Exception:
            time.sleep(2.0)
