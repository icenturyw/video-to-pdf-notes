import os
import re
import shutil
from functools import wraps

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from .auth import decrypt_secret, encrypt_secret, hash_password, verify_password
from .db import get_conn
from .pipeline import OUTPUT_MODE_SPECS, SUCCESS_STATUSES, normalize_source_url, run_groq_self_test

bp = Blueprint("main", __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("main.login"))
        return view(*args, **kwargs)

    return wrapped


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with get_conn(current_app.config["DATABASE"]) as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_job_for_user(user_id: int, job_id: int):
    with get_conn(current_app.config["DATABASE"]) as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()


def delete_job_for_user(user_id: int, job_id: int) -> tuple[bool, str]:
    job = get_job_for_user(user_id, job_id)
    if not job:
        return False, "任务不存在。"
    if job["status"] in {"pending", "running"}:
        return False, "任务正在执行中，暂不支持删除。"
    work_dir = job.get("work_dir") or ""
    with get_conn(current_app.config["DATABASE"]) as conn:
        conn.execute("DELETE FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id))
    if work_dir and os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    return True, "任务已删除。"


def find_reusable_job(user_id: int, normalized_source_url: str):
    if not normalized_source_url:
        return None
    with get_conn(current_app.config["DATABASE"]) as conn:
        jobs = conn.execute(
            """
            SELECT * FROM jobs
            WHERE user_id = ?
              AND status IN (?, ?)
            ORDER BY id DESC
            """,
            (user_id, "completed", "completed_with_warnings"),
        ).fetchall()
    for job in jobs:
        candidates = {
            (job.get("normalized_source_url") or "").strip(),
            normalize_source_url(job.get("source_url") or ""),
        }
        if (
            normalized_source_url in candidates
            and job.get("notes_pdf_path")
            and os.path.exists(job["notes_pdf_path"])
        ):
            return job
    return None


JOB_SORT_OPTIONS = {
    "created_desc": ("j.created_at DESC, j.id DESC", "最新创建"),
    "created_asc": ("j.created_at ASC, j.id ASC", "最早创建"),
    "updated_desc": ("j.updated_at DESC, j.id DESC", "最近更新"),
    "name_asc": ("LOWER(COALESCE(NULLIF(j.task_name, ''), NULLIF(j.title, ''), j.source_url)) ASC, j.id DESC", "名称 A-Z"),
    "name_desc": ("LOWER(COALESCE(NULLIF(j.task_name, ''), NULLIF(j.title, ''), j.source_url)) DESC, j.id DESC", "名称 Z-A"),
}

PDF_SORT_OPTIONS = JOB_SORT_OPTIONS


def create_job_and_start(user_id: int, source_url: str, platform: str, reuse_job: dict | None = None):
    normalized_source_url = normalize_source_url(source_url)
    output_mode = (reuse_job or {}).get("output_mode", "full_notes")
    fetch_strategy = "reuse" if reuse_job else "auto"
    source_kind = (reuse_job or {}).get("source_kind", "url")
    uploaded_video_path = (reuse_job or {}).get("uploaded_video_path", "")
    task_name = (reuse_job or {}).get("task_name", "")
    with get_conn(current_app.config["DATABASE"]) as conn:
        cursor = conn.execute(
            """
            INSERT INTO jobs(user_id, platform, source_kind, source_url, uploaded_video_path, task_name, normalized_source_url, work_dir, output_mode, fetch_strategy, status, reused_from_job_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, 'queued', ?)
            """,
            (
                user_id,
                platform,
                source_kind,
                source_url,
                uploaded_video_path,
                task_name,
                normalized_source_url,
                output_mode,
                fetch_strategy,
                (reuse_job or {}).get("id"),
            ),
        )
        job_id = cursor.lastrowid
        work_dir = os.path.join(current_app.config["JOB_DIR"], str(user_id), str(job_id))
        os.makedirs(work_dir, exist_ok=True)
        conn.execute("UPDATE jobs SET work_dir = ? WHERE id = ?", (work_dir, job_id))
    return job_id, normalized_source_url


def preferred_job_name(job: dict) -> str:
    return (job.get("task_name") or job.get("title") or job.get("source_url") or f"任务 #{job.get('id')}").strip()


def _safe_job_stem(name: str, job_id: int) -> str:
    safe = secure_filename(name or "")
    safe = safe.strip(" ._-")
    return safe or f"job-{job_id}"


def pdf_display_name(job: dict) -> str:
    base = preferred_job_name(job)
    safe = _safe_job_stem(base, job["id"])
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    return safe


def _rename_job_output_files(job: dict, new_name: str) -> dict:
    work_dir = (job.get("work_dir") or "").strip()
    if not work_dir or not os.path.isdir(work_dir):
        return {}
    stem = _safe_job_stem(new_name, job["id"])
    updates = {}
    path_fields = ["notes_pdf_path", "notes_html_path"]
    for field in path_fields:
        old_path = (job.get(field) or "").strip()
        if not old_path or not os.path.exists(old_path):
            continue
        ext = os.path.splitext(old_path)[1]
        new_path = os.path.join(work_dir, f"{stem}{ext}")
        if os.path.abspath(old_path) != os.path.abspath(new_path):
            os.replace(old_path, new_path)
        updates[field] = new_path
        old_base, _ = os.path.splitext(old_path)
        new_base, _ = os.path.splitext(new_path)
        for side_ext in [".tex", ".aux", ".log", ".out", ".toc"]:
            side_old = old_base + side_ext
            side_new = new_base + side_ext
            if os.path.exists(side_old):
                if os.path.abspath(side_old) != os.path.abspath(side_new):
                    os.replace(side_old, side_new)
    return updates


def clone_uploaded_source(job: dict, new_work_dir: str) -> str:
    source_path = (job.get("uploaded_video_path") or "").strip()
    if not source_path or not os.path.exists(source_path):
        return source_path
    uploads_dir = os.path.join(new_work_dir, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    target_path = os.path.join(uploads_dir, os.path.basename(source_path))
    shutil.copy2(source_path, target_path)
    return target_path


def reset_job_for_reuse(job_id: int, user_id: int) -> None:
    with get_conn(current_app.config["DATABASE"]) as conn:
        job = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        if not job:
            return
        notes_pdf_path = job.get("notes_pdf_path") or ""
        notes_html_path = job.get("notes_html_path") or ""
        for path in (notes_pdf_path, notes_html_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        conn.execute(
            """
            UPDATE jobs
            SET status = 'queued',
                fetch_strategy = 'reuse',
                cancel_requested = 0,
                processing_stage = '',
                error_message = '',
                warning_message = '',
                notes_html_path = '',
                notes_pdf_path = '',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (job_id, user_id),
        )


@bp.context_processor
def inject_globals():
    return {"auth_user": current_user()}


@bp.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("main.login"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 8:
            flash("用户名至少 3 位，密码至少 8 位。", "error")
            return render_template("register.html")
        try:
            with get_conn(current_app.config["DATABASE"]) as conn:
                cursor = conn.execute(
                    "INSERT INTO users(username, password_hash) VALUES (?, ?)",
                    (username, hash_password(password)),
                )
                user_id = cursor.lastrowid
                conn.execute(
                    "INSERT INTO user_settings(user_id, system_prompt) VALUES (?, ?)",
                    (user_id, ""),
                )
            flash("注册成功，请登录。", "success")
            return redirect(url_for("main.login"))
        except Exception:
            flash("用户名已存在。", "error")
    return render_template("register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_conn(current_app.config["DATABASE"]) as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not user or not verify_password(user["password_hash"], password):
            flash("用户名或密码错误。", "error")
            return render_template("login.html")
        session.clear()
        session["user_id"] = user["id"]
        return redirect(url_for("main.dashboard"))
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.login"))


@bp.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    user = current_user()
    sort_key = request.args.get("sort", "created_desc").strip()
    order_sql = JOB_SORT_OPTIONS.get(sort_key, JOB_SORT_OPTIONS["created_desc"])[0]
    if request.method == "POST":
        source_url = request.form.get("source_url", "").strip()
        uploaded_file = request.files.get("video_file")
        task_name = request.form.get("task_name", "").strip()
        platform = request.form.get("platform", "").strip() or "youtube"
        output_mode = request.form.get("output_mode", "full_notes").strip() or "full_notes"
        has_upload = bool(uploaded_file and uploaded_file.filename)
        normalized_source_url = normalize_source_url(source_url) if source_url else ""
        if not source_url and not has_upload:
            flash("请输入视频链接，或上传一个视频文件。", "error")
            return redirect(url_for("main.dashboard"))
        if output_mode not in OUTPUT_MODE_SPECS:
            flash("输出模式无效。", "error")
            return redirect(url_for("main.dashboard"))
        if source_url and has_upload:
            flash("链接和视频文件二选一，不要同时提交。", "error")
            return redirect(url_for("main.dashboard"))

        if has_upload:
            filename = secure_filename(uploaded_file.filename or "")
            if not filename:
                flash("上传文件名无效。", "error")
                return redirect(url_for("main.dashboard"))
            with get_conn(current_app.config["DATABASE"]) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO jobs(user_id, platform, source_kind, source_url, uploaded_video_path, task_name, normalized_source_url, work_dir, output_mode, fetch_strategy, status)
                    VALUES (?, ?, 'upload', '', '', ?, '', '', ?, 'auto', 'queued')
                    """,
                    (user["id"], "upload", task_name, output_mode),
                )
                job_id = cursor.lastrowid
                work_dir = os.path.join(current_app.config["JOB_DIR"], str(user["id"]), str(job_id))
                uploads_dir = os.path.join(work_dir, "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                stored_path = os.path.join(uploads_dir, filename)
                uploaded_file.save(stored_path)
                conn.execute(
                    "UPDATE jobs SET work_dir = ?, uploaded_video_path = ?, title = COALESCE(NULLIF(title, ''), ?) WHERE id = ?",
                    (work_dir, stored_path, task_name or os.path.splitext(filename)[0], job_id),
                )
        else:
            reusable_job = find_reusable_job(user["id"], normalized_source_url)
            if reusable_job and (reusable_job.get("output_mode") or "full_notes") == output_mode:
                flash("这个链接已经生成过，已直接复用历史结果。", "success")
                return redirect(url_for("main.job_detail", job_id=reusable_job["id"]))
            with get_conn(current_app.config["DATABASE"]) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO jobs(user_id, platform, source_kind, source_url, task_name, normalized_source_url, work_dir, output_mode, fetch_strategy, status)
                    VALUES (?, ?, 'url', ?, ?, ?, '', ?, 'auto', 'queued')
                    """,
                    (user["id"], platform, source_url, task_name, normalized_source_url, output_mode),
                )
                job_id = cursor.lastrowid
                work_dir = os.path.join(current_app.config["JOB_DIR"], str(user["id"]), str(job_id))
                os.makedirs(work_dir, exist_ok=True)
                conn.execute("UPDATE jobs SET work_dir = ? WHERE id = ?", (work_dir, job_id))
        flash("任务已进入队列，等待后台处理。", "success")
        return redirect(url_for("main.job_detail", job_id=job_id))

    with get_conn(current_app.config["DATABASE"]) as conn:
        jobs = conn.execute(
            f"SELECT * FROM jobs j WHERE user_id = ? ORDER BY {order_sql}",
            (user["id"],),
        ).fetchall()
    return render_template(
        "dashboard.html",
        jobs=jobs,
        output_modes=OUTPUT_MODE_SPECS,
        sort_key=sort_key,
        sort_options=JOB_SORT_OPTIONS,
        preferred_job_name=preferred_job_name,
        pdf_display_name=pdf_display_name,
    )


@bp.route("/pdfs")
@login_required
def pdf_library():
    user = current_user()
    sort_key = request.args.get("sort", "updated_desc").strip()
    order_sql = PDF_SORT_OPTIONS.get(sort_key, PDF_SORT_OPTIONS["updated_desc"])[0]
    with get_conn(current_app.config["DATABASE"]) as conn:
        jobs = conn.execute(
            f"""
            SELECT * FROM jobs j
            WHERE user_id = ?
              AND status = 'completed'
            ORDER BY {order_sql}
            """,
            (user["id"],),
        ).fetchall()
    generated_pdfs = [
        job for job in jobs
        if job.get("notes_pdf_path") and os.path.exists(job["notes_pdf_path"])
    ]
    return render_template(
        "pdf_library.html",
        generated_pdfs=generated_pdfs,
        sort_key=sort_key,
        sort_options=PDF_SORT_OPTIONS,
        output_modes=OUTPUT_MODE_SPECS,
        preferred_job_name=preferred_job_name,
        pdf_display_name=pdf_display_name,
    )


@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    user = current_user()
    with get_conn(current_app.config["DATABASE"]) as conn:
        if request.method == "POST":
            api_base_url = request.form.get("api_base_url", "").strip()
            model_name = request.form.get("model_name", "").strip()
            api_key = request.form.get("api_key", "").strip()
            system_prompt = request.form.get("system_prompt", "").strip()
            encrypted = encrypt_secret(current_app.config["SECRET_KEY"], api_key) if api_key else None
            if encrypted:
                conn.execute(
                    """
                    UPDATE user_settings
                    SET api_base_url = ?, model_name = ?, api_key_encrypted = ?, system_prompt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                    """,
                    (api_base_url, model_name, encrypted, system_prompt, user["id"]),
                )
            else:
                conn.execute(
                    """
                    UPDATE user_settings
                    SET api_base_url = ?, model_name = ?, system_prompt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                    """,
                    (api_base_url, model_name, system_prompt, user["id"]),
                )
            flash("配置已保存。", "success")

        settings_row = conn.execute(
            "SELECT * FROM user_settings WHERE user_id = ?",
            (user["id"],),
        ).fetchone()

    masked_key = ""
    if settings_row and settings_row.get("api_key_encrypted"):
        decrypted = decrypt_secret(current_app.config["SECRET_KEY"], settings_row["api_key_encrypted"])
        masked_key = decrypted[:4] + "..." + decrypted[-4:] if len(decrypted) >= 8 else "已保存"
    return render_template("settings.html", settings=settings_row, masked_key=masked_key)


@bp.route("/settings/test-groq", methods=["POST"])
@login_required
def settings_test_groq():
    user = current_user()
    test_dir = os.path.join(current_app.config["DATA_DIR"], "groq-self-test", str(user["id"]))
    try:
        result = run_groq_self_test(test_dir)
        return jsonify(
            {
                "ok": True,
                "message": "Groq 转写配置可用。",
                "model": result["model"],
                "text": result["text"],
                "request_id": result["request_id"],
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@bp.route("/jobs/<int:job_id>")
@login_required
def job_detail(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job:
        return Response("Not found", status=404)
    return render_template("job.html", job=job, preferred_job_name=preferred_job_name, pdf_display_name=pdf_display_name)


@bp.route("/jobs/<int:job_id>/rename", methods=["POST"])
@login_required
def job_rename(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job:
        return Response("Not found", status=404)
    new_name = request.form.get("task_name", "").strip()
    if len(new_name) < 1:
        flash("任务名称不能为空。", "error")
        return redirect(url_for("main.job_detail", job_id=job_id))
    file_updates = _rename_job_output_files(job, new_name)
    with get_conn(current_app.config["DATABASE"]) as conn:
        if file_updates:
            assignments = ", ".join([f"{k} = ?" for k in file_updates.keys()])
            values = list(file_updates.values())
            values.extend([new_name, new_name, job_id, user["id"]])
            conn.execute(
                f"UPDATE jobs SET {assignments}, task_name = ?, title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                values,
            )
        else:
            conn.execute(
                "UPDATE jobs SET task_name = ?, title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                (new_name, new_name, job_id, user["id"]),
            )
    flash("任务名称已更新，生成文件名已同步。", "success")
    return redirect(url_for("main.job_detail", job_id=job_id))


@bp.route("/jobs/<int:job_id>/rerun", methods=["POST"])
@login_required
def job_rerun_full(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job:
        return Response("Not found", status=404)
    with get_conn(current_app.config["DATABASE"]) as conn:
        cursor = conn.execute(
            """
            INSERT INTO jobs(user_id, platform, source_url, normalized_source_url, work_dir, output_mode, fetch_strategy, status)
            VALUES (?, ?, ?, ?, '', ?, 'refresh', 'queued')
            """,
            (user["id"], job["platform"], job["source_url"], job.get("normalized_source_url") or normalize_source_url(job["source_url"]), job.get("output_mode") or "full_notes"),
        )
        new_job_id = cursor.lastrowid
        work_dir = os.path.join(current_app.config["JOB_DIR"], str(user["id"]), str(new_job_id))
        os.makedirs(work_dir, exist_ok=True)
        uploaded_video_path = clone_uploaded_source(job, work_dir) if (job.get("source_kind") == "upload") else ""
        conn.execute(
            "UPDATE jobs SET work_dir = ?, task_name = ?, source_kind = ?, uploaded_video_path = ? WHERE id = ?",
            (work_dir, job.get("task_name") or "", job.get("source_kind") or "url", uploaded_video_path, new_job_id),
        )
    flash("已按完整流程重新入队。", "success")
    return redirect(url_for("main.job_detail", job_id=new_job_id))


@bp.route("/jobs/<int:job_id>/rerun-reuse", methods=["POST"])
@login_required
def job_rerun_reuse(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job:
        return Response("Not found", status=404)
    if job["status"] in {"queued", "running"}:
        flash("任务正在执行中，不能再次复用重生成。", "error")
        return redirect(url_for("main.job_detail", job_id=job_id))
    if not job.get("metadata_json_path") or not os.path.exists(job["metadata_json_path"]):
        flash("当前任务缺少可复用的元数据文件，请使用完整重跑。", "error")
        return redirect(url_for("main.job_detail", job_id=job_id))
    if not job.get("transcript_path") or not os.path.exists(job["transcript_path"]):
        flash("当前任务缺少可复用的转写文件，请使用完整重跑。", "error")
        return redirect(url_for("main.job_detail", job_id=job_id))
    reset_job_for_reuse(job_id, user["id"])
    flash("已在当前任务内复用中间文件重新入队。", "success")
    return redirect(url_for("main.job_detail", job_id=job_id))


@bp.route("/jobs/<int:job_id>/delete", methods=["POST"])
@login_required
def job_delete(job_id: int):
    user = current_user()
    ok, message = delete_job_for_user(user["id"], job_id)
    flash(message, "success" if ok else "error")
    return redirect(url_for("main.dashboard"))


@bp.route("/jobs/<int:job_id>/cancel", methods=["POST"])
@login_required
def job_cancel(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job:
        return Response("Not found", status=404)
    with get_conn(current_app.config["DATABASE"]) as conn:
        if job["status"] == "queued":
            conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', processing_stage = 'cancelled', log_text = log_text || ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                ("[系统] 任务在队列中被取消\n", job_id, user["id"]),
            )
            flash("已取消排队中的任务。", "success")
        elif job["status"] == "running":
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                (job_id, user["id"]),
            )
            flash("已发出取消请求，任务会在下一个安全检查点停止。", "success")
        else:
            flash("当前任务无需取消。", "error")
    return redirect(url_for("main.job_detail", job_id=job_id))


@bp.route("/jobs/<int:job_id>/status")
@login_required
def job_status(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job:
        return jsonify({"error": "not_found"}), 404

    pdf_ready = bool(job.get("notes_pdf_path")) and os.path.exists(job["notes_pdf_path"])
    html_ready = bool(job.get("notes_html_path")) and os.path.exists(job["notes_html_path"])
    terminal = job["status"] in SUCCESS_STATUSES or job["status"] in {"failed", "cancelled"}
    return jsonify(
        {
            "id": job["id"],
            "status": job["status"],
            "title": job.get("title") or "",
            "output_mode": job.get("output_mode") or "full_notes",
            "processing_stage": job.get("processing_stage") or "",
            "error_message": job.get("error_message") or "",
            "warning_message": job.get("warning_message") or "",
            "log_text": job.get("log_text") or "",
            "pdf_ready": pdf_ready,
            "html_ready": html_ready,
            "terminal": terminal,
            "pdf_url": url_for("main.job_pdf", job_id=job["id"]) if pdf_ready else "",
            "html_url": url_for("main.job_html", job_id=job["id"]) if html_ready else "",
        }
    )


@bp.route("/jobs/<int:job_id>/pdf")
@login_required
def job_pdf(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job or not job.get("notes_pdf_path") or not os.path.exists(job["notes_pdf_path"]):
        return Response("PDF not found", status=404)
    return send_file(job["notes_pdf_path"], mimetype="application/pdf", download_name=pdf_display_name(job))


@bp.route("/jobs/<int:job_id>/html")
@login_required
def job_html(job_id: int):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job or not job.get("notes_html_path") or not os.path.exists(job["notes_html_path"]):
        return Response("HTML not found", status=404)
    with open(job["notes_html_path"], "r", encoding="utf-8") as handle:
        content = handle.read()

    work_dir = job.get("work_dir") or ""
    if work_dir:
        work_dir_url = work_dir.replace(os.sep, "/").rstrip("/")

        def replace_local_asset(match):
            raw_path = match.group(1).strip()
            normalized = raw_path.replace("\\", "/")
            if normalized.startswith(work_dir_url + "/"):
                rel_path = normalized[len(work_dir_url) + 1 :]
                return f'src="{url_for("main.job_asset", job_id=job_id, asset_path=rel_path)}"'
            return match.group(0)

        content = re.sub(r'src="file://([^"]+)"', replace_local_asset, content)

    return Response(content, mimetype="text/html")


@bp.route("/jobs/<int:job_id>/asset/<path:asset_path>")
@login_required
def job_asset(job_id: int, asset_path: str):
    user = current_user()
    job = get_job_for_user(user["id"], job_id)
    if not job or not job.get("work_dir"):
        return Response("Asset not found", status=404)

    work_dir = os.path.abspath(job["work_dir"])
    normalized = os.path.normpath(asset_path).lstrip(os.sep)
    resolved = os.path.abspath(os.path.join(work_dir, normalized))
    if not resolved.startswith(work_dir + os.sep):
        return Response("Asset not found", status=404)
    if not os.path.exists(resolved) or not os.path.isfile(resolved):
        return Response("Asset not found", status=404)
    return send_file(resolved)
