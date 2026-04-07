import html
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat
from weasyprint import HTML

from .auth import decrypt_secret
from .db import get_conn


SKILL_SYSTEM_PROMPT = """你是一个高质量课程讲义生成器。你要参考 figure-rich LaTeX lecture notes 的写法，把视频内容重写成真正适合学习的中文讲义。

总原则：
1. 必须以视频真实教学内容为主，不写寒暄、口播、平台互动、广告和结束套话。
2. 不能只按字幕时间顺序平铺；要重建教学顺序，让读者先理解问题动机，再理解核心思想、机制细节、例子/证据，最后得到可带走的结论。
3. 输出必须像一个认真授课的人写出来的讲义：解释充分，逻辑过渡清晰，先讲直觉，再讲形式化细节。
4. 如果内容密集，要拆成较小的小节逐步展开，不要把复杂内容压成几条空泛 bullet。
5. 要主动提炼“必须记住的核心点”“背景补充”“常见误区/易错点”。
6. 如果素材不足，要明确写“信息不足”，不要臆造公式、实验结果、代码细节或外部引用。

输出必须是严格 JSON，对象格式如下：
{
  "title": "讲义标题",
  "summary": ["6-12条全局要点"],
  "sections": [
    {
      "heading": "章节标题",
      "goal": "本章想解决什么问题/读者为什么要学这一章",
      "bullets": ["6-14条章节主干要点"],
      "subsections": [
        {
          "heading": "小节标题",
          "paragraphs": ["2-5段较完整解释，每段是自然中文句子，不要只写词组"],
          "bullets": ["0-6条补充要点"],
          "important": ["0-3条必须记住的核心结论"],
          "knowledge": ["0-3条背景、类比、设计取舍或术语说明"],
          "warnings": ["0-3条常见误区、限制、易错点"],
          "formula": {
            "title": "公式标题",
            "expression": "仅在视频里明确出现或可稳定重建时填写 LaTeX 公式，否则为空",
            "symbol_notes": ["符号解释1", "符号解释2"]
          },
          "code": {
            "language": "Python",
            "caption": "代码片段说明",
            "content": "仅在视频明确展示或描述了关键代码时填写"
          }
        }
      ],
      "section_summary": ["2-5条本章小结"]
    }
  ],
  "conclusion": ["4-8条总结与延伸"]
}

额外要求：
1. 每个 section 尽量回答：解决什么问题、为什么简单方法不够、核心思想是什么、怎么工作、例子或证据是什么、读者应记住什么。
2. 每个大章节最后都要给出 section_summary。
3. 只有在视频内容明确支持时才输出 formula 或 code。
4. important / knowledge / warnings 必须具体，不能是空泛套话。
5. 如果当前输入只是长视频的一部分，只处理当前分段，但要把这一段写完整，不要缩成很短摘要。
"""

SUCCESS_STATUSES = {"completed", "completed_with_warnings"}
OPENROUTER_LIMIT_PER_MINUTE = 20
TRANSCRIPT_CHUNK_TARGET = 7000
TRANSCRIPT_CHUNK_MIN = 3000
MODEL_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MODEL_MAX_RETRIES = 5
GROQ_MAX_RETRIES = 4
GROQ_MAX_FILE_BYTES = 24 * 1024 * 1024
TERMINAL_STATUSES = {"completed", "completed_with_warnings", "failed", "cancelled"}
FRAME_EXTRACTION_OVERSAMPLE = 3
FRAME_EXTRACTION_MAX_CANDIDATES = 24
FRAME_DEDUPE_HASH_SIZE = 8
FRAME_DEDUPE_HAMMING_THRESHOLD = 8
FRAME_DEDUPE_BRIGHTNESS_THRESHOLD = 18
FRAME_BLACK_BRIGHTNESS_THRESHOLD = 18
FRAME_BLACK_CONTRAST_THRESHOLD = 10
FRAME_PRIORITY_MAX_COLORS = 48
OUTPUT_MODE_SPECS = {
    "summary": {
        "label": "摘要版",
        "summary_range": "4-8条",
        "bullet_range": "4-8条",
        "sections_min": 2,
        "goal": "适合快速回顾，尽量提炼最关键知识点。",
        "merge_summary_limit": 12,
        "merge_conclusion_limit": 8,
    },
    "full_notes": {
        "label": "完整讲义",
        "summary_range": "6-12条",
        "bullet_range": "6-14条",
        "sections_min": 3,
        "goal": "尽量完整覆盖知识点，适合学习型讲义阅读。",
        "merge_summary_limit": 18,
        "merge_conclusion_limit": 12,
    },
    "exam_prep": {
        "label": "考点提纲",
        "summary_range": "8-14条",
        "bullet_range": "6-12条",
        "sections_min": 4,
        "goal": "突出核心考点、易错点、记忆点和可能题型。",
        "merge_summary_limit": 20,
        "merge_conclusion_limit": 14,
    },
    "transcript_expanded": {
        "label": "逐段扩展",
        "summary_range": "6-10条",
        "bullet_range": "8-16条",
        "sections_min": 4,
        "goal": "尽量保留讲解顺序，做成更长、更细的分段讲义。",
        "merge_summary_limit": 16,
        "merge_conclusion_limit": 10,
    },
}


def normalize_source_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    raw = _expand_short_url(raw)
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc in {"youtu.be"}:
        video_id = parsed.path.strip("/")
        query_dict = dict(parse_qsl(parsed.query, keep_blank_values=False))
        kept_pairs = [("v", video_id)] if video_id else []
        if query_dict.get("list"):
            kept_pairs.append(("list", query_dict["list"]))
        query = urlencode(kept_pairs)
        return urlunparse(("https", "www.youtube.com", "/watch", "", query, ""))

    if netloc in {"m.youtube.com", "youtube.com"}:
        netloc = "www.youtube.com"

    if netloc in {"www.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else ""
            kept_pairs = [("v", video_id)] if video_id else []
            query = urlencode(kept_pairs)
            return urlunparse(("https", "www.youtube.com", "/watch", "", query, ""))
        if parsed.path == "/watch":
            query_dict = dict(parse_qsl(parsed.query, keep_blank_values=False))
            kept_pairs = []
            if query_dict.get("v"):
                kept_pairs.append(("v", query_dict["v"]))
            if query_dict.get("list"):
                kept_pairs.append(("list", query_dict["list"]))
            query = urlencode(kept_pairs)
            return urlunparse(("https", "www.youtube.com", "/watch", "", query, ""))

    bv_match = re.search(r"/video/(BV[0-9A-Za-z]+)", parsed.path, flags=re.IGNORECASE)
    if netloc.endswith("bilibili.com") and bv_match:
        bv_id = bv_match.group(1)
        page = dict(parse_qsl(parsed.query, keep_blank_values=False)).get("p")
        kept_pairs = [("p", page)] if page else []
        query = urlencode(kept_pairs)
        return urlunparse(("https", "www.bilibili.com", f"/video/{bv_id}", "", query, ""))

    path = parsed.path.rstrip("/")
    query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
    kept_pairs = []
    for key, value in query_pairs:
        if key.lower().startswith(("utm_", "si", "feature", "pp", "spm_id_from")):
            continue
        kept_pairs.append((key, value))
    query = urlencode(sorted(kept_pairs))
    return urlunparse((scheme, netloc, path, "", query, ""))


def _slugify_filename(value: str, fallback: str = "notes") -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", text, flags=re.UNICODE).strip("-._ ")
    return text[:96] or fallback


def _job_display_title(job: dict, info: dict | None = None, notes: dict | None = None) -> str:
    return (
        (job.get("task_name") or "").strip()
        or ((notes or {}).get("title") or "").strip()
        or ((info or {}).get("title") or "").strip()
        or (job.get("title") or "").strip()
        or (job.get("source_url") or "").strip()
        or f"job-{job.get('id')}"
    )


def _source_input(job: dict) -> str:
    if (job.get("source_kind") or "url") == "upload":
        return (job.get("uploaded_video_path") or "").strip()
    return (job.get("source_url") or "").strip()


def _is_local_source(job: dict) -> bool:
    return (job.get("source_kind") or "url") == "upload"


def _expand_short_url(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.netloc.lower() not in {"b23.tv", "youtu.be"}:
        return raw
    try:
        response = requests.head(raw, allow_redirects=True, timeout=5)
        final_url = response.url
        if final_url:
            return final_url
    except Exception:
        pass
    try:
        response = requests.get(raw, allow_redirects=True, timeout=5)
        final_url = response.url
        if final_url:
            return final_url
    except Exception:
        pass
    return raw


def append_job_log(database_path: str, job_id: int, message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {message}\n"
    with get_conn(database_path) as conn:
        row = conn.execute("SELECT log_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
        current = row["log_text"] if row else ""
        conn.execute(
            "UPDATE jobs SET log_text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (current + line, job_id),
        )


def update_job(database_path: str, job_id: int, **fields) -> None:
    if not fields:
        return
    keys = sorted(fields)
    assignments = ", ".join([f"{key} = ?" for key in keys] + ["updated_at = CURRENT_TIMESTAMP"])
    values = [fields[key] for key in keys]
    values.append(job_id)
    with get_conn(database_path) as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)


def _copy_if_exists(src: str, dst: str) -> bool:
    if not src or not os.path.exists(src):
        return False
    if os.path.abspath(src) == os.path.abspath(dst):
        return True
    shutil.copy2(src, dst)
    return True


def fetch_job_with_settings(database_path: str, job_id: int):
    with get_conn(database_path) as conn:
        return conn.execute(
            """
            SELECT
                j.*,
                s.api_base_url,
                s.model_name,
                s.api_key_encrypted,
                s.system_prompt
            FROM jobs j
            LEFT JOIN user_settings s ON s.user_id = j.user_id
            WHERE j.id = ?
            """,
            (job_id,),
        ).fetchone()


def fetch_cached_asset(database_path: str, normalized_source_url: str):
    with get_conn(database_path) as conn:
        return conn.execute(
            "SELECT * FROM source_assets WHERE normalized_source_url = ?",
            (normalized_source_url,),
        ).fetchone()


def upsert_cached_asset(database_path: str, normalized_source_url: str, platform: str, title: str, metadata_json_path: str, transcript_path: str, cover_path: str, frames_dir: str = "") -> None:
    transcript_chars = 0
    if transcript_path and os.path.exists(transcript_path):
        transcript_chars = os.path.getsize(transcript_path)
    with get_conn(database_path) as conn:
        existing = conn.execute(
            "SELECT id FROM source_assets WHERE normalized_source_url = ?",
            (normalized_source_url,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE source_assets
                SET platform = ?, title = ?, metadata_json_path = ?, transcript_path = ?, cover_path = ?, frames_dir = ?, transcript_chars = ?, updated_at = CURRENT_TIMESTAMP
                WHERE normalized_source_url = ?
                """,
                (platform, title, metadata_json_path, transcript_path, cover_path, frames_dir, transcript_chars, normalized_source_url),
            )
        else:
            conn.execute(
                """
                INSERT INTO source_assets(normalized_source_url, platform, title, metadata_json_path, transcript_path, cover_path, frames_dir, transcript_chars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (normalized_source_url, platform, title, metadata_json_path, transcript_path, cover_path, frames_dir, transcript_chars),
            )


class JobCancelled(Exception):
    pass


def ensure_job_not_cancelled(database_path: str, job_id: int) -> None:
    with get_conn(database_path) as conn:
        row = conn.execute("SELECT cancel_requested, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise JobCancelled("任务不存在")
    if row["status"] == "cancelled" or int(row.get("cancel_requested") or 0) == 1:
        raise JobCancelled("任务已取消")


def set_processing_stage(database_path: str, job_id: int, stage: str) -> None:
    update_job(database_path, job_id, processing_stage=stage)


def _provider_name(api_base_url: str) -> str:
    normalized = (api_base_url or "").strip().lower()
    if "openrouter.ai" in normalized:
        return "openrouter"
    return "generic"


def _check_and_record_rate_limit(database_path: str, user_id: int, provider: str) -> float:
    if provider != "openrouter":
        return 0.0
    window_start_dt = datetime.utcnow() - timedelta(minutes=1)
    window_start = window_start_dt.strftime("%Y-%m-%d %H:%M:%S")
    with get_conn(database_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS request_count, MIN(created_at) AS oldest_created_at
            FROM api_request_logs
            WHERE user_id = ? AND provider = ? AND created_at >= ?
            """,
            (user_id, provider, window_start),
        ).fetchone()
        request_count = int(row["request_count"])
        if request_count >= OPENROUTER_LIMIT_PER_MINUTE:
            oldest = row.get("oldest_created_at") or window_start
            try:
                oldest_dt = datetime.strptime(oldest, "%Y-%m-%d %H:%M:%S")
            except Exception:
                oldest_dt = window_start_dt
            delay = max(1.0, 61.0 - (datetime.utcnow() - oldest_dt).total_seconds())
            return delay
        conn.execute(
            "INSERT INTO api_request_logs(user_id, provider) VALUES (?, ?)",
            (user_id, provider),
        )
    return 0.0


def _parse_retry_after_seconds(response: requests.Response, attempt: int) -> float:
    header = (response.headers.get("Retry-After") or "").strip()
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            pass
    if response.status_code == 429:
        return min(60.0, 12.0 * attempt)
    return min(45.0, 2.0 * attempt)


def _response_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                return err.get("message") or json.dumps(err, ensure_ascii=False)
            return json.dumps(payload, ensure_ascii=False)
    except Exception:
        pass
    return (response.text or response.reason or f"HTTP {response.status_code}").strip()


def _post_groq_transcription(api_key: str, model_name: str, file_path: str, mime_type: str, timeout: int) -> requests.Response:
    last_error = "Groq 转写请求失败"
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            with open(file_path, "rb") as handle:
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={
                        "model": model_name,
                        "response_format": "verbose_json",
                        "temperature": "0",
                    },
                    files={"file": (os.path.basename(file_path), handle, mime_type)},
                    timeout=timeout,
                )
            if response.ok:
                return response
            if response.status_code in MODEL_RETRYABLE_STATUS_CODES and attempt < GROQ_MAX_RETRIES:
                time.sleep(_parse_retry_after_seconds(response, attempt))
                continue
            _raise_groq_error(response)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = str(exc)
            if attempt < GROQ_MAX_RETRIES:
                time.sleep(min(20.0, 2.0 * attempt))
                continue
            raise RuntimeError(f"Groq 网络请求失败：{last_error}")
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            break
    raise RuntimeError(last_error)


def _summarize_payload(payload: object, limit: int = 600) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


class RetryableModelPayloadError(RuntimeError):
    def __init__(self, message: str, wait_seconds: float = 0.0):
        super().__init__(message)
        self.wait_seconds = wait_seconds


def _payload_error_retry_delay(error_obj: dict) -> float:
    code = str(error_obj.get("code") or "").strip()
    message = str(error_obj.get("message") or "").lower()
    if code in {"408", "429", "500", "502", "503", "504"}:
        return 18.0
    if "rate increased too quickly" in message:
        return 20.0
    if "upstream error" in message:
        return 15.0
    return 0.0


def _extract_model_content(payload: object) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError(f"模型返回非 JSON 对象：{_summarize_payload(payload)}")

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        parts.append(str(item["text"]))
                merged = "".join(parts).strip()
                if merged:
                    return merged

    # Some providers may expose a top-level error or a different payload shape even on HTTP 200.
    error_obj = payload.get("error")
    if error_obj:
        if isinstance(error_obj, dict):
            retry_delay = _payload_error_retry_delay(error_obj)
            summary = f"模型返回错误对象：{_summarize_payload(error_obj)}"
            if retry_delay > 0:
                raise RetryableModelPayloadError(summary, wait_seconds=retry_delay)
            raise RuntimeError(summary)
        raise RuntimeError(f"模型返回错误对象：{_summarize_payload(error_obj)}")

    for key in ("output_text", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise RuntimeError(f"模型返回缺少 choices/content：{_summarize_payload(payload)}")


def _run_yt_dlp(url: str, work_dir: str) -> dict:
    command = [
        "yt-dlp",
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(
        command,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "yt-dlp failed")
    return json.loads(result.stdout)


def _run_ffprobe_metadata(video_path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe 元数据读取失败")
    payload = json.loads(result.stdout or "{}")
    format_info = payload.get("format") or {}
    duration_raw = format_info.get("duration")
    try:
        duration = int(float(duration_raw)) if duration_raw is not None else 0
    except Exception:
        duration = 0
    filename = os.path.basename(video_path)
    tags = format_info.get("tags") or {}
    title = tags.get("title") or os.path.splitext(filename)[0]
    return {
        "title": title,
        "channel": "",
        "uploader": "",
        "duration": duration,
        "upload_date": "",
        "description": "",
        "chapters": [],
        "webpage_url": "",
        "thumbnail": "",
        "_local_filename": filename,
    }


def _probe_duration_seconds(media_path: str) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0
    try:
        return max(0, int(float((result.stdout or "").strip())))
    except Exception:
        return 0


def _transcode_audio(input_path: str, output_path: str, bitrate: str, work_dir: str) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            bitrate,
            output_path,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "音频预处理失败")


def _split_audio_for_groq(input_path: str, work_dir: str) -> list[str]:
    file_size = os.path.getsize(input_path)
    if file_size <= GROQ_MAX_FILE_BYTES:
        return [input_path]
    duration = _probe_duration_seconds(input_path)
    if duration <= 0:
        raise RuntimeError("音频过大且无法识别时长，无法自动切片转写")
    chunk_count = max(2, int(file_size / GROQ_MAX_FILE_BYTES) + 1)
    segment_seconds = max(300, int(duration / chunk_count) + 30)
    output_pattern = os.path.join(work_dir, "audio-transcribe-part-%02d.mp3")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-c",
            "copy",
            output_pattern,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "音频切片失败")
    parts = [
        os.path.join(work_dir, name)
        for name in sorted(os.listdir(work_dir))
        if name.startswith("audio-transcribe-part-") and name.endswith(".mp3")
    ]
    if not parts:
        raise RuntimeError("音频切片后未生成有效分片")
    return parts


def _build_local_video_info(job: dict) -> dict:
    video_path = _source_input(job)
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError("上传的视频文件不存在")
    info = _run_ffprobe_metadata(video_path)
    if job.get("task_name"):
        info["title"] = job["task_name"]
    return info


def _extract_subtitle_text(info: dict) -> str:
    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    for tracks in list(subtitles.values()) + list(automatic.values()):
        for item in tracks:
            if item.get("ext") in {"json3", "srv3", "ttml", "vtt"} and item.get("url"):
                response = requests.get(item["url"], timeout=20)
                response.raise_for_status()
                return _subtitle_to_plaintext(response.text)
    return ""


def _run_groq_transcription(source_input: str, work_dir: str, *, is_local_source: bool = False) -> str:
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    model_name = (os.environ.get("GROQ_TRANSCRIBE_MODEL") or "whisper-large-v3-turbo").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 未配置")
    audio_template = os.path.join(work_dir, "audio.%(ext)s")
    if is_local_source:
        if not os.path.exists(source_input):
            raise RuntimeError("上传视频文件不存在")
        audio_path = source_input
    else:
        download = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestaudio/best",
                "--extract-audio",
                "--audio-format",
                "mp3",
                "-o",
                audio_template,
                source_input,
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if download.returncode != 0:
            raise RuntimeError(download.stderr.strip() or "音频下载失败")

        audio_path = None
        for name in os.listdir(work_dir):
            if name.startswith("audio.") and name.endswith((".mp3", ".m4a", ".webm", ".wav")):
                audio_path = os.path.join(work_dir, name)
                break
    if not audio_path:
        raise RuntimeError("未找到已下载音频文件")

    normalized_audio = os.path.join(work_dir, "audio-transcribe.mp3")
    _transcode_audio(audio_path, normalized_audio, "64k", work_dir)
    if os.path.getsize(normalized_audio) > GROQ_MAX_FILE_BYTES:
        _transcode_audio(audio_path, normalized_audio, "32k", work_dir)

    transcripts = []
    for part_path in _split_audio_for_groq(normalized_audio, work_dir):
        response = _post_groq_transcription(
            api_key=api_key,
            model_name=model_name,
            file_path=part_path,
            mime_type="audio/mpeg",
            timeout=300,
        )
        _raise_groq_error(response)
        payload = response.json()
        transcript = (payload.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("Groq 未返回有效转写文本")
        transcripts.append(transcript)
    return "\n".join(item for item in transcripts if item.strip()).strip()


def run_groq_self_test(work_dir: str) -> dict:
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    model_name = (os.environ.get("GROQ_TRANSCRIBE_MODEL") or "whisper-large-v3-turbo").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 未配置")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 未安装")

    os.makedirs(work_dir, exist_ok=True)
    audio_path = os.path.join(work_dir, "groq-self-test.wav")
    generate = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "1",
            audio_path,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if generate.returncode != 0:
        raise RuntimeError(generate.stderr.strip() or "测试音频生成失败")

    response = _post_groq_transcription(
        api_key=api_key,
        model_name=model_name,
        file_path=audio_path,
        mime_type="audio/wav",
        timeout=120,
    )
    _raise_groq_error(response)
    payload = response.json()
    return {
        "model": model_name,
        "text": (payload.get("text") or "").strip(),
        "request_id": ((payload.get("x_groq") or {}).get("id") if isinstance(payload, dict) else "") or "",
    }


def _raise_groq_error(response: requests.Response) -> None:
    if response.ok:
        return
    body = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                body = error_obj.get("message") or json.dumps(error_obj, ensure_ascii=False)
            else:
                body = json.dumps(payload, ensure_ascii=False)
        else:
            body = str(payload)
    except Exception:
        body = response.text.strip()
    body = (body or "").strip()
    summary = f"Groq API 返回 {response.status_code}"
    if body:
        summary = f"{summary}: {body[:500]}"
    raise RuntimeError(summary)


def _subtitle_to_plaintext(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "-->" in stripped:
            continue
        if stripped.isdigit():
            continue
        stripped = stripped.replace("&nbsp;", " ").replace("&amp;", "&")
        if "<" in stripped and ">" in stripped:
            while "<" in stripped and ">" in stripped:
                start = stripped.find("<")
                end = stripped.find(">", start)
                if end == -1:
                    break
                stripped = stripped[:start] + stripped[end + 1 :]
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _split_transcript(transcript: str, target_size: int = TRANSCRIPT_CHUNK_TARGET) -> list[str]:
    text = re.sub(r"\s+", " ", (transcript or "").strip())
    if not text:
        return []
    if len(text) <= target_size:
        return [text]

    sentences = re.split(r"(?<=[。！？；.!?])", text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current:
            current = sentence
            continue
        if len(current) + len(sentence) <= target_size:
            current += sentence
            continue
        if len(current) < TRANSCRIPT_CHUNK_MIN and len(sentence) < target_size:
            current += sentence
            continue
        chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)

    normalized = []
    for chunk in chunks:
        if len(chunk) <= target_size:
            normalized.append(chunk)
            continue
        start = 0
        while start < len(chunk):
            normalized.append(chunk[start:start + target_size])
            start += target_size
    return normalized


def _build_user_prompt(job: dict, info: dict, transcript: str, chunk_index: int, chunk_count: int) -> str:
    mode = OUTPUT_MODE_SPECS.get(job.get("output_mode") or "full_notes", OUTPUT_MODE_SPECS["full_notes"])
    metadata = {
        "title": info.get("title", ""),
        "channel": info.get("channel", ""),
        "uploader": info.get("uploader", ""),
        "duration": info.get("duration", ""),
        "upload_date": info.get("upload_date", ""),
        "description": (info.get("description") or "")[:4000],
        "chapters": info.get("chapters") or [],
        "webpage_url": info.get("webpage_url") or job.get("source_url") or info.get("_local_filename", ""),
        "platform": job["platform"],
    }
    return json.dumps(
        {
            "task": "根据以下视频信息生成中文课程讲义，风格参考专业 LaTeX 教学讲义，尽量完整覆盖当前分段内容。",
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "output_mode": job.get("output_mode") or "full_notes",
            "metadata": metadata,
            "transcript_chunk": transcript,
            "writing_requirements": [
                mode["goal"],
                f"summary 输出 {mode['summary_range']}。",
                f"每个 section 的 bullets 输出 {mode['bullet_range']}。",
                f"当前分段至少输出 {mode['sections_min']} 个 sections。",
                "不要只写很短的摘要，要展开为适合学习的讲义。",
                "每个 section 尽量写出 goal、subsections、section_summary。",
                "subsections 中优先写清动机、核心思想、机制、例子、结论。",
                "可以使用 important、knowledge、warnings 来提炼高信号教学点。",
                "只有视频内容明确支持时才输出公式和代码。",
                "如果当前分段是在延续上文，请直接按内容组织，不要重复寒暄。",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _call_model(job: dict, secret_key: str, user_prompt: str, database_path: str) -> dict:
    api_base_url = (job.get("api_base_url") or "").strip().rstrip("/")
    model_name = (job.get("model_name") or "").strip()
    api_key = decrypt_secret(secret_key, job.get("api_key_encrypted") or "")
    if not api_base_url or not model_name or not api_key:
        raise RuntimeError("模型配置不完整")
    provider = _provider_name(api_base_url)

    endpoint = f"{api_base_url}/chat/completions"
    system_prompt = (job.get("system_prompt") or "").strip() or SKILL_SYSTEM_PROMPT
    payload = {
        "model": model_name,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    last_error = "模型请求失败"
    for attempt in range(1, MODEL_MAX_RETRIES + 1):
        wait_seconds = _check_and_record_rate_limit(database_path, job["user_id"], provider)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            continue

        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if response.ok:
            data = response.json()
            try:
                content = _extract_model_content(data)
                return json.loads(content)
            except RetryableModelPayloadError as exc:
                last_error = str(exc)
                if attempt < MODEL_MAX_RETRIES:
                    wait_seconds = exc.wait_seconds or min(30.0, 5.0 * attempt)
                    append_job_log(
                        database_path,
                        job["id"],
                        f"模型返回可重试上游错误，第 {attempt}/{MODEL_MAX_RETRIES} 次重试前等待 {wait_seconds:.0f} 秒",
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(f"模型返回内容不可解析：{last_error}")
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_error = str(exc)
                if attempt < MODEL_MAX_RETRIES:
                    wait_seconds = min(20.0, 2.5 * attempt)
                    append_job_log(
                        database_path,
                        job["id"],
                        f"模型返回异常内容，第 {attempt}/{MODEL_MAX_RETRIES} 次重试前等待 {wait_seconds:.1f} 秒",
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(f"模型返回内容不可解析：{last_error}")

        last_error = _response_error_message(response)
        if response.status_code in MODEL_RETRYABLE_STATUS_CODES and attempt < MODEL_MAX_RETRIES:
            wait_seconds = _parse_retry_after_seconds(response, attempt)
            append_job_log(
                database_path,
                job["id"],
                f"模型请求收到 HTTP {response.status_code}，第 {attempt}/{MODEL_MAX_RETRIES} 次重试前等待 {wait_seconds:.0f} 秒",
            )
            time.sleep(wait_seconds)
            continue
        raise RuntimeError(f"模型请求失败（HTTP {response.status_code}）：{last_error}")

    raise RuntimeError(last_error)


def _normalize_notes(payload: dict, fallback_title: str) -> dict:
    def normalize_string_list(value, *, max_items: int | None = None) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        result = [str(item).strip() for item in value if str(item).strip()]
        if max_items is not None:
            return result[:max_items]
        return result

    sections = payload.get("sections") if isinstance(payload, dict) else []
    normalized_sections = []
    if isinstance(sections, list):
        for item in sections:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading") or "未命名章节").strip()
            goal = str(item.get("goal") or "").strip()
            cleaned = normalize_string_list(item.get("bullets"), max_items=18)
            raw_subsections = item.get("subsections") or []
            normalized_subsections = []
            if isinstance(raw_subsections, list):
                for subsection in raw_subsections:
                    if not isinstance(subsection, dict):
                        continue
                    formula_obj = subsection.get("formula") or {}
                    if not isinstance(formula_obj, dict):
                        formula_obj = {}
                    code_obj = subsection.get("code") or {}
                    if not isinstance(code_obj, dict):
                        code_obj = {}
                    normalized_subsection = {
                        "heading": str(subsection.get("heading") or "未命名小节").strip(),
                        "paragraphs": normalize_string_list(subsection.get("paragraphs"), max_items=6),
                        "bullets": normalize_string_list(subsection.get("bullets"), max_items=8),
                        "important": normalize_string_list(subsection.get("important"), max_items=4),
                        "knowledge": normalize_string_list(subsection.get("knowledge"), max_items=4),
                        "warnings": normalize_string_list(subsection.get("warnings"), max_items=4),
                        "formula": {
                            "title": str(formula_obj.get("title") or "").strip(),
                            "expression": str(formula_obj.get("expression") or "").strip(),
                            "symbol_notes": normalize_string_list(formula_obj.get("symbol_notes"), max_items=8),
                        },
                        "code": {
                            "language": str(code_obj.get("language") or "text").strip() or "text",
                            "caption": str(code_obj.get("caption") or "").strip(),
                            "content": str(code_obj.get("content") or "").rstrip(),
                        },
                    }
                    if (
                        normalized_subsection["heading"]
                        or normalized_subsection["paragraphs"]
                        or normalized_subsection["bullets"]
                        or normalized_subsection["important"]
                        or normalized_subsection["knowledge"]
                        or normalized_subsection["warnings"]
                        or normalized_subsection["formula"]["expression"]
                        or normalized_subsection["code"]["content"]
                    ):
                        normalized_subsections.append(normalized_subsection)
            section_summary = normalize_string_list(item.get("section_summary"), max_items=6)
            if cleaned or normalized_subsections or section_summary or goal:
                normalized_sections.append(
                    {
                        "heading": heading,
                        "goal": goal,
                        "bullets": cleaned,
                        "subsections": normalized_subsections,
                        "section_summary": section_summary,
                    }
                )

    summary = normalize_string_list(payload.get("summary") if isinstance(payload, dict) else [], max_items=24)
    conclusion = normalize_string_list(payload.get("conclusion") if isinstance(payload, dict) else [], max_items=20)

    return {
        "title": str((payload or {}).get("title") or fallback_title).strip(),
        "summary": summary,
        "sections": normalized_sections,
        "conclusion": conclusion,
    }


def _dedupe_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = re.sub(r"\s+", " ", item.strip())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
        if limit and len(result) >= limit:
            break
    return result


def _merge_chunk_notes(info: dict, chunk_notes: list[dict]) -> dict:
    mode_key = chunk_notes[0].get("_mode", "full_notes") if chunk_notes else "full_notes"
    mode = OUTPUT_MODE_SPECS.get(mode_key, OUTPUT_MODE_SPECS["full_notes"])
    all_summary = []
    all_sections = []
    all_conclusion = []
    for idx, note in enumerate(chunk_notes, start=1):
        all_summary.extend(note.get("summary", []))
        for section in note.get("sections", []):
            heading = section.get("heading", "未命名章节").strip()
            if len(chunk_notes) > 1 and heading:
                heading = f"第{idx}部分：{heading}"
            all_sections.append(
                {
                    "heading": heading,
                    "goal": section.get("goal", ""),
                    "bullets": section.get("bullets", []),
                    "subsections": section.get("subsections", []),
                    "section_summary": section.get("section_summary", []),
                }
            )
        all_conclusion.extend(note.get("conclusion", []))

    return {
        "title": chunk_notes[0].get("title") if chunk_notes else f"课程笔记：{info.get('title') or '未命名视频'}",
        "summary": _dedupe_keep_order(all_summary, limit=mode["merge_summary_limit"]),
        "sections": all_sections,
        "conclusion": _dedupe_keep_order(all_conclusion, limit=mode["merge_conclusion_limit"]),
    }


def _generate_detailed_notes(job: dict, info: dict, transcript: str, secret_key: str, database_path: str, job_id: int) -> dict:
    chunks = _split_transcript(transcript)
    if not chunks:
        raise RuntimeError("转写结果为空，无法生成讲义")

    append_job_log(database_path, job_id, f"讲义生成将分为 {len(chunks)} 个文本分段")
    notes_per_chunk = []
    for idx, chunk in enumerate(chunks, start=1):
        ensure_job_not_cancelled(database_path, job_id)
        append_job_log(database_path, job_id, f"开始生成第 {idx}/{len(chunks)} 个分段讲义")
        user_prompt = _build_user_prompt(job, info, chunk, idx, len(chunks))
        chunk_note = _call_model(job, secret_key, user_prompt, database_path)
        normalized = _normalize_notes(
            chunk_note,
            fallback_title=f"课程笔记：{info.get('title') or '未命名视频'}",
        )
        normalized["_mode"] = job.get("output_mode") or "full_notes"
        if not normalized["sections"]:
            raise RuntimeError(f"第 {idx} 个分段未返回有效章节内容")
        notes_per_chunk.append(normalized)
        append_job_log(database_path, job_id, f"第 {idx}/{len(chunks)} 个分段讲义生成完成")

    return _merge_chunk_notes(info, notes_per_chunk)


def _fallback_outline(info: dict, transcript: str) -> dict:
    transcript_lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    transcript_excerpt = transcript_lines[:12]
    description = (info.get("description") or "").strip()
    summary = [
        f"视频标题：{info.get('title') or '未识别'}",
        f"作者/频道：{info.get('channel') or info.get('uploader') or '未识别'}",
        f"时长：{info.get('duration') or '未知'} 秒",
    ]
    if description:
        summary.append(f"简介摘要：{description[:120]}")
    sections = [
        {
            "heading": "内容概览",
            "goal": "建立对本视频主题、来源和素材情况的整体认识。",
            "bullets": summary,
            "subsections": [
                {
                    "heading": "基础信息",
                    "paragraphs": ["当前未能完成高质量模型整理，因此先保留可确认的元数据与内容线索。"],
                    "bullets": summary,
                    "important": ["这是一份保守版讲义，覆盖范围受可用素材限制。"],
                    "knowledge": [],
                    "warnings": ["未成功进入完整教学重写流程，部分细节可能缺失。"],
                    "formula": {"title": "", "expression": "", "symbol_notes": []},
                    "code": {"language": "text", "caption": "", "content": ""},
                }
            ],
            "section_summary": ["当前讲义首先提供可确认的来源和视频基础信息。"],
        }
    ]
    if info.get("chapters"):
        sections.append(
            {
                "heading": "章节信息",
                "goal": "保留视频原始章节线索，帮助后续定位主题。",
                "bullets": [
                    f"{chapter.get('title') or '未命名章节'}"
                    for chapter in info["chapters"][:12]
                ],
                "subsections": [],
                "section_summary": ["以上内容来自视频原始章节数据。"],
            }
        )
    if transcript_excerpt:
        sections.append(
            {
                "heading": "字幕摘录",
                "goal": "保留一小段原始文本线索，帮助用户判断素材质量。",
                "bullets": transcript_excerpt,
                "subsections": [
                    {
                        "heading": "可用文本片段",
                        "paragraphs": ["以下内容直接来自当前可获取到的字幕/转写文本，尚未充分重写为完整教学讲义。"],
                        "bullets": transcript_excerpt,
                        "important": [],
                        "knowledge": [],
                        "warnings": [],
                        "formula": {"title": "", "expression": "", "symbol_notes": []},
                        "code": {"language": "text", "caption": "", "content": ""},
                    }
                ],
                "section_summary": ["可用文本素材存在，但尚未被完整整理为高质量讲义。"],
            }
        )
    else:
        sections.append(
            {
                "heading": "素材不足提示",
                "goal": "解释为什么当前无法生成高质量内容。",
                "bullets": [
                    "当前没有拿到可用字幕，已根据视频元信息生成保守版讲义。",
                    "如需更高质量结果，请在用户配置中填写模型 API，并确保源视频可抓取字幕或转写。",
                ],
                "subsections": [],
                "section_summary": ["当前素材不足，无法继续向下重建完整教学逻辑。"],
            }
        )
    return {
        "title": f"课程笔记：{info.get('title') or '未命名视频'}",
        "summary": summary,
        "sections": sections,
        "conclusion": [
            "当前讲义基于可获取到的视频元信息和字幕片段整理而成。",
            "如果需要更完整的内容，可补充模型配置或选择带可用字幕的视频重新生成。",
        ],
    }


def _latex_escape(value: str) -> str:
    text = str(value or "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _download_thumbnail(info: dict, work_dir: str) -> str:
    thumbnail = info.get("thumbnail")
    if not thumbnail:
        return ""
    ext = os.path.splitext(urlparse(thumbnail).path)[1] or ".jpg"
    local_path = os.path.join(work_dir, f"cover{ext}")
    response = requests.get(thumbnail, timeout=20)
    response.raise_for_status()
    with open(local_path, "wb") as handle:
        handle.write(response.content)
    return local_path


def _generate_cover_from_video(video_path: str, work_dir: str) -> str:
    if not video_path or not os.path.exists(video_path):
        return ""
    output_path = os.path.join(work_dir, "cover.jpg")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "10",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            output_path,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not os.path.exists(output_path):
        return ""
    return output_path


def _download_video_for_frames(source_input: str, work_dir: str, *, is_local_source: bool = False) -> str:
    if is_local_source:
        if not os.path.exists(source_input):
            raise RuntimeError("上传视频文件不存在")
        local_name = os.path.basename(source_input)
        target_path = os.path.join(work_dir, f"frame-source{os.path.splitext(local_name)[1] or '.mp4'}")
        if os.path.abspath(source_input) != os.path.abspath(target_path):
            shutil.copy2(source_input, target_path)
        return target_path
    output_template = os.path.join(work_dir, "frame-source.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bv*[height<=480]/b[height<=480]/best[height<=480]/best",
            "--no-playlist",
            "-o",
            output_template,
            source_input,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "抽帧视频下载失败")
    for name in sorted(os.listdir(work_dir)):
        if name.startswith("frame-source."):
            return os.path.join(work_dir, name)
    raise RuntimeError("抽帧视频文件不存在")


def _frame_target_count(duration_seconds: int | None) -> int:
    target_count = 4
    if duration_seconds:
        if duration_seconds >= 3600:
            target_count = 8
        elif duration_seconds >= 1800:
            target_count = 6
        elif duration_seconds >= 900:
            target_count = 5
    return target_count


def _candidate_frame_count(target_count: int) -> int:
    return min(max(target_count + 4, target_count * FRAME_EXTRACTION_OVERSAMPLE), FRAME_EXTRACTION_MAX_CANDIDATES)


def _frame_signature(path: str, hash_size: int = FRAME_DEDUPE_HASH_SIZE) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize((hash_size, hash_size))
            pixels = list(grayscale.getdata())
    except Exception:
        return None
    if not pixels:
        return None
    average = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | int(value >= average)
    return bits, int(round(average))


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _frame_visual_stats(path: str) -> tuple[float, float] | None:
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
    except Exception:
        return None
    if not stat.mean or not stat.stddev:
        return None
    return float(stat.mean[0]), float(stat.stddev[0])


def _is_black_frame(path: str) -> bool:
    stats = _frame_visual_stats(path)
    if stats is None:
        return False
    brightness, contrast = stats
    return brightness <= FRAME_BLACK_BRIGHTNESS_THRESHOLD and contrast <= FRAME_BLACK_CONTRAST_THRESHOLD


def _trim_intro_outro_candidates(frame_paths: list[str], target_count: int) -> list[str]:
    if len(frame_paths) <= max(target_count + 1, 4):
        return frame_paths
    trim_each_side = max(1, int(len(frame_paths) * 0.12))
    max_trim = max(1, (len(frame_paths) - target_count) // 2)
    trim_each_side = min(trim_each_side, max_trim)
    if trim_each_side <= 0:
        return frame_paths
    trimmed = frame_paths[trim_each_side : len(frame_paths) - trim_each_side]
    return trimmed or frame_paths


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _frame_priority_score(path: str) -> tuple[float, dict[str, float]]:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((320, 180))
            edge_image = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
            grayscale = np.asarray(rgb.convert("L"), dtype=np.float32)
            rgb_array = np.asarray(rgb, dtype=np.float32)
            edge_array = np.asarray(edge_image, dtype=np.float32)
    except Exception:
        return 0.0, {"priority": 0.0}

    brightness = float(grayscale.mean())
    contrast = float(grayscale.std())
    edge_density = float((edge_array > 28).mean())
    whitespace_ratio = float((grayscale >= 215).mean())
    dark_ratio = float((grayscale <= 55).mean())

    quantized = (rgb_array // 32).astype(np.uint8)
    flat_quantized = quantized.reshape(-1, 3)
    unique_colors = int(len(np.unique(flat_quantized, axis=0)))
    low_palette_score = _clamp01((FRAME_PRIORITY_MAX_COLORS - unique_colors) / FRAME_PRIORITY_MAX_COLORS)

    red = rgb_array[:, :, 0]
    green = rgb_array[:, :, 1]
    blue = rgb_array[:, :, 2]
    green_ratio = float(((green > 70) & (green - red > 18) & (green - blue > 18)).mean())

    text_density = _clamp01(edge_density * 7.5 + contrast / 140.0)
    slide_score = (
        text_density * 0.9
        + whitespace_ratio * 0.95
        + low_palette_score * 0.85
        + _clamp01(brightness / 255.0) * 0.35
    )
    code_score = (
        text_density * 1.0
        + dark_ratio * 1.05
        + _clamp01(contrast / 110.0) * 0.45
        + low_palette_score * 0.35
    )
    board_score = (
        text_density * 0.85
        + green_ratio * 1.3
        + _clamp01(contrast / 110.0) * 0.35
    )
    formula_score = (
        text_density * 1.05
        + whitespace_ratio * 0.75
        + low_palette_score * 0.55
        + _clamp01((255.0 - abs(brightness - 210.0)) / 255.0) * 0.25
    )
    priority = max(slide_score, code_score, board_score, formula_score) + text_density * 0.25
    return priority, {
        "priority": priority,
        "text_density": text_density,
        "slide": slide_score,
        "code": code_score,
        "board": board_score,
        "formula": formula_score,
    }


def _prioritize_content_frames(frame_paths: list[str], keep_limit: int) -> list[str]:
    if not frame_paths or keep_limit <= 0:
        return []
    scored = []
    for index, path in enumerate(frame_paths):
        score, details = _frame_priority_score(path)
        scored.append((score, details.get("text_density", 0.0), -index, index, path))
    scored.sort(reverse=True)
    chosen = sorted(scored[:keep_limit], key=lambda item: item[3])
    return [item[4] for item in chosen]


def _section_text_blob(section: dict) -> str:
    parts = [
        str(section.get("heading") or ""),
        str(section.get("goal") or ""),
        " ".join(section.get("bullets") or []),
        " ".join(section.get("section_summary") or []),
    ]
    for subsection in section.get("subsections") or []:
        parts.extend(
            [
                str(subsection.get("heading") or ""),
                " ".join(subsection.get("paragraphs") or []),
                " ".join(subsection.get("bullets") or []),
                " ".join(subsection.get("important") or []),
                " ".join(subsection.get("knowledge") or []),
                " ".join(subsection.get("warnings") or []),
                str(((subsection.get("formula") or {}).get("title")) or ""),
                str(((subsection.get("formula") or {}).get("expression")) or ""),
                str(((subsection.get("code") or {}).get("caption")) or ""),
                str(((subsection.get("code") or {}).get("content")) or ""),
            ]
        )
    return " ".join(parts).lower()


def _section_preference_weights(section: dict) -> dict[str, float]:
    text = _section_text_blob(section)
    weights = {"slide": 0.0, "code": 0.0, "board": 0.0, "formula": 0.0}
    keyword_groups = {
        "code": ["代码", "编程", "实现", "函数", "脚本", "接口", "算法", "python", "java", "cpp", "rust", "sql"],
        "formula": ["公式", "推导", "定理", "证明", "方程", "数学", "符号", "计算", "矩阵", "概率", "损失函数"],
        "board": ["板书", "手写", "演算", "草图", "图解", "步骤", "证明", "推演"],
        "slide": ["ppt", "课件", "概念", "定义", "总结", "目录", "框架", "流程", "原理", "结构"],
    }
    for label, keywords in keyword_groups.items():
        for keyword in keywords:
            if keyword in text:
                weights[label] += 1.0
    if not any(weights.values()):
        weights["slide"] = 0.4
        weights["formula"] = 0.2
        weights["code"] = 0.2
    return weights


def _section_chunk_index(section: dict, chunk_count_hint: int) -> int:
    heading = str(section.get("heading") or "").strip()
    match = re.match(r"^第(\d+)部分：", heading)
    if match:
        try:
            return max(0, min(chunk_count_hint - 1, int(match.group(1)) - 1))
        except Exception:
            return 0
    return -1


def _dedupe_similar_frames(frame_paths: list[str], keep_limit: int) -> list[str]:
    if not frame_paths or keep_limit <= 0:
        return []
    kept: list[str] = []
    signatures: dict[str, tuple[int, int] | None] = {}
    for path in frame_paths:
        frame_signature = _frame_signature(path)
        signatures[path] = frame_signature
        if frame_signature is None:
            kept.append(path)
        else:
            is_duplicate = False
            for candidate in kept:
                candidate_signature = signatures.get(candidate)
                if candidate_signature is None:
                    continue
                frame_hash, frame_brightness = frame_signature
                candidate_hash, candidate_brightness = candidate_signature
                if (
                    _hamming_distance(frame_hash, candidate_hash) <= FRAME_DEDUPE_HAMMING_THRESHOLD
                    and abs(frame_brightness - candidate_brightness) <= FRAME_DEDUPE_BRIGHTNESS_THRESHOLD
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(path)
        if len(kept) >= keep_limit:
            break

    if len(kept) < keep_limit:
        for path in frame_paths:
            if path in kept:
                continue
            kept.append(path)
            if len(kept) >= keep_limit:
                break
    return kept[:keep_limit]


def _extract_frame_images(video_path: str, work_dir: str, duration_seconds: int | None) -> list[str]:
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    target_count = _frame_target_count(duration_seconds)
    candidate_count = _candidate_frame_count(target_count)
    interval = max(20, int((duration_seconds or 900) / (candidate_count + 1)))
    pattern = os.path.join(frames_dir, "frame-%03d.jpg")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps=1/{interval},scale='min(1280,iw)':-2",
            "-frames:v",
            str(candidate_count),
            pattern,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "关键帧抽取失败")
    frame_paths = [
        os.path.join(frames_dir, name)
        for name in sorted(os.listdir(frames_dir))
        if name.endswith(".jpg")
    ]
    trimmed_paths = _trim_intro_outro_candidates(frame_paths, target_count)
    filtered_paths = [path for path in trimmed_paths if not _is_black_frame(path)]
    candidate_paths = filtered_paths or trimmed_paths or frame_paths
    deduped_paths = _dedupe_similar_frames(candidate_paths, keep_limit=len(candidate_paths))
    selected = _prioritize_content_frames(deduped_paths, keep_limit=target_count)
    selected_set = set(selected)
    for path in frame_paths:
        if path not in selected_set and os.path.exists(path):
            os.remove(path)
    return selected


def _copy_frames_dir(src_dir: str, dst_dir: str) -> list[str]:
    if not src_dir or not os.path.isdir(src_dir):
        return []
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return [
            os.path.join(dst_dir, name)
            for name in sorted(os.listdir(dst_dir))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _build_section_images(frame_paths: list[str], sections: list[dict]) -> list[str]:
    if not frame_paths or not sections:
        return []
    frame_meta = []
    for index, path in enumerate(frame_paths):
        score, details = _frame_priority_score(path)
        frame_meta.append(
            {
                "index": index,
                "path": path,
                "priority": score,
                "details": details,
            }
        )

    chunk_markers = [value for value in (_section_chunk_index(section, len(sections)) for section in sections) if value >= 0]
    chunk_count_hint = (max(chunk_markers) + 1) if chunk_markers else 0
    used_paths: set[str] = set()
    picks = []

    for section_index, section in enumerate(sections):
        preferred = _section_preference_weights(section)
        chunk_index = _section_chunk_index(section, chunk_count_hint) if chunk_count_hint else -1

        if chunk_index >= 0 and chunk_count_hint > 0:
            start = int(chunk_index * len(frame_meta) / chunk_count_hint)
            end = max(start + 1, int((chunk_index + 1) * len(frame_meta) / chunk_count_hint))
            primary_pool = frame_meta[start:end]
        else:
            primary_pool = frame_meta

        def rank(pool: list[dict]) -> list[dict]:
            ranked = []
            for item in pool:
                if item["path"] in used_paths:
                    continue
                details = item["details"]
                weighted_score = item["priority"] * 0.55
                weighted_score += details.get("slide", 0.0) * preferred.get("slide", 0.0)
                weighted_score += details.get("code", 0.0) * preferred.get("code", 0.0)
                weighted_score += details.get("board", 0.0) * preferred.get("board", 0.0)
                weighted_score += details.get("formula", 0.0) * preferred.get("formula", 0.0)
                weighted_score += details.get("text_density", 0.0) * 0.15
                ranked.append((weighted_score, -item["index"], item))
            ranked.sort(reverse=True)
            return [item for _, _, item in ranked]

        ranked_primary = rank(primary_pool)
        ranked_fallback = rank(frame_meta)
        chosen = (ranked_primary or ranked_fallback or [None])[0]
        if chosen is None:
            continue
        used_paths.add(chosen["path"])
        picks.append(chosen["path"])
        if len(used_paths) >= len(frame_meta):
            used_paths.clear()

    return picks


def _ensure_frames_for_job(
    app,
    database_path: str,
    job_id: int,
    job: dict,
    info: dict,
    normalized_source_url: str,
    section_images: list[str],
) -> list[str]:
    if section_images:
        return section_images
    append_job_log(database_path, job_id, "缓存中没有关键帧，开始补抽正文配图")
    set_processing_stage(database_path, job_id, "extracting_frames")
    ensure_job_not_cancelled(database_path, job_id)
    video_path = _download_video_for_frames(
        _source_input(job),
        job["work_dir"],
        is_local_source=_is_local_source(job),
    )
    frame_paths = _extract_frame_images(video_path, job["work_dir"], info.get("duration"))
    asset_dir = os.path.join(app.config["ASSET_DIR"], str(abs(hash(normalized_source_url))))
    if normalized_source_url:
        os.makedirs(asset_dir, exist_ok=True)
        asset_frames_dir = os.path.join(asset_dir, "frames")
        _copy_frames_dir(os.path.join(job["work_dir"], "frames"), asset_frames_dir)
        cover_path = ""
        for candidate in ("cover.jpg", "cover.png", "cover.webp"):
            possible = os.path.join(job["work_dir"], candidate)
            if os.path.exists(possible):
                cover_path = possible
                break
        upsert_cached_asset(
            database_path,
            normalized_source_url,
            job["platform"],
            info.get("title") or "",
            os.path.join(asset_dir, "metadata.json") if os.path.exists(os.path.join(asset_dir, "metadata.json")) else (job.get("metadata_json_path") or ""),
            os.path.join(asset_dir, "transcript.txt") if os.path.exists(os.path.join(asset_dir, "transcript.txt")) else (job.get("transcript_path") or ""),
            cover_path or "",
            asset_frames_dir,
        )
    append_job_log(database_path, job_id, f"关键帧补抽完成，共生成 {len(frame_paths)} 张")
    return frame_paths


def _build_tex(job: dict, info: dict, notes: dict, cover_path: str, section_images: list[str] | None = None) -> str:
    def bullet_list(items):
        if not items:
            return "\\begin{itemize}\n\\item 信息不足\n\\end{itemize}"
        bullets = "\n".join(f"\\item {_latex_escape(item)}" for item in items)
        return f"\\begin{{itemize}}\n{bullets}\n\\end{{itemize}}"

    def render_box(env_name: str, title: str, items: list[str]) -> list[str]:
        if not items:
            return []
        lines = [f"\\begin{{{env_name}}}{{{_latex_escape(title)}}}"]
        lines.append(bullet_list(items))
        lines.append(f"\\end{{{env_name}}}")
        return lines

    def render_subsection(subsection: dict) -> list[str]:
        lines = [f"\\subsection{{{_latex_escape(subsection.get('heading', '未命名小节'))}}}"]
        for paragraph in subsection.get("paragraphs", []):
            lines.append(_latex_escape(paragraph))
            lines.append("")
        if subsection.get("bullets"):
            lines.append(bullet_list(subsection["bullets"]))
        lines.extend(render_box("importantbox", "必须掌握", subsection.get("important", [])))
        lines.extend(render_box("knowledgebox", "背景与补充", subsection.get("knowledge", [])))
        lines.extend(render_box("warningbox", "常见误区", subsection.get("warnings", [])))
        formula = subsection.get("formula") or {}
        if formula.get("expression"):
            lines.append(f"\\paragraph{{{_latex_escape(formula.get('title') or '公式说明')}}}")
            lines.append("先给出公式，再说明各个符号的含义。")
            lines.append(f"\\[{formula['expression']}\\]")
            if formula.get("symbol_notes"):
                lines.append(bullet_list(formula["symbol_notes"]))
        code = subsection.get("code") or {}
        if code.get("content"):
            caption = _latex_escape(code.get("caption") or "代码片段")
            language = _latex_escape(code.get("language") or "text")
            lines.append(f"\\lstset{{language={language}}}")
            lines.append("\\begin{lstlisting}[caption={" + caption + "}]")
            lines.append(code["content"])
            lines.append("\\end{lstlisting}")
        return lines

    sections = []
    sections.append("\\section{摘要}")
    sections.append(bullet_list(notes.get("summary", [])))
    section_images = section_images or []
    for idx, section in enumerate(notes.get("sections", [])):
        heading = _latex_escape(section.get("heading", "未命名章节"))
        sections.append(f"\\section{{{heading}}}")
        if section.get("goal"):
            sections.append(f"\\begin{{knowledgebox}}{{本章目标}}{_latex_escape(section['goal'])}\\end{{knowledgebox}}")
        sections.append(bullet_list(section.get("bullets", [])))
        if idx < len(section_images):
            sections.append("\\begin{figure}[h]")
            sections.append("\\centering")
            sections.append(
                f"\\includegraphics[width=0.82\\textwidth,height=0.34\\textheight,keepaspectratio]{{{_latex_escape(section_images[idx])}}}"
            )
            sections.append(f"\\caption{{{_latex_escape(section.get('heading', '章节配图'))}}}")
            sections.append("\\end{figure}")
        for subsection in section.get("subsections", []):
            sections.extend(render_subsection(subsection))
        sections.append("\\subsection{本章小结}")
        sections.append(bullet_list(section.get("section_summary", [])))
    sections.append("\\section{总结与延伸}")
    sections.append(bullet_list(notes.get("conclusion", [])))

    cover_block = ""
    if cover_path:
        cover_block = f"\\includegraphics[width=0.82\\textwidth,height=0.45\\textheight,keepaspectratio]{{{_latex_escape(cover_path)}}}\\par"

    duration_text = str(info.get("duration") or "未知")
    source_label = job.get("source_url") or info.get("_local_filename") or "本地上传视频"
    source_display = _latex_escape(source_label)
    source_tex = source_display
    if job.get("source_url"):
        source_tex = f"\\href{{{_latex_escape(job['source_url'])}}}{{\\nolinkurl{{{_latex_escape(job['source_url'])}}}}}"
    return f"""\\documentclass[a4paper]{{article}}
\\usepackage[fontset=fandol]{{ctex}}
\\usepackage{{amsmath, amssymb, graphicx, hyperref, geometry, enumitem, listings, xcolor}}
\\usepackage[most]{{tcolorbox}}
\\geometry{{margin=2.3cm}}
\\setlist[itemize]{{leftmargin=2em}}
\\newtcolorbox{{knowledgebox}}[1]{{enhanced,colback=blue!5!white,colframe=blue!70!black,colbacktitle=blue!70!black,coltitle=white,fonttitle=\\bfseries,title=#1,attach boxed title to top left={{yshift=-2mm,xshift=2mm}},boxrule=1pt,sharp corners}}
\\newtcolorbox{{importantbox}}[1]{{enhanced,colback=yellow!10!white,colframe=yellow!80!black,colbacktitle=yellow!80!black,coltitle=black,fonttitle=\\bfseries,title=#1,sharp corners}}
\\newtcolorbox{{warningbox}}[1]{{enhanced,colback=red!5!white,colframe=red!75!black,colbacktitle=red!75!black,coltitle=white,fonttitle=\\bfseries,title=#1,sharp corners}}
\\lstset{{basicstyle=\\ttfamily\\small,keywordstyle=\\color{{blue}},stringstyle=\\color{{red!60!black}},commentstyle=\\color{{green!60!black}},breaklines=true,frame=single,numbers=left,numberstyle=\\tiny\\color{{gray}},captionpos=b}}
\\begin{{document}}
\\begin{{titlepage}}
\\centering
{{\\Large 课程笔记\\par}}
\\vspace{{1.2cm}}
{{\\huge\\bfseries {_latex_escape(notes.get("title") or info.get("title") or "课程笔记")}\\par}}
\\vspace{{0.8cm}}
{{\\large 五道口纳什 \\& 服务端生成器\\par}}
{{\\large 生成时间：{_latex_escape(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))}\\par}}
\\vspace{{1.0cm}}
{cover_block}
\\vfill
\\begin{{tcolorbox}}[width=0.9\\textwidth, colback=black!2!white, colframe=black!60, sharp corners]
\\textbf{{视频作者/频道}}：{_latex_escape(info.get("channel") or info.get("uploader") or "未知")}\\par
\\textbf{{发布日期}}：{_latex_escape(info.get("upload_date") or "未知")}\\par
\\textbf{{视频时长}}：{_latex_escape(duration_text)}\\par
\\textbf{{视频来源}}：{source_tex}\\par
\\end{{tcolorbox}}
\\end{{titlepage}}
\\tableofcontents
\\newpage
{os.linesep.join(sections)}
\\end{{document}}
"""


def _compile_xelatex(tex_path: str, work_dir: str) -> str:
    if not shutil.which("xelatex"):
        raise RuntimeError("xelatex 未安装")
    for _ in range(2):
        result = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                os.path.basename(tex_path),
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout[-1200:] or "XeLaTeX 编译失败"
            raise RuntimeError(message)
    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    if not os.path.exists(pdf_path):
        raise RuntimeError("XeLaTeX 未产出 PDF")
    return pdf_path


def _render_html(job: dict, info: dict, notes: dict, section_images: list[str] | None = None) -> str:
    def asset_url(path: str) -> str:
        if not path:
            return ""
        rel_path = os.path.relpath(path, job["work_dir"]).replace(os.sep, "/")
        return f"/jobs/{job['id']}/asset/{rel_path}"

    title = html.escape(notes.get("title") or info.get("title") or "课程笔记")
    summary_html = "".join(f"<li>{html.escape(item)}</li>" for item in notes.get("summary", []))
    sections_html = []
    section_images = section_images or []

    def render_box(class_name: str, title: str, items: list[str]) -> str:
        if not items:
            return ""
        box_items = "".join(f"<li>{html.escape(item)}</li>" for item in items)
        return f'<div class="{class_name}"><strong>{html.escape(title)}</strong><ul>{box_items}</ul></div>'

    def render_subsection(subsection: dict) -> str:
        paragraphs = "".join(f"<p>{html.escape(item)}</p>" for item in subsection.get("paragraphs", []))
        bullets = ""
        if subsection.get("bullets"):
            bullet_items = "".join(f"<li>{html.escape(item)}</li>" for item in subsection["bullets"])
            bullets = f"<ul>{bullet_items}</ul>"
        formula_html = ""
        formula = subsection.get("formula") or {}
        if formula.get("expression"):
            symbol_notes = "".join(f"<li>{html.escape(item)}</li>" for item in formula.get("symbol_notes", []))
            formula_html = (
                f'<div class="formula"><h4>{html.escape(formula.get("title") or "公式")}</h4>'
                f'<pre>{html.escape(formula["expression"])}</pre>'
                f'<ul>{symbol_notes}</ul></div>'
            )
        code_html = ""
        code = subsection.get("code") or {}
        if code.get("content"):
            code_html = (
                f'<div class="code-block"><div class="code-caption">{html.escape(code.get("caption") or "代码片段")}</div>'
                f'<pre><code>{html.escape(code["content"])}</code></pre></div>'
            )
        return (
            f'<div class="subsection"><h3>{html.escape(subsection.get("heading", "未命名小节"))}</h3>'
            f"{paragraphs}{bullets}"
            f'{render_box("importantbox-html", "必须掌握", subsection.get("important", []))}'
            f'{render_box("knowledgebox-html", "背景与补充", subsection.get("knowledge", []))}'
            f'{render_box("warningbox-html", "常见误区", subsection.get("warnings", []))}'
            f"{formula_html}{code_html}</div>"
        )

    for idx, section in enumerate(notes.get("sections", [])):
        bullets = "".join(f"<li>{html.escape(item)}</li>" for item in section.get("bullets", []))
        image_html = ""
        if idx < len(section_images):
            section_image_url = asset_url(section_images[idx])
            image_html = f'<div class="illustration"><img src="{html.escape(section_image_url)}" alt="section image"></div>'
        goal_html = ""
        if section.get("goal"):
            goal_html = f'<div class="knowledgebox-html"><strong>本章目标</strong><p>{html.escape(section["goal"])}</p></div>'
        subsection_html = "".join(render_subsection(item) for item in section.get("subsections", []))
        section_summary = "".join(f"<li>{html.escape(item)}</li>" for item in section.get("section_summary", []))
        sections_html.append(
            f"<section><h2>{html.escape(section.get('heading', '未命名章节'))}</h2>{goal_html}{image_html}<ul>{bullets}</ul>{subsection_html}<div class=\"section-summary\"><h3>本章小结</h3><ul>{section_summary}</ul></div></section>"
        )
    conclusion_html = "".join(f"<li>{html.escape(item)}</li>" for item in notes.get("conclusion", []))
    cover = ""
    cover_path = os.path.join(job["work_dir"], "cover.jpg")
    if not os.path.exists(cover_path):
        cover_path = os.path.join(job["work_dir"], "cover.png")
    if os.path.exists(cover_path):
        cover = f'<div class="cover"><img src="{html.escape(asset_url(cover_path))}" alt="thumbnail"></div>'
    elif info.get("thumbnail"):
        cover = f'<div class="cover"><img src="{html.escape(info["thumbnail"])}" alt="thumbnail"></div>'
    meta_pairs = [
        ("来源", job.get("source_url") or info.get("_local_filename") or "本地上传视频"),
        ("平台", job["platform"]),
        ("频道", info.get("channel") or info.get("uploader") or "未知"),
        ("上传日期", info.get("upload_date") or "未知"),
    ]
    meta_html = "".join(
        f"<li><strong>{html.escape(label)}：</strong>{html.escape(str(value))}</li>" for label, value in meta_pairs
    )
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    @page {{ size: A4; margin: 18mm; }}
    body {{ font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", sans-serif; color: #1f2937; line-height: 1.7; }}
    h1, h2 {{ color: #0f172a; }}
    h1 {{ font-size: 28px; margin-bottom: 10px; }}
    h2 {{ font-size: 18px; margin-top: 24px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }}
    .meta, .summary, section, .footer {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; margin: 16px 0; }}
    .cover, .illustration {{ margin: 18px 0; text-align: center; }}
    .cover img, .illustration img {{ max-width: 100%; max-height: 280px; border-radius: 12px; }}
    ul {{ margin: 8px 0 0 20px; }}
    .subtle {{ color: #475569; font-size: 12px; }}
    .meta strong {{ word-break: break-all; }}
    .meta li {{ overflow-wrap: anywhere; }}
    .subsection {{ margin-top: 18px; padding-top: 8px; border-top: 1px dashed #cbd5e1; }}
    .subsection h3, .section-summary h3 {{ font-size: 16px; margin: 8px 0; color: #0f172a; }}
    .knowledgebox-html, .importantbox-html, .warningbox-html, .formula, .code-block, .section-summary {{ margin: 14px 0; padding: 12px 14px; border-radius: 12px; }}
    .knowledgebox-html {{ background: #eff6ff; border: 1px solid #93c5fd; }}
    .importantbox-html {{ background: #fef9c3; border: 1px solid #facc15; }}
    .warningbox-html {{ background: #fee2e2; border: 1px solid #f87171; }}
    .formula {{ background: #eef2ff; border: 1px solid #a5b4fc; }}
    .formula pre, .code-block pre {{ white-space: pre-wrap; overflow-x: auto; }}
    .code-block {{ background: #0f172a; color: #e2e8f0; border: 1px solid #334155; }}
    .code-caption {{ margin-bottom: 8px; color: #cbd5e1; font-weight: 600; }}
    .section-summary {{ background: #f8fafc; border: 1px dashed #94a3b8; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {cover}
  <div class="meta">
    <ul>{meta_html}</ul>
  </div>
  <div class="summary">
    <h2>摘要</h2>
    <ul>{summary_html}</ul>
  </div>
  {''.join(sections_html)}
  <div class="footer">
    <h2>总结与延伸</h2>
    <ul>{conclusion_html}</ul>
    <p class="subtle">生成时间：{generated_at}</p>
  </div>
</body>
</html>"""


def process_job(app, job_id: int, reuse_job: dict | None = None) -> None:
    database_path = app.config["DATABASE"]
    secret_key = app.config["SECRET_KEY"]
    job = fetch_job_with_settings(database_path, job_id)
    if not job:
        return

    try:
        ensure_job_not_cancelled(database_path, job_id)
        update_job(
            database_path,
            job_id,
            status="running",
            error_message="",
            warning_message="",
            processing_stage="starting",
        )
        metadata_path = os.path.join(job["work_dir"], "metadata.json")
        transcript_path = os.path.join(job["work_dir"], "transcript.txt")
        source_input = _source_input(job)
        local_source = _is_local_source(job)
        normalized_source_url = job.get("normalized_source_url") or normalize_source_url(job["source_url"])
        cached_asset = fetch_cached_asset(database_path, normalized_source_url) if (normalized_source_url and not local_source) else None
        frames_dir = os.path.join(job["work_dir"], "frames")
        section_images = []

        if reuse_job or job.get("fetch_strategy") == "reuse":
            source = reuse_job or cached_asset
            if not source and job.get("metadata_json_path") and job.get("transcript_path"):
                source = job
            if not source:
                raise RuntimeError("复用失败：没有可用的缓存或历史任务")
            source_label = f"任务 #{source['id']}" if source.get("id") else "链接缓存"
            append_job_log(database_path, job_id, f"复用{source_label}的中间文件")
            if not _copy_if_exists(source.get("metadata_json_path") or "", metadata_path):
                raise RuntimeError("复用失败：历史元数据文件不存在")
            if not _copy_if_exists(source.get("transcript_path") or "", transcript_path):
                raise RuntimeError("复用失败：历史转写文件不存在")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            with open(transcript_path, "r", encoding="utf-8") as handle:
                transcript = handle.read().strip()
            if not transcript:
                raise RuntimeError("复用失败：历史转写内容为空")
            cover_src = source.get("cover_path") or os.path.join(source.get("work_dir", ""), "cover.jpg")
            if cover_src and os.path.exists(cover_src):
                _copy_if_exists(cover_src, os.path.join(job["work_dir"], os.path.basename(cover_src)))
            section_images = _copy_frames_dir(source.get("frames_dir") or os.path.join(source.get("work_dir", ""), "frames"), frames_dir)
            update_job(
                database_path,
                job_id,
                metadata_json_path=metadata_path,
                transcript_path=transcript_path,
                title=info.get("title") or source.get("title") or "",
            )
            append_job_log(database_path, job_id, "历史元数据和转写已复用")
            section_images = _ensure_frames_for_job(
                app,
                database_path,
                job_id,
                job,
                info,
                normalized_source_url,
                section_images,
            )
        elif cached_asset and job.get("fetch_strategy") == "auto":
            set_processing_stage(database_path, job_id, "reusing_cache")
            append_job_log(database_path, job_id, "检测到链接级缓存，直接复用 metadata/transcript")
            if not _copy_if_exists(cached_asset.get("metadata_json_path") or "", metadata_path):
                raise RuntimeError("缓存损坏：metadata 文件不存在")
            if not _copy_if_exists(cached_asset.get("transcript_path") or "", transcript_path):
                raise RuntimeError("缓存损坏：transcript 文件不存在")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            with open(transcript_path, "r", encoding="utf-8") as handle:
                transcript = handle.read().strip()
            if not transcript:
                raise RuntimeError("缓存损坏：转写内容为空")
            cover_src = cached_asset.get("cover_path") or ""
            if cover_src and os.path.exists(cover_src):
                _copy_if_exists(cover_src, os.path.join(job["work_dir"], os.path.basename(cover_src)))
            section_images = _copy_frames_dir(cached_asset.get("frames_dir") or "", frames_dir)
            update_job(database_path, job_id, metadata_json_path=metadata_path, transcript_path=transcript_path, title=info.get("title") or "")
            section_images = _ensure_frames_for_job(
                app,
                database_path,
                job_id,
                job,
                info,
                normalized_source_url,
                section_images,
            )
        else:
            set_processing_stage(database_path, job_id, "fetching_metadata")
            ensure_job_not_cancelled(database_path, job_id)
            append_job_log(database_path, job_id, "开始拉取视频元数据")
            if local_source:
                info = _build_local_video_info(job)
            else:
                info = _run_yt_dlp(source_input, job["work_dir"])
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(info, handle, ensure_ascii=False, indent=2)
            update_job(
                database_path,
                job_id,
                metadata_json_path=metadata_path,
                title=(job.get("task_name") or info.get("title") or ""),
            )
            append_job_log(database_path, job_id, "视频元数据已保存")

            set_processing_stage(database_path, job_id, "fetching_transcript")
            ensure_job_not_cancelled(database_path, job_id)
            append_job_log(database_path, job_id, "尝试抓取字幕")
            transcript = "" if local_source else _extract_subtitle_text(info)
            if not transcript.strip():
                append_job_log(database_path, job_id, "字幕不可用，开始 Groq Whisper 转写")
                transcript = _run_groq_transcription(source_input, job["work_dir"], is_local_source=local_source)
                append_job_log(database_path, job_id, "Groq Whisper 转写成功")

            with open(transcript_path, "w", encoding="utf-8") as handle:
                handle.write(transcript)
            update_job(database_path, job_id, transcript_path=transcript_path)
            asset_metadata_path = ""
            asset_transcript_path = ""
            asset_cover_path = ""
            asset_frames_dir = ""
            if normalized_source_url and not local_source:
                asset_dir = os.path.join(app.config["ASSET_DIR"], str(abs(hash(normalized_source_url))))
                os.makedirs(asset_dir, exist_ok=True)
                asset_metadata_path = os.path.join(asset_dir, "metadata.json")
                asset_transcript_path = os.path.join(asset_dir, "transcript.txt")
                shutil.copy2(metadata_path, asset_metadata_path)
                shutil.copy2(transcript_path, asset_transcript_path)
            cover_path = _download_thumbnail(info, job["work_dir"])
            if not cover_path and local_source:
                cover_path = _generate_cover_from_video(source_input, job["work_dir"])
            if cover_path and os.path.exists(cover_path) and normalized_source_url and not local_source:
                asset_cover_path = os.path.join(asset_dir, os.path.basename(cover_path))
                shutil.copy2(cover_path, asset_cover_path)
            set_processing_stage(database_path, job_id, "extracting_frames")
            append_job_log(database_path, job_id, "开始抽取关键帧配图")
            video_path = _download_video_for_frames(source_input, job["work_dir"], is_local_source=local_source)
            frame_paths = _extract_frame_images(video_path, job["work_dir"], info.get("duration"))
            section_images = frame_paths
            if normalized_source_url and not local_source:
                asset_frames_dir = os.path.join(asset_dir, "frames")
                _copy_frames_dir(os.path.join(job["work_dir"], "frames"), asset_frames_dir)
                upsert_cached_asset(
                    database_path,
                    normalized_source_url,
                    job["platform"],
                    info.get("title") or "",
                    asset_metadata_path,
                    asset_transcript_path,
                    asset_cover_path,
                    asset_frames_dir,
                )

        set_processing_stage(database_path, job_id, "generating_notes")
        ensure_job_not_cancelled(database_path, job_id)
        append_job_log(database_path, job_id, "开始按分段生成完整讲义")
        notes = _generate_detailed_notes(job, info, transcript, secret_key, database_path, job_id)
        append_job_log(database_path, job_id, "完整讲义内容已生成")
        section_images = _build_section_images(section_images, notes.get("sections", []))

        set_processing_stage(database_path, job_id, "rendering_output")
        ensure_job_not_cancelled(database_path, job_id)
        output_basename = _slugify_filename(_job_display_title(job, info, notes), fallback=f"job-{job_id}")
        notes_html = _render_html(job, info, notes, section_images)
        notes_html_path = os.path.join(job["work_dir"], f"{output_basename}.html")
        notes_tex_path = os.path.join(job["work_dir"], f"{output_basename}.tex")
        with open(notes_html_path, "w", encoding="utf-8") as handle:
            handle.write(notes_html)

        cover_path = os.path.join(job["work_dir"], "cover.jpg")
        if not os.path.exists(cover_path):
            cover_path = _download_thumbnail(info, job["work_dir"])
        if not cover_path and local_source:
            cover_path = _generate_cover_from_video(source_input, job["work_dir"])
        tex_source = _build_tex(job, info, notes, cover_path, section_images)
        with open(notes_tex_path, "w", encoding="utf-8") as handle:
            handle.write(tex_source)

        set_processing_stage(database_path, job_id, "compiling_pdf")
        ensure_job_not_cancelled(database_path, job_id)
        append_job_log(database_path, job_id, "开始 XeLaTeX 编译")
        compiled_pdf_path = _compile_xelatex(notes_tex_path, job["work_dir"])
        notes_pdf_path = os.path.join(job["work_dir"], f"{output_basename}.pdf")
        if os.path.abspath(compiled_pdf_path) != os.path.abspath(notes_pdf_path):
            shutil.move(compiled_pdf_path, notes_pdf_path)
        append_job_log(database_path, job_id, "XeLaTeX 编译成功")
        update_job(
            database_path,
            job_id,
            status="completed",
            processing_stage="completed",
            title=_job_display_title(job, info, notes),
            notes_html_path=notes_html_path,
            notes_pdf_path=notes_pdf_path,
        )
        append_job_log(database_path, job_id, "PDF 已生成")
    except JobCancelled as exc:
        update_job(
            database_path,
            job_id,
            status="cancelled",
            processing_stage="cancelled",
            error_message="",
            warning_message="",
        )
        append_job_log(database_path, job_id, f"任务已取消：{exc}")
    except Exception as exc:
        update_job(
            database_path,
            job_id,
            status="failed",
            processing_stage="failed",
            error_message=str(exc),
            warning_message="",
        )
        append_job_log(database_path, job_id, f"任务失败：{exc}")
