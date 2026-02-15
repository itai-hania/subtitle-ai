"""
Video Subtitle Generator API
Supports OpenAI Whisper for transcription and OpenAI/Ollama for translation
"""

import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import ipaddress
from pathlib import Path
from typing import Literal, Optional

import ollama
import openai
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from downloader import (
    download_video as download_video_from_url,
    get_video_info,
    is_valid_url,
)
from processing import (
    embed_subtitles,
    extract_audio,
    generate_srt,
    has_audio_stream,
    process_srt_for_rtl,
    srt_to_plain_text,
    subtitles_to_srt,
    transcribe_audio,
    validate_audio_size,
)
from trimmer import get_video_duration, get_video_info as get_video_file_info, trim_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Valid job ID pattern (full UUID: 8-4-4-4-12 hex digits)
JOB_ID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# Allowed target languages for translation (prevents LLM prompt injection)
ALLOWED_LANGUAGES = {
    "english", "hebrew", "spanish", "french", "german", "italian",
    "portuguese", "russian", "chinese", "japanese", "korean", "arabic",
    "turkish", "dutch", "polish", "swedish", "norwegian", "danish",
    "finnish", "greek", "czech", "romanian", "hungarian", "thai",
    "vietnamese", "indonesian", "malay", "hindi", "bengali", "ukrainian",
}

# Allowed translation services
ALLOWED_TRANSLATION_SERVICES = {"openai", "ollama"}

# Allowed Ollama models
ALLOWED_OLLAMA_MODELS = {
    "llama3.2", "llama3.1", "llama3", "llama2",
    "mistral", "mixtral", "gemma", "gemma2",
    "phi3", "phi", "qwen2", "qwen",
    "command-r", "aya",
}

# Allowed download qualities
ALLOWED_QUALITIES = {"720p", "1080p", "best"}

# Allowed video file extensions
ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
ALLOWED_VIDEO_SUFFIXES = {ext.lower() for ext in ALLOWED_VIDEO_EXTENSIONS}

# Safe filename pattern: alphanumeric, hyphens, underscores, dots
SAFE_FILENAME_PATTERN = re.compile(r'[^a-zA-Z0-9._-]')

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean environment variables safely."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Parse positive integer environment variables safely."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default=%d", name, value, default)
        return default


# Job and limiter capacity tuning
MAX_CONCURRENT_JOBS = _env_int("MAX_CONCURRENT_JOBS", 100)
MAX_STORED_JOBS = _env_int("MAX_STORED_JOBS", 1000)
RATE_LIMIT_MAX_KEYS = _env_int("RATE_LIMIT_MAX_KEYS", 10_000)
TRUST_PROXY_HEADERS = _env_flag("TRUST_PROXY_HEADERS", False)

# Statuses that count as active workload
ACTIVE_JOB_STATUSES = {
    "queued",
    "downloading",
    "extracting_audio",
    "transcribing",
    "translating",
    "embedding_subtitles",
    "reburning",
}

# Upload size limit (500 MB)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start cleanup scheduler on startup."""
    import asyncio

    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)
            cleanup_old_jobs()

    task = asyncio.create_task(periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(
    title="Video Subtitle Generator",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob: https://*.ytimg.com https://*.twimg.com; "
            "media-src 'self' blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware — restrict to same-origin (no wildcard with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Accept"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
DOWNLOAD_DIR = Path("downloads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Watermark configuration
WATERMARK_CONFIG = {
    "logo_path": Path("logo.jpg"),
    "english_text": "@FinancialEduX",
    "hebrew_text": "המחנך הפיננסי",
    "opacity": 0.7,
    "logo_height": 50,
    "english_font_size": 16,
    "hebrew_font_size": 18,
    "margin": 15,
    "text_spacing": 22,
}

# Store job status
jobs: dict = {}
jobs_lock = threading.Lock()


# ============================================
# Rate Limiting
# ============================================

class RateLimiter:
    """Simple in-memory per-IP rate limiter."""

    def __init__(self, max_keys: int = RATE_LIMIT_MAX_KEYS) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def _prune_stale_keys(self, cutoff: float) -> None:
        """Drop keys whose latest request is older than the cutoff."""
        stale = [
            key for key, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for key in stale:
            self._requests.pop(key, None)

    def check(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        """Return True if request is allowed, False if rate limited."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = self._requests.get(key)
            if timestamps is None:
                if len(self._requests) >= self._max_keys:
                    self._prune_stale_keys(cutoff)
                    if len(self._requests) >= self._max_keys:
                        return False
                timestamps = deque()
                self._requests[key] = timestamps

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= max_requests:
                return False
            timestamps.append(now)
            return True


rate_limiter = RateLimiter()


def _normalize_ip(value: str) -> Optional[str]:
    """Return a normalized IP address string, or None if invalid."""
    try:
        return str(ipaddress.ip_address(value.strip()))
    except (ValueError, AttributeError):
        return None


def get_client_ip(request: Request) -> str:
    """Extract client IP safely, with optional trusted proxy headers."""
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            for part in forwarded.split(","):
                parsed = _normalize_ip(part)
                if parsed:
                    return parsed

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            parsed = _normalize_ip(real_ip)
            if parsed:
                return parsed

    client_host = request.client.host if request.client else ""
    parsed_client = _normalize_ip(client_host)
    return parsed_client or "unknown"


def check_rate_limit(request: Request, max_requests: int = 10) -> None:
    """Check rate limit and raise 429 if exceeded."""
    client_ip = get_client_ip(request)
    if not rate_limiter.check(client_ip, max_requests):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

# Job expiration
JOB_TTL = timedelta(hours=24)


class Progress:
    """Named constants for progress percentages."""
    DOWNLOADING = 5
    EXTRACTING_AUDIO = 10
    TRANSCRIBING = 30
    TRANSLATING = 50
    EMBEDDING = 70
    REBURN_START = 10
    REBURN_SRT = 30
    REBURN_EMBED = 50
    NEARLY_DONE = 90
    COMPLETED = 100


# Fail fast if API key is missing
_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. Add it to .env file.")

# OpenAI client
openai_client = openai.OpenAI(api_key=_openai_api_key)


# ============================================
# Dependencies
# ============================================

def validate_job_id(job_id: str) -> bool:
    """Validate job_id is safe (alphanumeric/hyphens only, no path traversal)."""
    return bool(JOB_ID_PATTERN.match(job_id))


def get_validated_job(job_id: str) -> dict:
    """FastAPI dependency: validate job_id and return the job dict, or raise 404."""
    if not validate_job_id(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ============================================
# Helpers
# ============================================

def count_active_jobs_locked() -> int:
    """Count active jobs. Caller must hold jobs_lock."""
    return sum(1 for job in jobs.values() if job.get("status") in ACTIVE_JOB_STATUSES)


def enforce_capacity(require_active_slot: bool = False) -> None:
    """Enforce stored-job and active-job capacity limits."""
    cleanup_old_jobs()
    with jobs_lock:
        if len(jobs) >= MAX_STORED_JOBS:
            raise HTTPException(status_code=429, detail="Server is full. Please try again later.")
        if require_active_slot and count_active_jobs_locked() >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="Server is busy. Please try again later.")


def _find_video_file_by_prefix(job_id: str, directories: list[Path]) -> Optional[str]:
    """Fallback: find first matching video file by job_id prefix."""
    for search_dir in directories:
        try:
            for file in search_dir.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
                    return str(file)
        except OSError:
            continue
    return None


def find_video_path(job_id: str) -> Optional[str]:
    """Find video file path for a job. Prefers output (subtitled) video if available."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        logger.debug("Job %s not found in jobs dict", job_id)
        return None

    # Prefer the processed output video (with burned subtitles)
    output_file = job.get("output_file")
    if output_file:
        output_path = OUTPUT_DIR / output_file
        if output_path.exists():
            logger.debug("Found output video: %s", output_path)
            return str(output_path)

    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        logger.debug("Found via job dict: %s", video_path)
        return video_path

    # Fallback: search uploads and downloads directories
    fallback_path = _find_video_file_by_prefix(job_id, [UPLOAD_DIR, DOWNLOAD_DIR])
    if fallback_path:
        logger.debug("Found via filesystem search: %s", fallback_path)
        return fallback_path

    logger.debug("No video found for job %s", job_id)
    return None


def cleanup_old_jobs() -> None:
    """Remove jobs older than JOB_TTL and their associated files."""
    now = datetime.now()
    with jobs_lock:
        expired = {
            jid: job.get("files", [])
            for jid, job in jobs.items()
            if now - job.get("created_at", now) > JOB_TTL
        }
    for jid, tracked_files in expired.items():
        if tracked_files:
            # Fast path: delete tracked files directly
            for file_path in tracked_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            # Fallback for jobs without tracked files
            for search_dir in [UPLOAD_DIR, DOWNLOAD_DIR, OUTPUT_DIR]:
                for file in search_dir.iterdir():
                    if file.name.startswith(jid):
                        try:
                            file.unlink()
                        except OSError:
                            pass
        with jobs_lock:
            jobs.pop(jid, None)
    if expired:
        logger.info("Cleaned up %d expired job(s)", len(expired))


def sanitize_filename(filename: str) -> str:
    """Strip directory components and restrict to safe characters."""
    # Take only the basename (strip any directory traversal)
    name = Path(filename).name
    # Replace unsafe characters with underscores
    name = SAFE_FILENAME_PATTERN.sub('_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name).strip('_')
    # Ensure non-empty
    if not name or name == '.':
        name = "upload.mp4"
    return name


def validate_path_containment(path: Path, allowed_dir: Path) -> None:
    """Ensure resolved path stays within the allowed directory."""
    resolved = path.resolve()
    allowed_resolved = allowed_dir.resolve()
    if not str(resolved).startswith(str(allowed_resolved) + os.sep) and resolved != allowed_resolved:
        raise HTTPException(status_code=400, detail="Invalid file path")


def validate_language(language: str) -> str:
    """Validate language against allowlist."""
    if language.lower() not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    return language


def validate_translation_service(service: str) -> str:
    """Validate translation service against allowlist."""
    if service.lower() not in ALLOWED_TRANSLATION_SERVICES:
        raise HTTPException(status_code=400, detail=f"Unsupported translation service: {service}")
    return service


def validate_ollama_model(model: str) -> str:
    """Validate Ollama model against allowlist."""
    if model.lower() not in ALLOWED_OLLAMA_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported Ollama model: {model}")
    return model


async def save_upload_chunked(upload: UploadFile, dest: Path) -> int:
    """Write an UploadFile to disk in 1MB chunks. Returns bytes written."""
    total = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)}MB."
                )
            f.write(chunk)
    return total


def _make_text_download_response(content: str, filename: str) -> Response:
    """Return text content as a downloadable file response (no temp file)."""
    safe_name = sanitize_filename(filename)
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


# ============================================
# Processing Pipeline
# ============================================

def process_video(
    job_id: str,
    video_path: str,
    target_language: str,
    use_ollama: bool,
    ollama_model: str,
) -> None:
    """Main processing pipeline."""
    start_time = time.time()
    temp_dir = None

    logger.info("Job %s — Processing video at: %s", job_id, video_path)
    vp = Path(video_path)
    logger.info("File exists: %s, size: %s", vp.exists(), vp.stat().st_size if vp.exists() else "N/A")

    try:
        with jobs_lock:
            jobs[job_id]["status"] = "extracting_audio"
            jobs[job_id]["progress"] = Progress.EXTRACTING_AUDIO

        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.mp3")
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        srt_path_burning = os.path.join(temp_dir, "subtitles_burning.srt")

        # Check for audio stream
        if not has_audio_stream(video_path):
            raise Exception("Video has no audio track. Cannot generate subtitles.")

        # Step 1: Extract audio
        extract_audio(video_path, audio_path)
        validate_audio_size(audio_path)

        with jobs_lock:
            jobs[job_id]["status"] = "transcribing"
            jobs[job_id]["progress"] = Progress.TRANSCRIBING

        # Step 2: Transcribe with Whisper
        transcript = transcribe_audio(openai_client, audio_path)

        with jobs_lock:
            jobs[job_id]["status"] = "translating"
            jobs[job_id]["progress"] = Progress.TRANSLATING

        # Step 3: Generate SRT (with translation if needed)
        segments = transcript.segments if hasattr(transcript, 'segments') else transcript.get('segments', [])
        if not segments:
            raise Exception("No speech detected in video. Cannot generate subtitles.")

        srt_content, original_srt_content, token_usage, subtitles_data = generate_srt(
            segments, target_language, openai_client, use_ollama, ollama_model
        )

        # Save standard SRT for download
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # Prepare SRT for burning (apply RTL markers if needed)
        if target_language.lower() == "hebrew":
            srt_content_burning = process_srt_for_rtl(srt_content)
            with open(srt_path_burning, "w", encoding="utf-8") as f:
                f.write(srt_content_burning)
            burning_source = srt_path_burning
        else:
            burning_source = srt_path

        with jobs_lock:
            jobs[job_id]["status"] = "embedding_subtitles"
            jobs[job_id]["progress"] = Progress.EMBEDDING

        # Step 4: Embed subtitles
        output_filename = f"{job_id}_subtitled.mp4"
        output_path = OUTPUT_DIR / output_filename
        embed_subtitles(
            video_path, burning_source, str(output_path), target_language, WATERMARK_CONFIG
        )

        # Calculate elapsed time
        elapsed = time.time() - start_time
        elapsed_formatted = str(timedelta(seconds=int(elapsed)))

        # Calculate cost (gpt-5-mini pricing)
        input_cost = (token_usage["prompt_tokens"] / 1_000_000) * 0.40
        output_cost = (token_usage["completion_tokens"] / 1_000_000) * 1.60
        total_cost = input_cost + output_cost
        token_usage["input_cost"] = round(input_cost, 4)
        token_usage["output_cost"] = round(output_cost, 4)
        token_usage["total_cost"] = round(total_cost, 4)

        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "progress": Progress.COMPLETED,
                "output_file": output_filename,
                "srt_content": srt_content,
                "original_srt_content": original_srt_content,
                "elapsed_time": elapsed_formatted,
                "elapsed_seconds": int(elapsed),
                "token_usage": token_usage,
                "subtitles": subtitles_data,
                "edited": False,
            })
            jobs[job_id].setdefault("files", []).append(str(output_path))

        logger.info("Job %s completed in %s", job_id, elapsed_formatted)
        logger.info(
            "Token usage: Input=%s ($%.4f) | Output=%s ($%.4f) | Total: $%.4f",
            f"{token_usage['prompt_tokens']:,}", input_cost,
            f"{token_usage['completion_tokens']:,}", output_cost, total_cost
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        # Only expose known safe error messages to the user
        if not any(keyword in error_msg.lower() for keyword in [
            "no audio", "no speech", "whisper", "too long", "corrupt",
            "unsupported", "timed out", "disk", "codec", "permission",
        ]):
            error_msg = "Video processing failed. Please try again."
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = error_msg
            jobs[job_id]["elapsed_seconds"] = int(elapsed)
        logger.error("Error processing video for job %s: %s", job_id, e)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def find_source_video_path(job_id: str) -> Optional[str]:
    """Find the original source video (upload/download/trimmed), NOT the subtitled output."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return None

    # Prefer trimmed video if available
    trimmed_path = UPLOAD_DIR / f"{job_id}_trimmed.mp4"
    if trimmed_path.exists():
        return str(trimmed_path)

    # Use the original video_path stored at upload/download time
    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        return video_path

    # Fallback: search uploads and downloads directories
    return _find_video_file_by_prefix(job_id, [UPLOAD_DIR, DOWNLOAD_DIR])


def reburn_video_task(job_id: str) -> None:
    """Background task to re-burn subtitles into video."""
    temp_dir = None
    try:
        with jobs_lock:
            job = jobs[job_id]
            job["status"] = "reburning"
            job["progress"] = Progress.REBURN_START

        # IMPORTANT: Use the original source video, NOT the subtitled output
        video_path = find_source_video_path(job_id)

        if not video_path or not Path(video_path).exists():
            raise Exception("Video file not found")

        with jobs_lock:
            job["progress"] = Progress.REBURN_SRT

        temp_dir = tempfile.mkdtemp()
        srt_path = os.path.join(temp_dir, "subtitles.srt")

        srt_content = subtitles_to_srt(job["subtitles"])

        target_language = job.get("language", "english")
        if target_language.lower() == "hebrew":
            srt_content = process_srt_for_rtl(srt_content)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        with jobs_lock:
            job["progress"] = Progress.REBURN_EMBED

        output_filename = f"{job_id}_edited.mp4"
        output_path = OUTPUT_DIR / output_filename

        embed_subtitles(
            video_path, srt_path, str(output_path), target_language, WATERMARK_CONFIG
        )

        with jobs_lock:
            job["status"] = "completed"
            job["progress"] = Progress.COMPLETED
            job["output_file"] = output_filename
            job.setdefault("files", []).append(str(output_path))

    except Exception as e:
        logger.error("Error reburning video for job %s: %s", job_id, e)
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Failed to re-burn subtitles. Please try again."
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def download_video_task(job_id: str, url: str, quality: str) -> None:
    """Background task for downloading video."""
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "downloading"
            jobs[job_id]["progress"] = Progress.DOWNLOADING

        def progress_callback(percent: float, status: str) -> None:
            with jobs_lock:
                jobs[job_id]["progress"] = min(int(percent * 0.9), 90)
                jobs[job_id]["download_status"] = status

        output_path = download_video_from_url(
            url=url,
            output_dir=DOWNLOAD_DIR,
            job_id=job_id,
            quality=quality,
            progress_callback=progress_callback,
        )

        # Probe once after download and cache results
        try:
            cached_duration = get_video_duration(str(output_path))
            cached_info = get_video_file_info(str(output_path))
        except Exception:
            cached_duration = None
            cached_info = None

        with jobs_lock:
            jobs[job_id]["status"] = "downloaded"
            jobs[job_id]["progress"] = Progress.COMPLETED
            jobs[job_id]["video_path"] = str(output_path)
            jobs[job_id]["download_status"] = "complete"
            jobs[job_id].setdefault("files", []).append(str(output_path))
            if cached_duration is not None:
                jobs[job_id]["cached_duration"] = cached_duration
            if cached_info is not None:
                jobs[job_id]["cached_video_info"] = cached_info

    except Exception as e:
        logger.error("Error downloading video for job %s: %s", job_id, e)
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Failed to download video. Please try again."


# ============================================
# Upload Endpoints
# ============================================

@app.post("/upload")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    language: str = Form(...),
    translation_service: str = Form("openai"),
    ollama_model: str = Form("llama3.2"),
):
    """Upload video and start processing."""
    check_rate_limit(request, max_requests=10)
    if not video.filename or not video.filename.lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    language = validate_language(language)
    translation_service = validate_translation_service(translation_service)
    use_ollama = translation_service.lower() == "ollama"
    if use_ollama:
        ollama_model = validate_ollama_model(ollama_model)

    enforce_capacity(require_active_slot=True)

    job_id = str(uuid.uuid4())
    safe_name = sanitize_filename(video.filename)
    video_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    validate_path_containment(video_path, UPLOAD_DIR)
    await save_upload_chunked(video, video_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "output_file": None,
            "error": None,
            "original_filename": safe_name,
            "video_path": str(video_path),
            "language": language,
            "created_at": datetime.now(),
            "files": [str(video_path)],
        }

    background_tasks.add_task(process_video, job_id, str(video_path), language, use_ollama, ollama_model)

    return {"job_id": job_id, "message": "Video uploaded successfully. Processing started."}


@app.post("/upload-only")
async def upload_video_only(request: Request, video: UploadFile = File(...)):
    """Upload video without starting processing (for wizard flow)."""
    check_rate_limit(request, max_requests=10)
    if not video.filename or not video.filename.lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    enforce_capacity(require_active_slot=False)

    job_id = str(uuid.uuid4())
    safe_name = sanitize_filename(video.filename)
    video_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    validate_path_containment(video_path, UPLOAD_DIR)
    await save_upload_chunked(video, video_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "uploaded",
            "progress": 0,
            "output_file": None,
            "error": None,
            "original_filename": safe_name,
            "video_path": str(video_path),
            "source_type": "upload",
            "created_at": datetime.now(),
            "files": [str(video_path)],
        }

    return {"job_id": job_id, "message": "Video uploaded successfully."}


# ============================================
# Processing Endpoints
# ============================================

class ProcessRequest(BaseModel):
    language: str = "English"
    translation_service: str = "openai"
    ollama_model: str = "llama3.2"


@app.post("/process/{job_id}")
async def process_existing_video(
    job_id: str,
    http_request: Request,
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    job: dict = Depends(get_validated_job),
):
    """Start translation processing on an already downloaded/uploaded video."""
    check_rate_limit(http_request, max_requests=10)
    enforce_capacity(require_active_slot=True)
    language = validate_language(request.language)
    translation_service = validate_translation_service(request.translation_service)
    use_ollama = translation_service.lower() == "ollama"
    if use_ollama:
        validate_ollama_model(request.ollama_model)

    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found. Upload or download a video first.")

    with jobs_lock:
        if job.get("status") in ACTIVE_JOB_STATUSES:
            raise HTTPException(status_code=409, detail="Job is already running")
        job["status"] = "queued"
        job["progress"] = 0
        job["output_file"] = None
        job["error"] = None
        job["language"] = language
        job["original_filename"] = job.get("original_filename", Path(video_path).name)

    background_tasks.add_task(process_video, job_id, video_path, language, use_ollama, request.ollama_model)

    return {"job_id": job_id, "message": "Processing started."}


@app.get("/status/{job_id}")
async def get_status(job_id: str, job: dict = Depends(get_validated_job)):
    """Get processing status. Returns only safe fields (no internal paths)."""
    safe_fields = {
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "language": job.get("language"),
        "output_file": Path(job["output_file"]).name if job.get("output_file") else None,
        "subtitles": job.get("subtitles"),
        "edited": job.get("edited", False),
        "srt_content": job.get("srt_content"),
        "original_srt_content": job.get("original_srt_content"),
        "token_usage": job.get("token_usage"),
        "original_filename": job.get("original_filename"),
        "elapsed_time": job.get("elapsed_time"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "video_info": job.get("video_info"),
        "source_type": job.get("source_type"),
        "download_status": job.get("download_status"),
        "trimmed": job.get("trimmed"),
        "trim_start": job.get("trim_start"),
        "trim_end": job.get("trim_end"),
    }
    return {k: v for k, v in safe_fields.items() if v is not None}


@app.get("/download/{job_id}")
async def download_video(job_id: str, job: dict = Depends(get_validated_job)):
    """Download processed video."""
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video is not ready yet")

    output_path = OUTPUT_DIR / job["output_file"]
    validate_path_containment(output_path, OUTPUT_DIR)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    safe_download_name = sanitize_filename(f"subtitled_{job['original_filename']}")
    return FileResponse(
        path=str(output_path),
        filename=safe_download_name,
        media_type="video/mp4",
    )


# ============================================
# Text Download Endpoints (StreamingResponse — no temp files)
# ============================================

@app.get("/download-srt/{job_id}")
async def download_srt(job_id: str, job: dict = Depends(get_validated_job)):
    """Download SRT subtitle file."""
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Subtitles are not ready yet")
    content = job.get("srt_content", "")
    filename = f"subtitles_{Path(job['original_filename']).stem}.srt"
    return _make_text_download_response(content, filename)


@app.get("/download-transcription/{job_id}")
async def download_transcription(job_id: str, job: dict = Depends(get_validated_job)):
    """Download original transcription SRT file (before translation)."""
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription is not ready yet")
    content = job.get("original_srt_content", "")
    if not content:
        raise HTTPException(status_code=404, detail="Original transcription not available")
    filename = f"transcription_{Path(job['original_filename']).stem}.srt"
    return _make_text_download_response(content, filename)


@app.get("/download-srt-txt/{job_id}")
async def download_srt_txt(job_id: str, job: dict = Depends(get_validated_job)):
    """Download translated subtitles as plain TXT file."""
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Subtitles are not ready yet")
    content = srt_to_plain_text(job.get("srt_content", ""))
    filename = f"subtitles_{Path(job['original_filename']).stem}.txt"
    return _make_text_download_response(content, filename)


@app.get("/download-transcription-txt/{job_id}")
async def download_transcription_txt(job_id: str, job: dict = Depends(get_validated_job)):
    """Download original transcription as plain TXT file."""
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription is not ready yet")
    content = job.get("original_srt_content", "")
    if not content:
        raise HTTPException(status_code=404, detail="Original transcription not available")
    plain = srt_to_plain_text(content)
    filename = f"transcription_{Path(job['original_filename']).stem}.txt"
    return _make_text_download_response(plain, filename)


# ============================================
# Ollama
# ============================================

@app.get("/ollama-models")
async def get_ollama_models():
    """Get available Ollama models."""
    try:
        models = ollama.list()
        return {"models": [m['name'] for m in models.get('models', [])]}
    except Exception as e:
        logger.error("Failed to list Ollama models: %s", e)
        return {"models": [], "error": "Failed to connect to Ollama"}


# ============================================
# Video Download Endpoints
# ============================================

class DownloadRequest(BaseModel):
    url: str
    quality: Literal["720p", "1080p", "best"] = "720p"


@app.get("/video-info")
async def get_video_info_endpoint(url: str = Query(..., description="YouTube or X/Twitter URL")):
    """Get video metadata without downloading."""
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="URL is required")
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL. Please provide a YouTube or X/Twitter video link.")
    try:
        info = get_video_info(url)
        return {
            "title": info.title,
            "duration": info.duration,
            "thumbnail": info.thumbnail,
            "source": info.source.value,
            "video_id": info.video_id,
        }
    except Exception as e:
        logger.error("Failed to get video info: %s", e)
        raise HTTPException(status_code=400, detail="Failed to get video info. Please check the URL.")


@app.post("/download-url")
async def download_from_url(http_request: Request, background_tasks: BackgroundTasks, request: DownloadRequest):
    """Start downloading video from URL."""
    check_rate_limit(http_request, max_requests=5)
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL. Please provide a YouTube or X/Twitter video link.")

    enforce_capacity(require_active_slot=True)

    try:
        info = get_video_info(url)
    except Exception as e:
        logger.error("Failed to get video info: %s", e)
        raise HTTPException(status_code=400, detail="Failed to get video info. Please check the URL.")

    # Reject videos longer than 1 hour
    if info.duration > 3600:
        raise HTTPException(status_code=400, detail="Video is too long. Maximum duration is 1 hour.")

    job_id = str(uuid.uuid4())

    video_info = {
        "title": info.title,
        "duration": info.duration,
        "thumbnail": info.thumbnail,
        "video_id": info.video_id,
    }

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "output_file": None,
            "error": None,
            "source_type": info.source.value,
            "source_url": url,
            "video_info": video_info,
            "video_path": None,
            "download_status": "pending",
            "created_at": datetime.now(),
            "files": [],
        }

    background_tasks.add_task(download_video_task, job_id, url, request.quality)

    return {
        "job_id": job_id,
        "message": "Download started",
        "video_info": video_info,
    }


@app.get("/download-status/{job_id}")
async def get_download_status(job_id: str, job: dict = Depends(get_validated_job)):
    """Get download progress status."""
    return {
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "download_status": job.get("download_status"),
        "error": job.get("error"),
        "video_info": job.get("video_info"),
    }


@app.get("/video-preview/{job_id}")
async def preview_video(job_id: str, _job: dict = Depends(get_validated_job)):
    """Stream video for HTML5 player preview."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


# ============================================
# Video Trimming Endpoints
# ============================================

class TrimRequest(BaseModel):
    start_time: float
    end_time: float


@app.get("/video-duration/{job_id}")
async def get_video_duration_endpoint(job_id: str, _job: dict = Depends(get_validated_job)):
    """Get video duration for trimming UI."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        duration = get_video_duration(video_path)
        file_info = get_video_file_info(video_path)

        # Cache for reuse in /trim and /skip-trim
        with jobs_lock:
            _job["cached_duration"] = duration
            _job["cached_video_info"] = file_info

        return {
            "duration": duration,
            "width": file_info.get("width"),
            "height": file_info.get("height"),
            "fps": file_info.get("fps"),
            "size_bytes": file_info.get("size_bytes"),
        }
    except Exception as e:
        logger.error("Failed to get video duration for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to get video information.")


@app.post("/trim/{job_id}")
async def trim_video_endpoint(job_id: str, request: TrimRequest, job: dict = Depends(get_validated_job)):
    """Trim video to specified time range."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    start_time = request.start_time
    end_time = request.end_time

    try:
        duration = job.get("cached_duration") or get_video_duration(video_path)
        if start_time < 0:
            start_time = 0
        if end_time > duration:
            end_time = duration
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="End time must be greater than start time")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate trim times for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to validate trim parameters.")

    trimmed_path = UPLOAD_DIR / f"{job_id}_trimmed.mp4"

    try:
        trim_video(video_path, str(trimmed_path), start_time, end_time, reencode=True)

        if not has_audio_stream(str(trimmed_path)):
            logger.warning("Trimmed file lost audio track, retrying with stream copy")
            trim_video(video_path, str(trimmed_path), start_time, end_time, reencode=False)

        with jobs_lock:
            job["original_video"] = video_path
            job["video_path"] = str(trimmed_path)
            job["trim_start"] = start_time
            job["trim_end"] = end_time
            job["trimmed"] = True
            job.setdefault("files", []).append(str(trimmed_path))

        return {
            "message": "Video trimmed successfully",
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Trim failed for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Video trimming failed. Please try again.")


@app.post("/skip-trim/{job_id}")
async def skip_trim(job_id: str, job: dict = Depends(get_validated_job)):
    """Skip trimming and use original video."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    with jobs_lock:
        job["video_path"] = video_path

    try:
        duration = job.get("cached_duration") or get_video_duration(video_path)
        with jobs_lock:
            job["trim_start"] = 0
            job["trim_end"] = duration
            job["trimmed"] = False

        return {"message": "Trimming skipped", "duration": duration}
    except Exception as e:
        logger.error("Failed to get video duration for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to process video. Please try again.")


# ============================================
# Subtitle Editing Endpoints
# ============================================

class SubtitleUpdate(BaseModel):
    id: int
    text: str


class SubtitlesUpdateRequest(BaseModel):
    subtitles: list[SubtitleUpdate]


@app.get("/subtitles/{job_id}")
async def get_subtitles(job_id: str, job: dict = Depends(get_validated_job)):
    """Get subtitle segments for editing."""
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=404, detail="No subtitles found for this job")

    return {
        "subtitles": subtitles,
        "language": job.get("language"),
        "edited": job.get("edited", False),
        "count": len(subtitles),
    }


@app.put("/subtitles/{job_id}")
async def update_subtitles(job_id: str, request: SubtitlesUpdateRequest, job: dict = Depends(get_validated_job)):
    """Update subtitle text (for editing before re-burn)."""
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=404, detail="No subtitles found for this job")

    subtitles_dict = {s["id"]: s for s in subtitles}

    updated_count = 0
    for update in request.subtitles:
        if update.id in subtitles_dict:
            subtitles_dict[update.id]["text"] = update.text
            updated_count += 1

    with jobs_lock:
        job["subtitles"] = [subtitles_dict[s["id"]] for s in subtitles]
        job["edited"] = True
        job["srt_content"] = subtitles_to_srt(job["subtitles"])

    return {
        "message": f"Updated {updated_count} subtitle(s)",
        "updated_count": updated_count,
        "total_count": len(subtitles),
    }


@app.post("/reburn/{job_id}")
async def reburn_video(job_id: str, request: Request, background_tasks: BackgroundTasks, job: dict = Depends(get_validated_job)):
    """Re-embed edited subtitles into video."""
    check_rate_limit(request, max_requests=10)
    enforce_capacity(require_active_slot=True)
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before re-burning")

    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=400, detail="No subtitles to burn")

    with jobs_lock:
        if job.get("status") in ACTIVE_JOB_STATUSES:
            raise HTTPException(status_code=409, detail="Job is already running")
        # Reserve a slot immediately to prevent duplicate reburn requests
        job["status"] = "queued"
        job["progress"] = 0

    background_tasks.add_task(reburn_video_task, job_id)

    return {"message": "Re-burning started", "job_id": job_id}


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
