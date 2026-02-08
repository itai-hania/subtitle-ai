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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ollama
import openai
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# Valid job ID pattern (alphanumeric and hyphens only)
JOB_ID_PATTERN = re.compile(r'^[a-zA-Z0-9-]{1,36}$')

load_dotenv()

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


app = FastAPI(title="Video Subtitle Generator", lifespan=lifespan)

# CORS middleware — restrict to same-origin (no wildcard with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ============================================
# Helpers
# ============================================

def find_video_path(job_id: str) -> Optional[str]:
    """Find video file path for a job. Prefers output (subtitled) video if available."""
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
    for search_dir in [UPLOAD_DIR, DOWNLOAD_DIR]:
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            for file in search_dir.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() == f'.{ext}':
                    logger.debug("Found via filesystem search: %s", file)
                    return str(file)

    logger.debug("No video found for job %s", job_id)
    return None


def cleanup_old_jobs() -> None:
    """Remove jobs older than JOB_TTL and their associated files."""
    now = datetime.now()
    with jobs_lock:
        expired = [
            jid for jid, job in jobs.items()
            if now - job.get("created_at", now) > JOB_TTL
        ]
    for jid in expired:
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
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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

        logger.info("Job %s completed in %s", job_id, elapsed_formatted)
        logger.info(
            "Token usage: Input=%s ($%.4f) | Output=%s ($%.4f) | Total: $%.4f",
            f"{token_usage['prompt_tokens']:,}", input_cost,
            f"{token_usage['completion_tokens']:,}", output_cost, total_cost
        )

    except Exception as e:
        elapsed = time.time() - start_time
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["elapsed_seconds"] = int(elapsed)
        logger.error("Error processing video: %s", e)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def reburn_video_task(job_id: str) -> None:
    """Background task to re-burn subtitles into video."""
    temp_dir = None
    try:
        with jobs_lock:
            job = jobs[job_id]
            job["status"] = "reburning"
            job["progress"] = Progress.REBURN_START

        # Use trimmed video if available, otherwise find the original
        trimmed_path = UPLOAD_DIR / f"{job_id}_trimmed.mp4"
        if trimmed_path.exists():
            video_path = str(trimmed_path)
        else:
            video_path = find_video_path(job_id)

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

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
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

        with jobs_lock:
            jobs[job_id]["status"] = "downloaded"
            jobs[job_id]["progress"] = Progress.COMPLETED
            jobs[job_id]["video_path"] = str(output_path)
            jobs[job_id]["download_status"] = "complete"

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


# ============================================
# Upload Endpoints
# ============================================

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    language: str = Form(...),
    translation_service: str = Form("openai"),
    ollama_model: str = Form("llama3.2"),
):
    """Upload video and start processing."""
    if not video.filename or not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    await save_upload_chunked(video, video_path)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_file": None,
        "error": None,
        "original_filename": video.filename,
        "video_path": str(video_path),
        "language": language,
        "created_at": datetime.now(),
    }

    use_ollama = translation_service.lower() == "ollama"
    background_tasks.add_task(process_video, job_id, str(video_path), language, use_ollama, ollama_model)

    return {"job_id": job_id, "message": "Video uploaded successfully. Processing started."}


@app.post("/upload-only")
async def upload_video_only(video: UploadFile = File(...)):
    """Upload video without starting processing (for wizard flow)."""
    if not video.filename or not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    await save_upload_chunked(video, video_path)

    jobs[job_id] = {
        "status": "uploaded",
        "progress": 0,
        "output_file": None,
        "error": None,
        "original_filename": video.filename,
        "video_path": str(video_path),
        "source_type": "upload",
        "created_at": datetime.now(),
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
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    job: dict = Depends(get_validated_job),
):
    """Start translation processing on an already downloaded/uploaded video."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found. Upload or download a video first.")

    job["status"] = "queued"
    job["progress"] = 0
    job["output_file"] = None
    job["error"] = None
    job["language"] = request.language
    job["original_filename"] = job.get("original_filename", Path(video_path).name)

    use_ollama = request.translation_service.lower() == "ollama"
    background_tasks.add_task(process_video, job_id, video_path, request.language, use_ollama, request.ollama_model)

    return {"job_id": job_id, "message": "Processing started."}


@app.get("/status/{job_id}")
async def get_status(job_id: str, job: dict = Depends(get_validated_job)):
    """Get processing status."""
    return job


@app.get("/download/{job_id}")
async def download_video(job_id: str, job: dict = Depends(get_validated_job)):
    """Download processed video."""
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video is not ready yet")

    output_path = OUTPUT_DIR / job["output_file"]
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=str(output_path),
        filename=f"subtitled_{job['original_filename']}",
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
        return {"models": [], "error": str(e)}


# ============================================
# Video Download Endpoints
# ============================================

class DownloadRequest(BaseModel):
    url: str
    quality: str = "720p"


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
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/download-url")
async def download_from_url(background_tasks: BackgroundTasks, request: DownloadRequest):
    """Start downloading video from URL."""
    url = request.url.strip()
    quality = request.quality

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL. Please provide a YouTube or X/Twitter video link.")
    if quality not in ["720p", "1080p", "best"]:
        quality = "720p"

    try:
        info = get_video_info(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_file": None,
        "error": None,
        "source_type": info.source.value,
        "source_url": url,
        "video_info": {
            "title": info.title,
            "duration": info.duration,
            "thumbnail": info.thumbnail,
            "video_id": info.video_id,
        },
        "video_path": None,
        "download_status": "pending",
        "created_at": datetime.now(),
    }

    background_tasks.add_task(download_video_task, job_id, url, quality)

    return {
        "job_id": job_id,
        "message": "Download started",
        "video_info": jobs[job_id]["video_info"],
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
        "video_path": job.get("video_path"),
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
        return {
            "duration": duration,
            "width": file_info.get("width"),
            "height": file_info.get("height"),
            "fps": file_info.get("fps"),
            "size_bytes": file_info.get("size_bytes"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trim/{job_id}")
async def trim_video_endpoint(job_id: str, request: TrimRequest, job: dict = Depends(get_validated_job)):
    """Trim video to specified time range."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    start_time = request.start_time
    end_time = request.end_time

    try:
        duration = get_video_duration(video_path)
        if start_time < 0:
            start_time = 0
        if end_time > duration:
            end_time = duration
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="End time must be greater than start time")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    trimmed_path = UPLOAD_DIR / f"{job_id}_trimmed.mp4"

    try:
        trim_video(video_path, str(trimmed_path), start_time, end_time, reencode=True)

        if not has_audio_stream(str(trimmed_path)):
            logger.warning("Trimmed file lost audio track, retrying")
            trim_video(video_path, str(trimmed_path), start_time, end_time, reencode=True)

        job["original_video"] = video_path
        job["video_path"] = str(trimmed_path)
        job["trim_start"] = start_time
        job["trim_end"] = end_time
        job["trimmed"] = True

        return {
            "message": "Video trimmed successfully",
            "trimmed_path": str(trimmed_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skip-trim/{job_id}")
async def skip_trim(job_id: str, job: dict = Depends(get_validated_job)):
    """Skip trimming and use original video."""
    video_path = find_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    job["video_path"] = video_path

    try:
        duration = get_video_duration(video_path)
        job["trim_start"] = 0
        job["trim_end"] = duration
        job["trimmed"] = False

        return {"message": "Trimming skipped", "video_path": video_path, "duration": duration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    job["subtitles"] = [subtitles_dict[s["id"]] for s in subtitles]
    job["edited"] = True
    job["srt_content"] = subtitles_to_srt(job["subtitles"])

    return {
        "message": f"Updated {updated_count} subtitle(s)",
        "updated_count": updated_count,
        "total_count": len(subtitles),
    }


@app.post("/reburn/{job_id}")
async def reburn_video(job_id: str, background_tasks: BackgroundTasks, job: dict = Depends(get_validated_job)):
    """Re-embed edited subtitles into video."""
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before re-burning")

    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=400, detail="No subtitles to burn")

    background_tasks.add_task(reburn_video_task, job_id)

    return {"message": "Re-burning started", "job_id": job_id}


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
