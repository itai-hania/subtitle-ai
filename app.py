"""
Video Subtitle Generator API
Supports OpenAI Whisper for transcription and OpenAI/Ollama for translation
"""

import os
import uuid
import json
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional
from datetime import timedelta
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import ollama

from downloader import detect_source, get_video_info, download_video as download_video_from_url, is_valid_url, VideoSource
from trimmer import get_video_duration, trim_video, get_video_info as get_video_file_info

load_dotenv()

app = FastAPI(title="Video Subtitle Generator")

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
jobs = {}

# OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using OpenAI Whisper API"""
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript


def batch_translate_openai(texts: list, target_language: str, chunk_num: int = 0) -> tuple:
    """Translate a batch of texts using OpenAI GPT in a single API call.
    Returns (translations, token_usage) tuple."""
    if target_language.lower() == "english" or not texts:
        return texts, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Create numbered format for batch translation
    numbered_text = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])
    
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="low",
        max_completion_tokens=10000,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert Hebrew subtitle translator with deep knowledge of both English and Hebrew idioms, grammar, and natural speech patterns. Translate each numbered line to {target_language}.

CRITICAL RULES:
1. Keep the EXACT same numbering format: [1], [2], [3], etc.
2. Each translation MUST start with its number in brackets on its own line
3. Translate EVERY word - do not leave ANY English words
4. Output ONLY the numbered translations, nothing else

HEBREW TRANSLATION QUALITY RULES:
5. Use NATURAL Hebrew expressions - never translate word-by-word literally
6. Match proper Hebrew grammar: correct gender (זכר/נקבה), verb conjugation, and tense
7. Use common everyday Hebrew vocabulary that native speakers actually use
8. Preserve the original tone: casual speech stays casual, formal stays formal
9. Translate idioms to their Hebrew equivalents, not literally (e.g., "break a leg" → "בהצלחה")
10. Keep sentences concise - subtitles should be easy to read quickly
11. For technical terms with no Hebrew equivalent, transliterate appropriately

Example format:
[1] תרגום ראשון
[2] תרגום שני
[3] תרגום שלישי"""
            },
            {
                "role": "user",
                "content": numbered_text
            }
        ]
    )
    
    # Track token usage
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0
    }
    
    # Parse the numbered response back into a list
    result_text = response.choices[0].message.content
    
    # Log input and output for analysis
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"translation_log_{timestamp}_chunk{chunk_num}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== CHUNK {chunk_num} ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Target language: {target_language}\n")
        f.write(f"Number of segments: {len(texts)}\n")
        f.write(f"Input tokens: {token_usage['prompt_tokens']}\n")
        f.write(f"Output tokens: {token_usage['completion_tokens']}\n")
        f.write(f"\n{'='*50}\n")
        f.write("INPUT TEXT:\n")
        f.write(f"{'='*50}\n")
        f.write(numbered_text)
        f.write(f"\n\n{'='*50}\n")
        f.write("OUTPUT TEXT:\n")
        f.write(f"{'='*50}\n")
        f.write(result_text)
        f.write(f"\n\n{'='*50}\n")
    
    print(f"    Logged to: {log_file}")
    
    translations = []
    
    # Try to extract translations by line number with improved regex
    for i in range(len(texts)):
        # Match [N] followed by content until next [N+1] or end
        pattern = rf'\[{i+1}\]\s*(.+?)(?=\n\s*\[{i+2}\]|\n\s*\[\d+\]|$)'
        match = re.search(pattern, result_text, re.DOTALL)
        if match:
            translation = match.group(1).strip()
            # Clean up any trailing newlines or extra whitespace
            translation = ' '.join(translation.split())
            translations.append(translation)
        else:
            # Fallback: try to find by just looking for the number
            simple_pattern = rf'\[{i+1}\]\s*([^\[\n]+)'
            simple_match = re.search(simple_pattern, result_text)
            if simple_match:
                translations.append(simple_match.group(1).strip())
            else:
                # Last resort: use original text
                print(f"Warning: Could not parse translation for segment {i+1}")
                translations.append(texts[i])
    
    # Validation: ensure we have exactly the right number of translations
    if len(translations) != len(texts):
        print(f"ERROR: Translation count mismatch! Expected {len(texts)}, got {len(translations)}")
    
    # Log segment mapping for sync debugging
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\nSEGMENT MAPPING:\n")
        f.write("="*50 + "\n")
        for i, (orig, trans) in enumerate(zip(texts, translations), 1):
            f.write(f"[{i}] ORIG: {orig[:50]}...\n")
            f.write(f"[{i}] TRANS: {trans[:50]}...\n")
            f.write("-"*30 + "\n")
    
    return translations, token_usage


def batch_translate_ollama(texts: list, target_language: str, model: str = "llama3.2") -> list:
    """Translate a batch of texts using Ollama in a single API call"""
    if target_language.lower() == "english" or not texts:
        return texts
    
    # Create numbered format for batch translation
    numbered_text = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])
    
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a professional subtitle translator. Translate each numbered line to {target_language}.

CRITICAL RULES:
1. Keep the EXACT same numbering format: [1], [2], [3], etc.
2. Each translation MUST start with its number in brackets on its own line
3. Translate EVERY word - do not leave any English
4. Keep the same meaning and tone
5. Output ONLY the numbered translations, nothing else

Example format:
[1] תרגום ראשון
[2] תרגום שני
[3] תרגום שלישי"""
            },
            {
                "role": "user",
                "content": numbered_text
            }
        ]
    )
    
    # Parse the numbered response back into a list
    result_text = response['message']['content']
    translations = []
    
    # Try to extract translations by line number with improved regex
    for i in range(len(texts)):
        # Match [N] followed by content until next [N+1] or end
        pattern = rf'\[{i+1}\]\s*(.+?)(?=\n\s*\[{i+2}\]|\n\s*\[\d+\]|$)'
        match = re.search(pattern, result_text, re.DOTALL)
        if match:
            translation = match.group(1).strip()
            # Clean up any trailing newlines or extra whitespace
            translation = ' '.join(translation.split())
            translations.append(translation)
        else:
            # Fallback: try to find by just looking for the number
            simple_pattern = rf'\[{i+1}\]\s*([^\[\n]+)'
            simple_match = re.search(simple_pattern, result_text)
            if simple_match:
                translations.append(simple_match.group(1).strip())
            else:
                # Last resort: use original text
                print(f"Warning: Could not parse translation for segment {i+1}")
                translations.append(texts[i])
    
    return translations


def generate_srt(segments: list, target_language: str, use_ollama: bool = False, ollama_model: str = "llama3.2") -> tuple:
    """Generate SRT subtitle file content from transcription segments.
    Returns (translated_srt, original_srt, token_usage) tuple."""
    
    # Extract all segment data first
    segment_data = []
    for segment in segments:
        # Handle both object attributes and dictionary access
        start = getattr(segment, 'start', None) or segment.get('start', 0) if isinstance(segment, dict) else segment.start
        end = getattr(segment, 'end', None) or segment.get('end', 0) if isinstance(segment, dict) else segment.end
        text_raw = getattr(segment, 'text', None) or segment.get('text', '') if isinstance(segment, dict) else segment.text
        segment_data.append({
            'start': start,
            'end': end,
            'text': text_raw.strip()
        })
    
    # Build original SRT content (before translation)
    original_srt_content = []
    texts = [s['text'] for s in segment_data]
    for i, seg in enumerate(segment_data, 1):
        start_time = format_timestamp(seg['start'])
        end_time = format_timestamp(seg['end'])
        original_srt_content.append(f"{i}")
        original_srt_content.append(f"{start_time} --> {end_time}")
        original_srt_content.append(seg['text'])
        original_srt_content.append("")
    original_srt = "\n".join(original_srt_content)
    
    # Track total token usage
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Batch translate all texts at once if needed (in chunks for very long videos)
    if target_language.lower() != "english":
        print(f"Batch translating {len(texts)} segments to {target_language}...")
        
        # Process in chunks of 75 segments (balance between speed and reliability)
        CHUNK_SIZE = 50  # Smaller chunks to prevent LLM output token exhaustion
        translated_texts = []
        
        for chunk_start in range(0, len(texts), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(texts))
            chunk = texts[chunk_start:chunk_end]
            print(f"  Translating chunk {chunk_start + 1}-{chunk_end} of {len(texts)}...")
            
            if use_ollama:
                chunk_translations = batch_translate_ollama(chunk, target_language, ollama_model)
            else:
                chunk_num = chunk_start // CHUNK_SIZE + 1
                chunk_translations, chunk_tokens = batch_translate_openai(chunk, target_language, chunk_num)
                total_tokens["prompt_tokens"] += chunk_tokens["prompt_tokens"]
                total_tokens["completion_tokens"] += chunk_tokens["completion_tokens"]
                total_tokens["total_tokens"] += chunk_tokens["total_tokens"]
                translated_texts.extend(chunk_translations)
                continue
            
            translated_texts.extend(chunk_translations)
        
        print("Batch translation complete!")
    else:
        translated_texts = texts
    
    # Build translated SRT content and structured segment data
    srt_content = []
    subtitles_data = []
    for i, (seg, translated_text) in enumerate(zip(segment_data, translated_texts), 1):
        start_time = format_timestamp(seg['start'])
        end_time = format_timestamp(seg['end'])
        
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(translated_text)
        srt_content.append("")  # Empty line between entries
        
        # Store structured subtitle data for editing
        subtitles_data.append({
            "id": i,
            "start": seg['start'],
            "end": seg['end'],
            "text": translated_text,
            "original_text": seg['text']
        })
    
    return "\n".join(srt_content), original_srt, total_tokens, subtitles_data


def wrap_rtl(text: str) -> str:
    """Wrap text with Unicode RTL embedding markers for proper RTL rendering"""
    # RLE (Right-to-Left Embedding) at start, PDF (Pop Directional Formatting) at end
    RLE = '\u202B'
    PDF = '\u202C'
    return f"{RLE}{text}{PDF}"


def process_srt_for_rtl(srt_content: str) -> str:
    """Process SRT content to add RTL markers for Hebrew text"""
    new_blocks = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            # First two lines are ID and timing
            header = lines[:2]
            # Rest are text - wrap with RTL markers
            text_lines = lines[2:]
            processed_lines = [wrap_rtl(line) for line in text_lines]
            new_blocks.append("\n".join(header + processed_lines))
        else:
            new_blocks.append(block)
            
    return "\n\n".join(new_blocks)


def subtitles_to_srt(subtitles: list) -> str:
    """Convert subtitles list back to SRT format."""
    srt_content = []
    for sub in subtitles:
        start_time = format_timestamp(sub['start'])
        end_time = format_timestamp(sub['end'])
        srt_content.append(f"{sub['id']}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(sub['text'])
        srt_content.append("")
    return "\n".join(srt_content)


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using FFmpeg with compression for Whisper API"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "mp3",
        "-ar", "16000",      # 16kHz sample rate (Whisper's native rate)
        "-ac", "1",          # Mono audio
        "-b:a", "64k",       # 64kbps bitrate (speech-optimized, keeps file under 25MB)
        "-y", audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def embed_subtitles(video_path: str, srt_path: str, output_path: str, target_language: str):
    """Embed subtitles and watermark into video using FFmpeg"""
    style = "FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Bold=1"
    
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")
    
    wm = WATERMARK_CONFIG
    logo_path = wm["logo_path"]
    
    if logo_path.exists():
        logo_escaped = str(logo_path.absolute()).replace("\\", "/").replace(":", "\\:")
        opacity = wm["opacity"]
        logo_h = wm["logo_height"]
        margin = wm["margin"]
        eng_size = wm["english_font_size"]
        heb_size = wm["hebrew_font_size"]
        spacing = wm["text_spacing"]
        eng_text = wm["english_text"]
        heb_text = wm["hebrew_text"][::-1]  # Reverse for FFmpeg RTL fix
        
        hebrew_font = "/Library/Fonts/Arial Unicode.ttf"
        hebrew_font_escaped = hebrew_font.replace(":", "\\\\:")
        
        filter_complex = (
            f"[0:v]subtitles='{srt_escaped}':force_style='{style}'[sub];"
            f"[1:v]scale=-1:{logo_h},format=rgba,colorchannelmixer=aa={opacity}[logo];"
            f"[sub]drawtext=text='{eng_text}':"
            f"fontsize={eng_size}:fontcolor=white@{opacity}:"
            f"x={logo_h}+{margin*2}:y={margin}:"
            f"shadowcolor=black:shadowx=1:shadowy=1[text1];"
            f"[text1]drawtext=text='{heb_text}':"
            f"fontfile='{hebrew_font_escaped}':"
            f"fontsize={heb_size}:fontcolor=white@{opacity}:"
            f"x={logo_h}+{margin*2}:y={margin + spacing}:"
            f"shadowcolor=black:shadowx=1:shadowy=1[text2];"
            f"[text2][logo]overlay={margin}:{margin}[out]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", str(logo_path.absolute()),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path
        ]
    else:
        print(f"Warning: Logo not found at {logo_path}, watermark disabled")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-vsync", "passthrough",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path
        ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'No error message'}")
        raise


def process_video(job_id: str, video_path: str, target_language: str, use_ollama: bool, ollama_model: str):
    """Main processing pipeline"""
    start_time = time.time()
    
    try:
        jobs[job_id]["status"] = "extracting_audio"
        jobs[job_id]["progress"] = 10
        
        # Create temp directory for this job
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.mp3")
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        srt_path_burning = os.path.join(temp_dir, "subtitles_burning.srt")
        
        # Step 1: Extract audio
        extract_audio(video_path, audio_path)
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["progress"] = 30
        
        # Step 2: Transcribe with Whisper
        transcript = transcribe_audio(audio_path)
        jobs[job_id]["status"] = "translating"
        jobs[job_id]["progress"] = 50
        
        # Step 3: Generate SRT (with translation if needed)
        segments = transcript.segments if hasattr(transcript, 'segments') else transcript.get('segments', [])
        srt_content, original_srt_content, token_usage, subtitles_data = generate_srt(segments, target_language, use_ollama, ollama_model)
        
        # Save standard SRT for download
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
            
        # Prepare SRT for burning (Apply RTL markers if needed)
        if target_language.lower() == "hebrew":
            srt_content_burning = process_srt_for_rtl(srt_content)
            with open(srt_path_burning, "w", encoding="utf-8") as f:
                f.write(srt_content_burning)
            burning_source = srt_path_burning
        else:
            burning_source = srt_path
        
        jobs[job_id]["status"] = "embedding_subtitles"
        jobs[job_id]["progress"] = 70
        
        # Step 4: Embed subtitles
        output_filename = f"{job_id}_subtitled.mp4"
        output_path = OUTPUT_DIR / output_filename
        embed_subtitles(video_path, burning_source, str(output_path), target_language)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["output_file"] = output_filename
        jobs[job_id]["srt_content"] = srt_content
        jobs[job_id]["original_srt_content"] = original_srt_content
        jobs[job_id]["elapsed_time"] = elapsed_formatted
        jobs[job_id]["elapsed_seconds"] = int(elapsed_time)
        jobs[job_id]["token_usage"] = token_usage
        jobs[job_id]["subtitles"] = subtitles_data
        jobs[job_id]["edited"] = False
        
        print(f"Job {job_id} completed in {elapsed_formatted}")
        
        # Calculate cost based on gpt-5-nano pricing
        # Input: $0.05 per 1M tokens, Output: $0.40 per 1M tokens
        input_cost = (token_usage["prompt_tokens"] / 1_000_000) * 0.05
        output_cost = (token_usage["completion_tokens"] / 1_000_000) * 0.40
        total_cost = input_cost + output_cost
        
        token_usage["input_cost"] = round(input_cost, 4)
        token_usage["output_cost"] = round(output_cost, 4)
        token_usage["total_cost"] = round(total_cost, 4)
        
        print(f"Token usage: Input={token_usage['prompt_tokens']:,} (${input_cost:.4f}) | Output={token_usage['completion_tokens']:,} (${output_cost:.4f}) | Total cost: ${total_cost:.4f}")
        
        # Cleanup temp files
        os.remove(audio_path)
        os.remove(srt_path)
        if os.path.exists(srt_path_burning):
            os.remove(srt_path_burning)
        os.rmdir(temp_dir)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["elapsed_seconds"] = int(elapsed_time)
        print(f"Error processing video: {e}")


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    language: str = Form(...),
    translation_service: str = Form("openai"),
    ollama_model: str = Form("llama3.2")
):
    """Upload video and start processing"""
    
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_file": None,
        "error": None,
        "original_filename": video.filename,
        "language": language
    }
    
    # Start processing in background
    use_ollama = translation_service.lower() == "ollama"
    background_tasks.add_task(
        process_video, 
        job_id, 
        str(video_path), 
        language, 
        use_ollama, 
        ollama_model
    )
    
    return {"job_id": job_id, "message": "Video uploaded successfully. Processing started."}


@app.post("/upload-only")
async def upload_video_only(
    video: UploadFile = File(...)
):
    """Upload video without starting processing (for wizard flow)."""
    
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
    job_id = str(uuid.uuid4())[:8]
    
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)
    
    jobs[job_id] = {
        "status": "uploaded",
        "progress": 0,
        "output_file": None,
        "error": None,
        "original_filename": video.filename,
        "video_path": str(video_path),
        "source_type": "upload"
    }
    
    return {"job_id": job_id, "message": "Video uploaded successfully."}


class ProcessRequest(BaseModel):
    language: str = "English"
    translation_service: str = "openai"
    ollama_model: str = "llama3.2"


@app.post("/process/{job_id}")
async def process_existing_video(
    job_id: str,
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """Start translation processing on an already downloaded/uploaded video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    video_path = job.get("video_path")
    
    # Also check for uploaded files
    if not video_path:
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            for file in UPLOAD_DIR.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() == f'.{ext}':
                    video_path = str(file)
                    break
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found. Upload or download a video first.")
    
    # Update job for processing
    job["status"] = "queued"
    job["progress"] = 0
    job["output_file"] = None
    job["error"] = None
    job["language"] = request.language
    job["original_filename"] = job.get("original_filename", Path(video_path).name)
    
    # Start processing in background
    use_ollama = request.translation_service.lower() == "ollama"
    background_tasks.add_task(
        process_video,
        job_id,
        video_path,
        request.language,
        use_ollama,
        request.ollama_model
    )
    
    return {"job_id": job_id, "message": "Processing started."}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download processed video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video is not ready yet")
    
    output_path = OUTPUT_DIR / job["output_file"]
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=str(output_path),
        filename=f"subtitled_{job['original_filename']}",
        media_type="video/mp4"
    )


@app.get("/download-srt/{job_id}")
async def download_srt(job_id: str):
    """Download SRT subtitle file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Subtitles are not ready yet")
    
    srt_content = job.get("srt_content", "")
    
    # Create temp SRT file
    srt_filename = f"{job_id}_subtitles.srt"
    srt_path = OUTPUT_DIR / srt_filename
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    return FileResponse(
        path=str(srt_path),
        filename=f"subtitles_{job['original_filename'].replace('.mp4', '.srt')}",
        media_type="text/plain"
    )


@app.get("/download-transcription/{job_id}")
async def download_transcription(job_id: str):
    """Download original transcription SRT file (before translation)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription is not ready yet")
    
    original_content = job.get("original_srt_content", "")
    if not original_content:
        raise HTTPException(status_code=404, detail="Original transcription not available")
    
    # Create temp SRT file
    srt_filename = f"{job_id}_transcription.srt"
    srt_path = OUTPUT_DIR / srt_filename
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(original_content)
    
    return FileResponse(
        path=str(srt_path),
        filename=f"transcription_{job['original_filename'].replace('.mp4', '.srt')}",
        media_type="text/plain"
    )


def srt_to_plain_text(srt_content: str) -> str:
    """Convert SRT content to plain text (removes timestamps and numbering)"""
    lines = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        block_lines = block.split('\n')
        if len(block_lines) >= 3:
            # Skip first two lines (number and timestamp), take the rest
            text_lines = block_lines[2:]
            lines.extend(text_lines)
    
    return '\n'.join(lines)


@app.get("/download-srt-txt/{job_id}")
async def download_srt_txt(job_id: str):
    """Download translated subtitles as plain TXT file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Subtitles are not ready yet")
    
    srt_content = job.get("srt_content", "")
    plain_text = srt_to_plain_text(srt_content)
    
    # Create temp TXT file
    txt_filename = f"{job_id}_subtitles.txt"
    txt_path = OUTPUT_DIR / txt_filename
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(plain_text)
    
    return FileResponse(
        path=str(txt_path),
        filename=f"subtitles_{job['original_filename'].replace('.mp4', '.txt')}",
        media_type="text/plain"
    )


@app.get("/download-transcription-txt/{job_id}")
async def download_transcription_txt(job_id: str):
    """Download original transcription as plain TXT file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription is not ready yet")
    
    original_content = job.get("original_srt_content", "")
    if not original_content:
        raise HTTPException(status_code=404, detail="Original transcription not available")
    
    plain_text = srt_to_plain_text(original_content)
    
    # Create temp TXT file
    txt_filename = f"{job_id}_transcription.txt"
    txt_path = OUTPUT_DIR / txt_filename
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(plain_text)
    
    return FileResponse(
        path=str(txt_path),
        filename=f"transcription_{job['original_filename'].replace('.mp4', '.txt')}",
        media_type="text/plain"
    )


@app.get("/ollama-models")
async def get_ollama_models():
    """Get available Ollama models"""
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
        raise HTTPException(
            status_code=400, 
            detail="Invalid URL. Please provide a YouTube or X/Twitter video link."
        )
    
    try:
        info = get_video_info(url)
        return {
            "title": info.title,
            "duration": info.duration,
            "thumbnail": info.thumbnail,
            "source": info.source.value,
            "video_id": info.video_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def download_video_task(job_id: str, url: str, quality: str):
    """Background task for downloading video."""
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["progress"] = 5
        
        def progress_callback(percent, status):
            # Scale download progress to 0-90%
            jobs[job_id]["progress"] = min(int(percent * 0.9), 90)
            jobs[job_id]["download_status"] = status
        
        output_path = download_video_from_url(
            url=url,
            output_dir=DOWNLOAD_DIR,
            job_id=job_id,
            quality=quality,
            progress_callback=progress_callback
        )
        
        jobs[job_id]["status"] = "downloaded"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["video_path"] = str(output_path)
        jobs[job_id]["download_status"] = "complete"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/download-url")
async def download_from_url(
    background_tasks: BackgroundTasks,
    request: DownloadRequest
):
    """Start downloading video from URL."""
    url = request.url.strip()
    quality = request.quality
    
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    if not is_valid_url(url):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Please provide a YouTube or X/Twitter video link."
        )
    
    # Validate quality
    if quality not in ["720p", "1080p", "best"]:
        quality = "720p"
    
    # Get video info first
    try:
        info = get_video_info(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job status with enhanced fields
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
            "video_id": info.video_id
        },
        "video_path": None,
        "download_status": "pending"
    }
    
    # Start download in background
    background_tasks.add_task(download_video_task, job_id, url, quality)
    
    return {
        "job_id": job_id,
        "message": "Download started",
        "video_info": jobs[job_id]["video_info"]
    }


@app.get("/download-status/{job_id}")
async def get_download_status(job_id: str):
    """Get download progress status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "download_status": job.get("download_status"),
        "error": job.get("error"),
        "video_info": job.get("video_info"),
        "video_path": job.get("video_path")
    }


@app.get("/video-preview/{job_id}")
async def preview_video(job_id: str):
    """Stream video for HTML5 player preview."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    video_path = job.get("video_path")
    
    if not video_path:
        # Check if this is an upload job
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            for file in UPLOAD_DIR.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() == f'.{ext}':
                    video_path = str(file)
                    break
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }
    )


# ============================================
# Video Trimming Endpoints
# ============================================

class TrimRequest(BaseModel):
    start_time: float
    end_time: float


@app.get("/video-duration/{job_id}")
async def get_video_duration_endpoint(job_id: str):
    """Get video duration for trimming UI."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    video_path = job.get("video_path")
    
    # Also check for uploaded files
    if not video_path:
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            for file in UPLOAD_DIR.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() == f'.{ext}':
                    video_path = str(file)
                    break
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        duration = get_video_duration(video_path)
        file_info = get_video_file_info(video_path)
        return {
            "duration": duration,
            "width": file_info.get("width"),
            "height": file_info.get("height"),
            "fps": file_info.get("fps"),
            "size_bytes": file_info.get("size_bytes")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trim/{job_id}")
async def trim_video_endpoint(job_id: str, request: TrimRequest):
    """Trim video to specified time range."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    video_path = job.get("video_path")
    
    # Also check for uploaded files
    if not video_path:
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            for file in UPLOAD_DIR.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() == f'.{ext}':
                    video_path = str(file)
                    break
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    start_time = request.start_time
    end_time = request.end_time
    
    # Validate times
    try:
        duration = get_video_duration(video_path)
        if start_time < 0:
            start_time = 0
        if end_time > duration:
            end_time = duration
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="End time must be greater than start time")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Create trimmed file path
    trimmed_path = UPLOAD_DIR / f"{job_id}_trimmed.mp4"
    
    try:
        trim_video(video_path, str(trimmed_path), start_time, end_time)
        
        # Update job with trim info
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
            "duration": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skip-trim/{job_id}")
async def skip_trim(job_id: str):
    """Skip trimming and use original video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    video_path = job.get("video_path")
    
    if not video_path:
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            for file in UPLOAD_DIR.iterdir():
                if file.name.startswith(job_id) and file.suffix.lower() == f'.{ext}':
                    video_path = str(file)
                    job["video_path"] = video_path
                    break
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        duration = get_video_duration(video_path)
        job["trim_start"] = 0
        job["trim_end"] = duration
        job["trimmed"] = False
        
        return {
            "message": "Trimming skipped",
            "video_path": video_path,
            "duration": duration
        }
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
async def get_subtitles(job_id: str):
    """Get subtitle segments for editing."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    
    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=404, detail="No subtitles found for this job")
    
    return {
        "subtitles": subtitles,
        "language": job.get("language"),
        "edited": job.get("edited", False),
        "count": len(subtitles)
    }


@app.put("/subtitles/{job_id}")
async def update_subtitles(job_id: str, request: SubtitlesUpdateRequest):
    """Update subtitle text (for editing before re-burn)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    
    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=404, detail="No subtitles found for this job")
    
    # Create a lookup for quick updates
    subtitles_dict = {s["id"]: s for s in subtitles}
    
    updated_count = 0
    for update in request.subtitles:
        if update.id in subtitles_dict:
            subtitles_dict[update.id]["text"] = update.text
            updated_count += 1
    
    # Rebuild subtitles list maintaining order
    job["subtitles"] = [subtitles_dict[s["id"]] for s in subtitles]
    job["edited"] = True
    
    # Rebuild SRT content from updated subtitles
    job["srt_content"] = subtitles_to_srt(job["subtitles"])
    
    return {
        "message": f"Updated {updated_count} subtitle(s)",
        "updated_count": updated_count,
        "total_count": len(subtitles)
    }


def reburn_video_task(job_id: str):
    """Background task to re-burn subtitles into video."""
    try:
        job = jobs[job_id]
        job["status"] = "reburing"
        job["progress"] = 10
        
        # Get video path (use original if trimmed, otherwise uploaded)
        video_path = job.get("original_video") or job.get("video_path")
        
        # If no original_video, find uploaded file
        if not video_path:
            for ext in ['mp4', 'mov', 'avi', 'mkv']:
                for file in UPLOAD_DIR.iterdir():
                    if file.name.startswith(job_id) and not "_trimmed" in file.name and file.suffix.lower() == f'.{ext}':
                        video_path = str(file)
                        break
        
        # Use trimmed if available
        trimmed_path = UPLOAD_DIR / f"{job_id}_trimmed.mp4"
        if trimmed_path.exists():
            video_path = str(trimmed_path)
        
        if not video_path or not Path(video_path).exists():
            raise Exception("Video file not found")
        
        job["progress"] = 30
        
        # Create temp SRT file from updated subtitles
        temp_dir = tempfile.mkdtemp()
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        
        srt_content = subtitles_to_srt(job["subtitles"])
        
        # Apply RTL markers if Hebrew
        target_language = job.get("language", "english")
        if target_language.lower() == "hebrew":
            srt_content = process_srt_for_rtl(srt_content)
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        job["progress"] = 50
        
        # Re-embed subtitles
        output_filename = f"{job_id}_edited.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        embed_subtitles(video_path, srt_path, str(output_path), target_language)
        
        job["progress"] = 90
        
        # Update job
        job["status"] = "completed"
        job["progress"] = 100
        job["output_file"] = output_filename
        
        # Cleanup temp files
        os.remove(srt_path)
        os.rmdir(temp_dir)
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/reburn/{job_id}")
async def reburn_video(job_id: str, background_tasks: BackgroundTasks):
    """Re-embed edited subtitles into video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.get("status") not in ["completed"]:
        raise HTTPException(status_code=400, detail="Job must be completed before re-burning")
    
    subtitles = job.get("subtitles", [])
    if not subtitles:
        raise HTTPException(status_code=400, detail="No subtitles to burn")
    
    # Start re-burn in background
    background_tasks.add_task(reburn_video_task, job_id)
    
    return {
        "message": "Re-burning started",
        "job_id": job_id
    }


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
