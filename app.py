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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
import openai
import ollama

load_dotenv()

app = FastAPI(title="Video Subtitle Generator")

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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
        model="gpt-5-nano",
        reasoning_effort="low",
        max_completion_tokens=10000,
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
    
    # Build translated SRT content
    srt_content = []
    for i, (seg, translated_text) in enumerate(zip(segment_data, translated_texts), 1):
        start_time = format_timestamp(seg['start'])
        end_time = format_timestamp(seg['end'])
        
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(translated_text)
        srt_content.append("")  # Empty line between entries
    
    return "\n".join(srt_content), original_srt, total_tokens


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
    """Embed subtitles into video using FFmpeg"""
    # Standard font settings
    style = "FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Bold=1"
    
    # Escape special characters in path for FFmpeg filter
    # FFmpeg requires escaping: \ : ' for the subtitles filter
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
        # Video settings: Preserve original timestamps and framerate
        "-c:v", "libx264", 
        "-preset", "fast",
        "-vsync", "passthrough",  # Preserve original timestamps exactly
        # Audio settings: Copy audio to maintain perfect sync
        "-c:a", "copy",
        # Output settings
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
        srt_content, original_srt_content, token_usage = generate_srt(segments, target_language, use_ollama, ollama_model)
        
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


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
