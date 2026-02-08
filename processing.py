"""
Video Processing Pipeline
Handles transcription, translation, SRT generation, subtitle embedding, and watermark overlay.
"""

import logging
import os
import re
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import openai
import ollama

logger = logging.getLogger(__name__)

# FFmpeg operation timeouts (seconds)
FFMPEG_EXTRACT_TIMEOUT = 600   # 10 min for audio extraction
FFMPEG_EMBED_TIMEOUT = 1800    # 30 min for subtitle embedding
FFMPEG_PROBE_TIMEOUT = 10      # 10 sec for ffprobe

# Whisper API file size limit
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB


def has_audio_stream(video_path: str) -> bool:
    """Check if video file has an audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_PROBE_TIMEOUT
        )
        return result.stdout.strip() == "audio"
    except (subprocess.TimeoutExpired, Exception):
        return False


def parse_ffmpeg_error(stderr: str) -> str:
    """Convert FFmpeg errors to user-friendly messages."""
    stderr_lower = stderr.lower() if stderr else ""
    if "no space left on device" in stderr_lower:
        return "Server disk is full. Please try again later."
    if "invalid argument" in stderr_lower or "invalid data" in stderr_lower:
        return "Video file is corrupt or uses an unsupported format."
    if ("codec not currently supported" in stderr_lower
            or ("decoder" in stderr_lower and "not found" in stderr_lower)):
        return "Video codec is not supported. Try converting to MP4 first."
    if "permission denied" in stderr_lower:
        return "Permission denied when accessing video file."
    if "no such file" in stderr_lower:
        return "Video file not found."
    if "does not contain any stream" in stderr_lower:
        return "Video file has no valid streams."
    return f"Video processing failed: {stderr[:200] if stderr else 'Unknown error'}"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = round((secs % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{milliseconds:03d}"


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video using FFmpeg with compression for Whisper API."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "mp3",
        "-ar", "16000",      # 16kHz sample rate (Whisper's native rate)
        "-ac", "1",          # Mono audio
        "-b:a", "64k",       # 64kbps bitrate (speech-optimized)
        "-y", audio_path
    ]
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=FFMPEG_EXTRACT_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise Exception("Audio extraction timed out. The video may be too long or corrupt.")
    except subprocess.CalledProcessError as e:
        raise Exception(parse_ffmpeg_error(e.stderr.decode() if e.stderr else ""))


def validate_audio_size(audio_path: str) -> None:
    """Check that extracted audio is within Whisper API limits."""
    size = Path(audio_path).stat().st_size
    if size == 0:
        raise Exception("Audio extraction produced empty file. Video may have incompatible codec.")
    if size > WHISPER_MAX_BYTES:
        size_mb = size / (1024 * 1024)
        raise Exception(
            f"Audio file is {size_mb:.1f}MB, exceeding Whisper's 25MB limit. "
            "Try trimming the video to under ~50 minutes."
        )


def transcribe_audio(client: openai.OpenAI, audio_path: str) -> dict:
    """Transcribe audio using OpenAI Whisper API."""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript


# ============================================
# Translation
# ============================================

def parse_numbered_translations(result_text: str, original_texts: list[str]) -> list[str]:
    """Parse numbered translations from LLM output.

    Expects format like:
        [1] First translation
        [2] Second translation

    Falls back to original text for segments that can't be parsed.
    """
    translations: list[str] = []

    for i in range(len(original_texts)):
        # Match [N] followed by content until next [N+1] or end
        pattern = rf'\[{i+1}\]\s*(.+?)(?=\n\s*\[{i+2}\]|\n\s*\[\d+\]|\n\n(?!\s*\[)|\Z)'
        match = re.search(pattern, result_text, re.DOTALL)
        if match:
            translation = match.group(1).strip()
            translation = ' '.join(translation.split())
            translations.append(translation)
        else:
            # Fallback: try to find by just looking for the number
            simple_pattern = rf'\[{i+1}\]\s*([^\[\n]+)'
            simple_match = re.search(simple_pattern, result_text)
            if simple_match:
                translations.append(simple_match.group(1).strip())
            else:
                logger.warning("Could not parse translation for segment %d", i + 1)
                translations.append(original_texts[i])

    if len(translations) != len(original_texts):
        logger.error(
            "Translation count mismatch! Expected %d, got %d",
            len(original_texts), len(translations)
        )

    return translations


def batch_translate_openai(
    client: openai.OpenAI,
    texts: list[str],
    target_language: str,
    chunk_num: int = 0,
) -> tuple[list[str], dict]:
    """Translate a batch of texts using OpenAI GPT in a single API call.
    Returns (translations, token_usage) tuple."""
    if target_language.lower() == "english" or not texts:
        return texts, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    numbered_text = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])

    response = client.chat.completions.create(
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
6. Match proper Hebrew grammar: correct gender, verb conjugation, and tense
7. Use common everyday Hebrew vocabulary that native speakers actually use
8. Preserve the original tone: casual speech stays casual, formal stays formal
9. Translate idioms to their Hebrew equivalents, not literally
10. Keep sentences concise - subtitles should be easy to read quickly
11. For technical terms with no Hebrew equivalent, transliterate appropriately

Example format:
[1] First translation
[2] Second translation
[3] Third translation"""
            },
            {
                "role": "user",
                "content": numbered_text
            }
        ]
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }

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
        f.write(f"\n{'='*50}\nINPUT TEXT:\n{'='*50}\n")
        f.write(numbered_text)
        f.write(f"\n\n{'='*50}\nOUTPUT TEXT:\n{'='*50}\n")
        f.write(result_text)
        f.write(f"\n\n{'='*50}\n")

    logger.info("    Logged to: %s", log_file)

    translations = parse_numbered_translations(result_text, texts)

    # Log segment mapping
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\nSEGMENT MAPPING:\n" + "=" * 50 + "\n")
        for i, (orig, trans) in enumerate(zip(texts, translations), 1):
            f.write(f"[{i}] ORIG: {orig[:50]}...\n")
            f.write(f"[{i}] TRANS: {trans[:50]}...\n")
            f.write("-" * 30 + "\n")

    return translations, token_usage


def batch_translate_ollama(
    texts: list[str],
    target_language: str,
    model: str = "llama3.2",
) -> list[str]:
    """Translate a batch of texts using Ollama in a single API call."""
    if target_language.lower() == "english" or not texts:
        return texts

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
[1] First translation
[2] Second translation
[3] Third translation"""
            },
            {
                "role": "user",
                "content": numbered_text
            }
        ]
    )

    result_text = response['message']['content']
    return parse_numbered_translations(result_text, texts)


# ============================================
# SRT Generation
# ============================================

def generate_srt(
    segments: list,
    target_language: str,
    client: openai.OpenAI,
    use_ollama: bool = False,
    ollama_model: str = "llama3.2",
) -> tuple[str, str, dict, list[dict]]:
    """Generate SRT subtitle file content from transcription segments.
    Returns (translated_srt, original_srt, token_usage, subtitles_data) tuple."""

    # Extract all segment data first
    segment_data: list[dict] = []
    for segment in segments:
        if isinstance(segment, dict):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text_raw = segment.get('text', '')
        else:
            start = segment.start
            end = segment.end
            text_raw = segment.text
        segment_data.append({
            'start': start,
            'end': end,
            'text': text_raw.strip()
        })

    # Build original SRT content (before translation)
    original_srt_content: list[str] = []
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

    # Batch translate if needed
    if target_language.lower() != "english":
        logger.info("Batch translating %d segments to %s...", len(texts), target_language)
        CHUNK_SIZE = 50
        translated_texts: list[str] = []

        for chunk_start in range(0, len(texts), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(texts))
            chunk = texts[chunk_start:chunk_end]
            logger.info(
                "  Translating chunk %d-%d of %d...",
                chunk_start + 1, chunk_end, len(texts)
            )

            if use_ollama:
                chunk_translations = batch_translate_ollama(chunk, target_language, ollama_model)
                translated_texts.extend(chunk_translations)
            else:
                chunk_num = chunk_start // CHUNK_SIZE + 1
                chunk_translations, chunk_tokens = batch_translate_openai(
                    client, chunk, target_language, chunk_num
                )
                total_tokens["prompt_tokens"] += chunk_tokens["prompt_tokens"]
                total_tokens["completion_tokens"] += chunk_tokens["completion_tokens"]
                total_tokens["total_tokens"] += chunk_tokens["total_tokens"]
                translated_texts.extend(chunk_translations)

        logger.info("Batch translation complete!")
    else:
        translated_texts = texts

    # Build translated SRT content and structured segment data
    srt_content: list[str] = []
    subtitles_data: list[dict] = []
    for i, (seg, translated_text) in enumerate(zip(segment_data, translated_texts), 1):
        start_time = format_timestamp(seg['start'])
        end_time = format_timestamp(seg['end'])

        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(translated_text)
        srt_content.append("")

        subtitles_data.append({
            "id": i,
            "start": seg['start'],
            "end": seg['end'],
            "text": translated_text,
            "original_text": seg['text']
        })

    return "\n".join(srt_content), original_srt, total_tokens, subtitles_data


# ============================================
# RTL Support
# ============================================

def wrap_rtl(text: str) -> str:
    """Wrap text with Unicode RTL embedding markers for proper RTL rendering."""
    RLE = '\u202B'
    PDF = '\u202C'
    return f"{RLE}{text}{PDF}"


def process_srt_for_rtl(srt_content: str) -> str:
    """Process SRT content to add RTL markers for Hebrew text."""
    new_blocks: list[str] = []
    blocks = srt_content.strip().split('\n\n')

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            header = lines[:2]
            text_lines = lines[2:]
            processed_lines = [wrap_rtl(line) for line in text_lines]
            new_blocks.append("\n".join(header + processed_lines))
        else:
            new_blocks.append(block)

    return "\n\n".join(new_blocks)


def subtitles_to_srt(subtitles: list[dict]) -> str:
    """Convert subtitles list back to SRT format."""
    srt_content: list[str] = []
    for sub in subtitles:
        start_time = format_timestamp(sub['start'])
        end_time = format_timestamp(sub['end'])
        srt_content.append(f"{sub['id']}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(sub['text'])
        srt_content.append("")
    return "\n".join(srt_content)


def srt_to_plain_text(srt_content: str) -> str:
    """Convert SRT content to plain text (removes timestamps and numbering)."""
    lines: list[str] = []
    blocks = srt_content.strip().split('\n\n')
    for block in blocks:
        block_lines = block.split('\n')
        if len(block_lines) >= 3:
            text_lines = block_lines[2:]
            lines.extend(text_lines)
    return '\n'.join(lines)


# ============================================
# FFmpeg Subtitle Embedding
# ============================================

def get_hebrew_font_path() -> str:
    """Get the path to a Hebrew-compatible font based on the current OS."""
    system = platform.system()
    if system == "Darwin":
        candidates = [
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    elif system == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "arialuni.ttf"),
            os.path.join(windir, "Fonts", "arial.ttf"),
            os.path.join(windir, "Fonts", "segoeui.ttf"),
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    logger.warning("No Hebrew-compatible font found for %s, tried: %s", system, candidates)
    return candidates[0]


def embed_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    target_language: str,
    watermark_config: dict,
) -> None:
    """Embed subtitles and watermark into video using FFmpeg."""
    hebrew_font = get_hebrew_font_path()
    fonts_dir = os.path.dirname(hebrew_font)
    fonts_dir_escaped = fonts_dir.replace("\\", "/").replace(":", "\\:")
    style = (
        "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H40000000,BackColour=&H80000000,"
        "Outline=1,Shadow=2,Bold=0,MarginV=30"
    )

    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")

    wm = watermark_config
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

        hebrew_font_escaped = hebrew_font.replace("\\", "/").replace(":", "\\\\:")

        filter_complex = (
            f"[0:v]subtitles='{srt_escaped}':fontsdir='{fonts_dir_escaped}'"
            f":force_style='{style}'[sub];"
            f"[1:v]scale=-1:{logo_h},format=rgba,"
            f"colorchannelmixer=aa={opacity}[logo];"
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
        logger.warning("Logo not found at %s, watermark disabled", logo_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", (
                f"subtitles='{srt_escaped}':fontsdir='{fonts_dir_escaped}'"
                f":force_style='{style}'"
            ),
            "-c:v", "libx264",
            "-preset", "fast",
            "-vsync", "passthrough",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path
        ]

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=FFMPEG_EMBED_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise Exception(
            "Subtitle embedding timed out. The video may be too long or complex."
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else 'No error message'
        logger.error("FFmpeg error: %s", error_msg)
        raise Exception(parse_ffmpeg_error(error_msg))
