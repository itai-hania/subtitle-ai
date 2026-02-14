"""
Video Trimmer Module
Handles video trimming using FFmpeg
"""

import logging
import os
import subprocess
from pathlib import Path
logger = logging.getLogger(__name__)

# Allowed directories for output files
ALLOWED_OUTPUT_DIRS = {Path("uploads"), Path("outputs"), Path("downloads")}

# Timeout for trimming operations (seconds)
TRIM_TIMEOUT = 600  # 10 minutes


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds
        
    Raises:
        Exception: If FFmpeg fails or file not found
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        duration = float(result.stdout.strip())
        return duration
    except subprocess.TimeoutExpired:
        raise Exception("Timed out getting video duration")
    except subprocess.CalledProcessError as e:
        logger.error("FFprobe failed for %s: %s", video_path, e.stderr)
        raise Exception("Failed to get video duration.")
    except ValueError:
        raise Exception("Failed to parse video duration")


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for FFmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def trim_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    reencode: bool = False
) -> Path:
    """
    Trim video to specified time range.
    
    Args:
        input_path: Path to source video
        output_path: Path for trimmed output
        start_time: Start time in seconds
        end_time: End time in seconds
        reencode: If True, re-encode the video (slower but more accurate).
                  If False, use stream copy (fast but may have slight inaccuracies).
    
    Returns:
        Path to the trimmed video
        
    Raises:
        ValueError: If time range is invalid
        Exception: If FFmpeg fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Validate output path containment
    resolved_output = output_path.resolve()
    allowed = False
    for allowed_dir in ALLOWED_OUTPUT_DIRS:
        allowed_resolved = allowed_dir.resolve()
        if str(resolved_output).startswith(str(allowed_resolved) + os.sep):
            allowed = True
            break
    if not allowed:
        raise ValueError("Invalid output path")

    # Validate time range
    if start_time < 0:
        start_time = 0

    if end_time <= start_time:
        raise ValueError("End time must be greater than start time")

    # Get duration to validate end_time
    duration = get_video_duration(str(input_path))
    if end_time > duration:
        end_time = duration

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate duration for FFmpeg
    trim_duration = end_time - start_time
    
    if reencode:
        # Re-encode for accuracy — -ss before -i for fast seeking
        cmd = [
            "ffmpeg", "-y",
            "-ss", format_time(start_time),
            "-i", str(input_path),
            "-t", format_time(trim_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path)
        ]
    else:
        # Stream copy for speed (may have slight keyframe inaccuracies)
        # Using -ss before -i for faster seeking
        cmd = [
            "ffmpeg", "-y",
            "-ss", format_time(start_time),
            "-i", str(input_path),
            "-t", format_time(trim_duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=TRIM_TIMEOUT)
        return output_path
    except subprocess.TimeoutExpired:
        raise Exception("Video trimming timed out. The video may be too long or complex.")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown error"
        logger.error("FFmpeg trim failed: %s", error_msg[:500])
        raise Exception("Failed to trim video. Please try again.")


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata including duration, resolution, and codec info.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video metadata
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,codec_name,r_frame_rate",
        "-show_entries", "format=duration,size",
        "-of", "json",
        str(video_path)
    ]
    
    try:
        import json
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        
        stream = data.get("streams", [{}])[0]
        format_info = data.get("format", {})
        
        # Parse frame rate (usually comes as "30/1" or similar)
        fps_str = stream.get("r_frame_rate", "0/1")
        try:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den != 0 else 0
        except (ValueError, ZeroDivisionError):
            fps = 0
        
        return {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "duration": float(format_info.get("duration", 0)),
            "size_bytes": int(format_info.get("size", 0)),
            "codec": stream.get("codec_name"),
            "fps": round(fps, 2)
        }
    except subprocess.CalledProcessError as e:
        logger.error("FFprobe info failed: %s", e.stderr)
        raise Exception("Failed to get video info.")
    except Exception as e:
        logger.error("Failed to parse video info: %s", e)
        raise Exception("Failed to get video info.")
