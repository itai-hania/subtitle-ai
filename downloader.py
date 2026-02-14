"""
Video Downloader Module
Handles downloading videos from YouTube and X/Twitter using yt-dlp
"""

import logging
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

import yt_dlp

logger = logging.getLogger(__name__)

# Allowed URL schemes
ALLOWED_SCHEMES = {"http", "https"}

# Allowed hostnames for video downloads
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
TWITTER_HOSTS = {"twitter.com", "www.twitter.com", "mobile.twitter.com", "x.com", "www.x.com"}

# Download limits
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_VIDEO_DURATION = 3600  # 1 hour in seconds
SOCKET_TIMEOUT = 30  # seconds


class VideoSource(Enum):
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    UNKNOWN = "unknown"


@dataclass
class VideoInfo:
    """Video metadata from URL"""
    title: str
    duration: float  # seconds
    thumbnail: Optional[str]
    source: VideoSource
    url: str
    video_id: str


def detect_source(url: str) -> VideoSource:
    """Detect if URL is from YouTube, X/Twitter, or unknown source.

    Uses urllib.parse for safe hostname extraction instead of regex.
    Only allows http/https schemes to prevent SSRF via file://, ftp://, etc.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return VideoSource.UNKNOWN

    # Only allow http and https schemes
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return VideoSource.UNKNOWN

    hostname = (parsed.hostname or "").lower()

    if hostname in YOUTUBE_HOSTS:
        return VideoSource.YOUTUBE

    if hostname in TWITTER_HOSTS:
        return VideoSource.TWITTER

    return VideoSource.UNKNOWN


def get_video_info(url: str) -> VideoInfo:
    """
    Get video metadata without downloading.
    
    Args:
        url: Video URL (YouTube or X/Twitter)
        
    Returns:
        VideoInfo dataclass with title, duration, thumbnail, etc.
        
    Raises:
        ValueError: If URL is invalid or unsupported
        Exception: If yt-dlp fails to extract info
    """
    source = detect_source(url)
    if source == VideoSource.UNKNOWN:
        raise ValueError("Unsupported URL. Please use YouTube or X/Twitter links.")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'socket_timeout': SOCKET_TIMEOUT,
    }

    if source == VideoSource.TWITTER:
        username = os.getenv('X_USERNAME')
        password = os.getenv('X_PASSWORD')
        if username and password:
            ydl_opts['username'] = username
            ydl_opts['password'] = password

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            duration = float(info.get('duration', 0))
            if duration > MAX_VIDEO_DURATION:
                raise ValueError(
                    f"Video is too long ({duration / 60:.0f} min). "
                    f"Maximum duration is {MAX_VIDEO_DURATION // 60} minutes."
                )

            return VideoInfo(
                title=info.get('title', 'Unknown Title'),
                duration=duration,
                thumbnail=info.get('thumbnail'),
                source=source,
                url=url,
                video_id=info.get('id', 'unknown')
            )
    except ValueError:
        raise
    except Exception as e:
        logger.error("Failed to get video info: %s", e)
        raise Exception("Failed to get video info. Please check the URL and try again.")


def download_video(
    url: str,
    output_dir: Path,
    job_id: str,
    quality: str = "720p",
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Download video from URL.
    
    Args:
        url: Video URL (YouTube or X/Twitter)
        output_dir: Directory to save the downloaded video
        job_id: Unique job identifier for filename
        quality: Video quality - "720p", "1080p", "best"
        progress_callback: Optional callback function(percent, status)
        
    Returns:
        Path to the downloaded video file
        
    Raises:
        ValueError: If URL is unsupported
        Exception: If download fails
    """
    source = detect_source(url)
    if source == VideoSource.UNKNOWN:
        raise ValueError("Unsupported URL. Please use YouTube or X/Twitter links.")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_template = str(output_dir / f"{job_id}_downloaded.%(ext)s")

    def progress_hook(d):
        if progress_callback and d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                percent = int((downloaded / total) * 100)
                progress_callback(percent, 'downloading')
        elif progress_callback and d['status'] == 'finished':
            progress_callback(100, 'processing')

    if quality == "720p":
        format_str = 'bestvideo[height<=720]+bestaudio/best[height<=720]/best'
    elif quality == "1080p":
        format_str = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best'
    else:
        format_str = 'bestvideo+bestaudio/best'

    ydl_opts = {
        'format': format_str,
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress_hook],
        'socket_timeout': SOCKET_TIMEOUT,
        'max_filesize': MAX_DOWNLOAD_BYTES,
    }

    if source == VideoSource.TWITTER:
        username = os.getenv('X_USERNAME')
        password = os.getenv('X_PASSWORD')
        if username and password:
            ydl_opts['username'] = username
            ydl_opts['password'] = password

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded file (extension might vary)
        for ext in ['mp4', 'webm', 'mkv', 'mov']:
            output_path = output_dir / f"{job_id}_downloaded.{ext}"
            if output_path.exists():
                return output_path

        # Fallback: look for any file starting with job_id
        for file in output_dir.iterdir():
            if file.name.startswith(f"{job_id}_downloaded"):
                return file

        raise Exception("Download completed but output file not found")

    except Exception as e:
        logger.error("Failed to download video: %s", e)
        raise Exception("Failed to download video. Please check the URL and try again.")


def is_valid_url(url: str) -> bool:
    """Check if URL is a valid YouTube or X/Twitter video URL."""
    return detect_source(url) != VideoSource.UNKNOWN
