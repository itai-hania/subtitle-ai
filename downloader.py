"""
Video Downloader Module
Handles downloading videos from YouTube and X/Twitter using yt-dlp
"""

import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import yt_dlp


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
    """Detect if URL is from YouTube, X/Twitter, or unknown source."""
    url_lower = url.lower().strip()
    
    # YouTube patterns
    youtube_patterns = [
        r'(youtube\.com/watch\?v=)',
        r'(youtu\.be/)',
        r'(youtube\.com/shorts/)',
        r'(youtube\.com/embed/)',
    ]
    for pattern in youtube_patterns:
        if re.search(pattern, url_lower):
            return VideoSource.YOUTUBE
    
    # X/Twitter patterns
    twitter_patterns = [
        r'(twitter\.com/.+/status/)',
        r'(x\.com/.+/status/)',
        r'(mobile\.twitter\.com/.+/status/)',
    ]
    for pattern in twitter_patterns:
        if re.search(pattern, url_lower):
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
        raise ValueError(f"Unsupported URL. Please use YouTube or X/Twitter links.")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    # Add Twitter auth if available
    if source == VideoSource.TWITTER:
        username = os.getenv('X_USERNAME')
        password = os.getenv('X_PASSWORD')
        if username and password:
            ydl_opts['username'] = username
            ydl_opts['password'] = password
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            return VideoInfo(
                title=info.get('title', 'Unknown Title'),
                duration=float(info.get('duration', 0)),
                thumbnail=info.get('thumbnail'),
                source=source,
                url=url,
                video_id=info.get('id', 'unknown')
            )
    except Exception as e:
        raise Exception(f"Failed to get video info: {str(e)}")


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
        raise ValueError(f"Unsupported URL. Please use YouTube or X/Twitter links.")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Build format string based on quality
    if quality == "720p":
        format_str = 'bestvideo[height<=720]+bestaudio/best[height<=720]/best'
    elif quality == "1080p":
        format_str = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best'
    else:  # "best"
        format_str = 'bestvideo+bestaudio/best'
    
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
    
    ydl_opts = {
        'format': format_str,
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress_hook],
    }
    
    # Add Twitter auth if available
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
        raise Exception(f"Failed to download video: {str(e)}")


def is_valid_url(url: str) -> bool:
    """Check if URL is a valid YouTube or X/Twitter video URL."""
    return detect_source(url) != VideoSource.UNKNOWN
