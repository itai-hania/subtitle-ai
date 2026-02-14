"""Shared test fixtures for SubtitleAI tests."""

import pytest
from fastapi.testclient import TestClient

from app import app, jobs, jobs_lock, rate_limiter


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear the jobs dict and rate limiter before each test."""
    with jobs_lock:
        jobs.clear()
    with rate_limiter._lock:
        rate_limiter._requests.clear()
    yield
    with jobs_lock:
        jobs.clear()


@pytest.fixture
def sample_job():
    """Insert a completed sample job into the jobs dict and return (job_id, job)."""
    job_id = "550e8400-e29b-41d4-a716-446655440000"
    job = {
        "status": "completed",
        "progress": 100,
        "output_file": "550e8400-e29b-41d4-a716-446655440000_subtitled.mp4",
        "error": None,
        "original_filename": "demo.mp4",
        "video_path": "/tmp/fake_video.mp4",
        "language": "Hebrew",
        "srt_content": (
            "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nGoodbye world\n"
        ),
        "original_srt_content": (
            "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nGoodbye world\n"
        ),
        "subtitles": [
            {"id": 1, "start": 1.0, "end": 3.0, "text": "שלום עולם", "original_text": "Hello world"},
            {"id": 2, "start": 4.0, "end": 6.0, "text": "להתראות עולם", "original_text": "Goodbye world"},
        ],
        "edited": False,
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    jobs[job_id] = job
    return job_id, job
