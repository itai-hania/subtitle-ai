"""API validation and error path tests (Issue 11B)."""

import pytest
from datetime import datetime

from app import jobs, jobs_lock

# Test UUIDs for consistent use across tests
PROC_UUID = "11111111-1111-1111-1111-111111111111"
NOTR_UUID = "22222222-2222-2222-2222-222222222222"
WORK_UUID = "33333333-3333-3333-3333-333333333333"
NOSUB_UUID = "44444444-4444-4444-4444-444444444444"


# ============================================
# Job validation (get_validated_job dependency)
# ============================================

class TestJobValidation:
    def test_nonexistent_job_returns_404(self, client):
        response = client.get("/status/550e8400-e29b-41d4-a716-446655440099")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_invalid_job_id_returns_404(self, client):
        response = client.get("/status/../../etc/passwd")
        assert response.status_code == 404

    def test_empty_job_id_pattern(self, client):
        # FastAPI may not route this, but test the path
        response = client.get("/status/;rm -rf /")
        assert response.status_code == 404

    def test_valid_job_returns_data(self, client, sample_job):
        job_id, _ = sample_job
        response = client.get(f"/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


# ============================================
# Upload validation
# ============================================

class TestUploadValidation:
    def test_invalid_file_type(self, client):
        import io
        file = io.BytesIO(b"not a video")
        response = client.post(
            "/upload-only",
            files={"video": ("test.txt", file, "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_invalid_file_type_exe(self, client):
        import io
        file = io.BytesIO(b"MZ\x90\x00")
        response = client.post(
            "/upload-only",
            files={"video": ("malware.exe", file, "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_valid_mp4_extension_accepted(self, client):
        import io
        # Small fake MP4 — upload should succeed even with no real video data
        file = io.BytesIO(b"\x00" * 100)
        response = client.post(
            "/upload-only",
            files={"video": ("test.mp4", file, "video/mp4")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


# ============================================
# Download endpoints — wrong state
# ============================================

class TestDownloadValidation:
    def test_download_video_not_completed(self, client):
        # Create a job that's still processing
        jobs[PROC_UUID] = {
            "status": "transcribing",
            "progress": 30,
            "output_file": None,
            "error": None,
            "original_filename": "test.mp4",
        }
        response = client.get(f"/download/{PROC_UUID}")
        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()

    def test_download_srt_not_completed(self, client):
        jobs[PROC_UUID] = {
            "status": "embedding_subtitles",
            "progress": 70,
            "output_file": None,
            "error": None,
            "original_filename": "test.mp4",
        }
        response = client.get(f"/download-srt/{PROC_UUID}")
        assert response.status_code == 400

    def test_download_srt_completed(self, client, sample_job):
        job_id, _ = sample_job
        response = client.get(f"/download-srt/{job_id}")
        assert response.status_code == 200
        assert "Hello world" in response.text

    def test_download_srt_txt_completed(self, client, sample_job):
        job_id, _ = sample_job
        response = client.get(f"/download-srt-txt/{job_id}")
        assert response.status_code == 200
        # Should be plain text without timestamps
        assert "-->" not in response.text
        assert "Hello world" in response.text

    def test_download_transcription_no_content(self, client):
        jobs[NOTR_UUID] = {
            "status": "completed",
            "progress": 100,
            "output_file": "test.mp4",
            "error": None,
            "original_filename": "test.mp4",
            "original_srt_content": "",
        }
        response = client.get(f"/download-transcription/{NOTR_UUID}")
        assert response.status_code == 404
        assert "not available" in response.json()["detail"].lower()


# ============================================
# Subtitle endpoints — validation
# ============================================

class TestSubtitleValidation:
    def test_get_subtitles_not_completed(self, client):
        jobs[WORK_UUID] = {
            "status": "translating",
            "progress": 50,
        }
        response = client.get(f"/subtitles/{WORK_UUID}")
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

    def test_get_subtitles_completed(self, client, sample_job):
        job_id, _ = sample_job
        response = client.get(f"/subtitles/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["language"] == "Hebrew"

    def test_update_subtitles_not_completed(self, client):
        jobs[WORK_UUID] = {
            "status": "translating",
            "progress": 50,
        }
        response = client.put(
            f"/subtitles/{WORK_UUID}",
            json={"subtitles": [{"id": 1, "text": "test"}]},
        )
        assert response.status_code == 400

    def test_update_subtitles_success(self, client, sample_job):
        job_id, job = sample_job
        response = client.put(
            f"/subtitles/{job_id}",
            json={"subtitles": [{"id": 1, "text": "Updated text"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated_count"] == 1
        # Verify the update was applied
        assert job["subtitles"][0]["text"] == "Updated text"
        assert job["edited"] is True


# ============================================
# Reburn validation
# ============================================

class TestReburnValidation:
    def test_reburn_not_completed(self, client):
        jobs[WORK_UUID] = {
            "status": "translating",
            "progress": 50,
        }
        response = client.post(f"/reburn/{WORK_UUID}")
        assert response.status_code == 400

    def test_reburn_no_subtitles(self, client):
        jobs[NOSUB_UUID] = {
            "status": "completed",
            "progress": 100,
            "subtitles": [],
        }
        response = client.post(f"/reburn/{NOSUB_UUID}")
        assert response.status_code == 400
        assert "No subtitles" in response.json()["detail"]


# ============================================
# Video info / URL validation
# ============================================

class TestUrlValidation:
    def test_video_info_empty_url(self, client):
        response = client.get("/video-info?url=")
        assert response.status_code == 400

    def test_video_info_invalid_url(self, client):
        response = client.get("/video-info?url=https://example.com/notavideo")
        assert response.status_code == 400
        assert "Invalid URL" in response.json()["detail"]

    def test_download_url_invalid(self, client):
        response = client.post(
            "/download-url",
            json={"url": "https://example.com/notavideo"},
        )
        assert response.status_code == 400

    def test_download_url_empty(self, client):
        response = client.post(
            "/download-url",
            json={"url": ""},
        )
        assert response.status_code == 400


# ============================================
# Trim validation
# ============================================

class TestTrimValidation:
    def test_trim_nonexistent_job(self, client):
        response = client.post(
            "/trim/550e8400-e29b-41d4-a716-446655449999",
            json={"start_time": 0, "end_time": 10},
        )
        assert response.status_code == 404

    def test_skip_trim_nonexistent_job(self, client):
        response = client.post("/skip-trim/550e8400-e29b-41d4-a716-446655449999")
        assert response.status_code == 404

    def test_video_duration_nonexistent_job(self, client):
        response = client.get("/video-duration/550e8400-e29b-41d4-a716-446655449999")
        assert response.status_code == 404
