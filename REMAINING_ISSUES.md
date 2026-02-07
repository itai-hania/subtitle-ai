# Remaining Code Issues

Issues identified during code review that were not fixed yet.

---

## 1. Race Condition in Job State Management

**Severity:** Medium (low traffic app)
**Location:** `app.py` — `jobs` dict modified from background tasks without synchronization

**Problem:** Multiple concurrent requests can cause inconsistent state reads when background tasks (`process_video`, `download_video_task`, `reburn_video_task`) modify `jobs[job_id]` while `/status/{job_id}` reads it.

**Fix:** Use `threading.Lock` per job or switch to a thread-safe store.

```python
import threading
job_locks = {}

def get_job_lock(job_id):
    if job_id not in job_locks:
        job_locks[job_id] = threading.Lock()
    return job_locks[job_id]
```

---

## 2. Unbounded Jobs Dictionary (Memory Leak)

**Severity:** Medium
**Location:** `app.py:54` — `jobs = {}`

**Problem:** Jobs are never cleaned up. Each job stores video paths, subtitles, SRT content. After thousands of jobs, server runs out of memory.

**Fix:** Implement job expiration with periodic cleanup.

```python
from datetime import datetime, timedelta

JOB_TTL = timedelta(hours=24)

def cleanup_old_jobs():
    now = datetime.now()
    expired = [jid for jid, job in jobs.items()
               if now - job.get("created_at", now) > JOB_TTL]
    for jid in expired:
        del jobs[jid]
```

---

## 3. No Audio Stream Validation

**Severity:** Medium
**Location:** `app.py` — `extract_audio()` and `process_video()`

**Problem:** If a video has no audio track (screen recordings, animations), FFmpeg fails with a non-descriptive error and the job gets stuck.

**Fix:** Check for audio streams before extraction.

```python
def has_audio_stream(video_path: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=codec_type",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() == "audio"
```

---

## 4. Missing CORS Headers

**Severity:** Low (only matters if frontend moves to separate server)
**Location:** `app.py`

**Problem:** No CORS middleware configured. If the frontend ever runs on a different port/domain (e.g., dev server), video streaming and API calls will fail.

**Fix:**

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

---

## 5. Magic Numbers for Progress Percentages

**Severity:** Low (readability)
**Location:** `app.py` — hardcoded progress values (10, 30, 50, 70, 100)

**Fix:** Use named constants.

```python
class Progress:
    EXTRACTING_AUDIO = 10
    TRANSCRIBING = 30
    TRANSLATING = 50
    EMBEDDING = 70
    COMPLETED = 100
```

---

## 6. FFmpeg Error Messages Not User-Friendly

**Severity:** Low
**Location:** `app.py` — FFmpeg errors surface as raw stderr

**Fix:** Parse common FFmpeg errors into user-friendly messages (e.g., "disk full", "codec not supported", "corrupt file").

---

## 7. Mousedown Event Listener Stacking

**Severity:** Low
**Location:** `static/script.js:519-537` — trim handle drag listeners

**Problem:** Rapid clicks on trim handles before mouseup can stack `mousemove` listeners. The `{ once: true }` on mouseup prevents most issues, but edge cases exist.

**Fix:** Guard against adding duplicate listeners by removing old ones before adding new.

---

## 8. Dead Code: updateSteps Function

**Severity:** Low (cosmetic)
**Location:** `static/script.js` — `updateSteps()` references DOM IDs (`step-upload`, `step-extract`, etc.) that exist in HTML but the visual feedback doesn't work because these IDs conflict with wizard step naming.

**Fix:** Either remove the function or fix the DOM element IDs to make the processing sub-steps visually update during translation.
