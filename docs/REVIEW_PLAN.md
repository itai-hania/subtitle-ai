# SubtitleAI Code Review — Implementation Plan

**Date:** 2026-02-08
**Status:** Approved, implementing

---

## Architecture

| # | Decision | Description | Files | Effort |
|---|----------|-------------|-------|--------|
| 1A | Minimal split | Extract processing pipeline to `processing.py` | `app.py`, new `processing.py` | 30 min |
| 2A | Fix critical security | Restrict CORS, add 500MB upload limit, streaming upload | `app.py` | 1h |
| 3C | Do nothing | Keep in-memory job store (sufficient for single-user) | — | — |
| 4A | FastAPI dependency | Create `get_validated_job()` for consistent job_id validation | `app.py` | 20 min |

## Code Quality

| # | Decision | Description | Files | Effort |
|---|----------|-------------|-------|--------|
| 5A | Extract shared helper | `parse_numbered_translations()` used by OpenAI + Ollama | `processing.py` | 15 min |
| 6B | Update CLAUDE.md | Code uses gpt-5-mini intentionally; update docs to match | `CLAUDE.md` | 5 min |
| 7A | StreamingResponse | Replace 4 download endpoints with helper + return content from memory | `app.py` | 30 min |
| 8A | Replace print with logging | Module-level logger, proper levels (DEBUG/INFO/WARNING/ERROR) | All Python files | 30 min |

## Tests

| # | Decision | Description | Files | Effort |
|---|----------|-------------|-------|--------|
| 12A | Test infrastructure | `tests/` dir, `conftest.py`, pytest config | new `tests/`, `pyproject.toml` | 30 min |
| 9A | Unit test pure functions | format_timestamp, validate_job_id, parse_ffmpeg_error, RTL, SRT | `tests/test_utils.py` | 1h |
| 10A | Translation parsing tests | Comprehensive edge cases: brackets, multiline, skipped numbers | `tests/test_translation.py` | 45 min |
| 11B | API validation tests | Error paths: invalid inputs, missing jobs, wrong states | `tests/test_api.py` | 1h |

## Performance

| # | Decision | Description | Files | Effort |
|---|----------|-------------|-------|--------|
| 13A | Chunked upload | Read/write in 1MB chunks instead of loading entire file | `app.py` | 10 min |
| 14A | FFmpeg timeouts | 10-30 min timeouts on all subprocess.run calls | `processing.py`, `trimmer.py` | 15 min |
| 15A | Store video_path always | Ensure all codepaths set job["video_path"], reduce filesystem scans | `app.py` | 20 min |
| 16B | Whisper size validation | Check audio file size after extraction, clear error if >25MB | `processing.py` | 10 min |

---

## Implementation Order

### Phase 1: Architecture Refactoring
1. **4A** — FastAPI dependency `get_validated_job()` (foundation for all route changes)
2. **1A + 5A** — Extract `processing.py` with shared parsing helper
3. **8A** — Replace all `print()` with `logging` (during extraction)

### Phase 2: Security & Performance
4. **2A + 13A** — Fix CORS + upload size limit + chunked streaming write
5. **14A + 16B** — FFmpeg timeouts + Whisper audio size validation
6. **15A** — Always store `video_path` in job dict

### Phase 3: Code Quality
7. **7A** — StreamingResponse for download endpoints (eliminate temp files)
8. **6B** — Update CLAUDE.md

### Phase 4: Tests
9. **12A** — Test infrastructure setup
10. **9A** — Unit tests for pure functions
11. **10A** — Translation parsing tests
12. **11B** — API validation/error path tests

**Total estimated effort: ~6-7 hours**
**Test after each phase to verify no regressions.**
