# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SubtitleAI is a web-based video subtitle generator. Users upload videos or paste YouTube/X URLs, the app transcribes audio with OpenAI Whisper, translates with GPT-5-nano, and burns subtitles + a branded watermark into the video using FFmpeg.

## Running the App

```bash
# Setup (one-time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python run.py
# OR directly:
source venv/bin/activate && python app.py
```

Server starts at `http://127.0.0.1:8000`. Requires Python 3.10+, FFmpeg in PATH, and `OPENAI_API_KEY` in `.env`.

There are no automated tests.

## Architecture

**Single-page app** with a FastAPI backend (`app.py`) and vanilla JS frontend (`static/`).

### Backend (Python)

- **`app.py`** — The entire backend: FastAPI routes (~20 endpoints), job management, video processing pipeline, translation service, subtitle generation, watermark embedding. All state is in-memory (`jobs` dict protected by `threading.Lock`).
- **`downloader.py`** — yt-dlp wrapper for downloading from YouTube and X/Twitter.
- **`trimmer.py`** — FFmpeg wrapper for video trimming with re-encode support.

### Frontend (Vanilla JS)

- **`static/script.js`** — State-driven SPA with states: `source → downloading → processing → editor → export`. Manages video player, subtitle overlay, trim controls, and polling.
- **`static/styles.css`** — Dark-themed CSS.
- **`static/index.html`** — Single HTML page.

### Processing Pipeline

```
Video Input → Extract Audio (FFmpeg, 16kHz mono MP3)
  → Whisper API transcription → GPT-5-nano translation (batched, 75 segments/chunk)
  → Generate SRT → FFmpeg embed subtitles + watermark → Output MP4
```

### File Storage

- `uploads/` — User-uploaded videos
- `downloads/` — yt-dlp downloaded videos
- `outputs/` — Processed videos with subtitles
- `logs/` — Translation debug logs

All stored by job ID (UUID). Jobs auto-expire after 24 hours.

## Key Conventions

- **Translation model**: Always use `gpt-5-nano` with `reasoning_effort="low"` and `max_completion_tokens=2000`. Do not change the model unless explicitly asked.
- **GPT-5-nano pricing**: Input $0.05/1M tokens, Output $0.40/1M tokens.
- **RTL handling**: Hebrew text uses bidi markers (`\u202B`/`\u202C`) for subtitles and reversed character order for FFmpeg drawtext.
- **Watermark**: Logo + bilingual text (@FinancialEduX / המחנך הפיננסי) at 70% opacity, upper-right corner. Config is `WATERMARK_CONFIG` dict in `app.py`.
- **Job IDs**: UUID format, validated with `JOB_ID_PATTERN` regex for path traversal protection.
- **FFmpeg patterns**: Use `-ss` before `-i` for fast seeking. Subtitle embedding uses `filter_complex` chaining: subtitles → drawtext → drawtext → overlay.
- **Font resolution**: Cross-platform Hebrew font lookup in `find_hebrew_font()` — checks macOS, Linux, and Windows paths.

## Development Rules

### Plan Before You Code

- Before implementing any feature or fix, think through the approach first. Identify which files are affected, what the edge cases are, and how it integrates with existing code.
- For non-trivial changes, outline the plan before writing code.

### Keep It Simple

- Write the simplest code that solves the problem. Avoid abstractions until they're needed more than once.
- Prefer flat over nested. Prefer explicit over clever.
- Small functions that do one thing. If a function needs a comment explaining what it does, it should be split or renamed.
- No dead code. Delete unused imports, variables, and functions — don't comment them out.

### Python Best Practices

- Use type hints on all function signatures (parameters and return types).
- Use `pathlib.Path` instead of `os.path` for file operations.
- Use f-strings for string formatting.
- Prefer `dataclass` or `pydantic.BaseModel` over raw dicts for structured data.
- Handle errors explicitly — no bare `except:`. Catch specific exceptions.
- Use `logging` module instead of `print()` for debug output.

### Type Checking & Linting

- Run `mypy --strict` on all Python files before considering a change complete.
- Run `ruff check` for linting and `ruff format` for formatting.
- Fix all type errors and lint warnings — do not suppress with `# type: ignore` unless there is a documented reason.

### Testing

- Write tests for all new functions and bug fixes. Use `pytest`.
- Test files go in `tests/` directory, named `test_<module>.py`.
- Run tests with: `pytest tests/ -v`
- Aim for tests that are fast, isolated, and deterministic. Mock external services (OpenAI API, FFmpeg subprocess calls, yt-dlp).
