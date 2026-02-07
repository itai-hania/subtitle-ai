# SubtitleAI v2 — Unified Studio Redesign Plan

## Problem Statement

The current 5-step wizard (Source → Trim → Translate → Edit → Export) has critical bugs and a fragmented UX. User testing revealed:

1. **YouTube/X downloads fail**: "No speech detected" error after downloading — file path or codec issue with downloaded videos (AV1+Opus). Works fine with local uploads.
2. **Editor is terrible**: Stacked vertical layout, tiny subtitle list, poor video-subtitle sync
3. **UX feels disjointed**: 5 separate wizard pages feels like 5 different apps

## Proposed Approach

**Complete frontend redesign** to a unified single-page studio layout (inspired by TranslateMom Studio), keeping the dark theme:

```
┌─────────────────────────────────────────────────────┐
│  🎬 Video Translator Studio                         │
├────────────────────────┬────────────────────────────┤
│                        │  [Source State]             │
│    VIDEO PLAYER        │  ☐ Upload / URL input       │
│    (large, center)     │  ☐ Language settings         │
│                        │  ☐ Translate button          │
│                        │                              │
│                        │  — after translate —         │
│                        │                              │
│                        │  [Editor State]              │
│                        │  ☐ Scrollable subtitle list  │
│                        │  ☐ Each: time + original     │
│                        │  ☐      + editable text      │
│                        │  ☐ Auto-scroll to current    │
├────────────────────────┴────────────────────────────┤
│  TRIM BAR (waveform style, full width)               │
│  ┃00:00 ═══════════════════════════════════ 02:22┃  │
│  Duration: 02:22                                     │
├─────────────────────────────────────────────────────┤
│  [Export section when done]                          │
└─────────────────────────────────────────────────────┘
```

## Current State (Branch: `feature/video-download-and-editor`)

### What works:
- **Local file upload** → trim → translate → edit → export ✅
- **Trim with precision controls** (drag, keyboard arrows, editable time inputs) ✅
- **Translation** via OpenAI Whisper + GPT (with local upload) ✅
- **Export** (video with burned subtitles, SRT, TXT downloads) ✅
- **Re-encode trim** for frame-accurate cuts ✅
- **Thread-safe job management** with `threading.Lock` ✅
- **Job cleanup** (24h TTL, hourly sweep) ✅
- **Path traversal protection** on job IDs ✅

### What's broken:
- **YouTube/X download → translate**: Downloads work, but translation fails with "No speech detected in video" error
  - Downloaded file at `downloads/{job_id}_downloaded.mp4` has AV1 video + Opus audio
  - `has_audio_stream()` and `extract_audio()` work correctly (verified manually with ffprobe/ffmpeg)
  - **Likely root cause**: `find_video_path()` may pick up a stale file from previous job, OR `process_video()` gets wrong path
  - Need to add logging and test end-to-end
- **Editor UX is bad**: Current layout stacks video → timeline → subtitle list vertically in a narrow card

### Key files:
- `app.py` — FastAPI backend (~1430 lines). Endpoints: `/upload-only`, `/download-url`, `/download-status`, `/video-preview`, `/video-duration`, `/trim`, `/skip-trim`, `/process/{job_id}`, `/status`, `/subtitles` (GET/PUT), `/reburn`, `/download`, `/download-srt`, etc.
- `downloader.py` — yt-dlp wrapper (~200 lines). Functions: `detect_source()`, `get_video_info()`, `download_video()`, `is_valid_url()`
- `trimmer.py` — FFmpeg wrapper (~188 lines). Functions: `get_video_duration()`, `trim_video()`, `get_video_info()`, `format_time()`
- `static/index.html` — Current 5-step wizard HTML
- `static/script.js` — Current wizard JS (~1280 lines)
- `static/styles.css` — Current dark theme CSS (~1450 lines)
- `.env` — Contains `OPENAI_API_KEY`, `X_USERNAME`, `X_PASSWORD` (gitignored)

### Key backend functions:
- `generate_srt()` returns 4-tuple: `(srt_content, original_srt, token_usage, subtitles_data)`
- `process_video()` — main pipeline: extract audio → transcribe (Whisper) → translate (GPT) → embed subtitles (FFmpeg)
- `find_video_path(job_id)` — checks `job["video_path"]` first, then searches uploads/ and downloads/ dirs
- `has_audio_stream()` — ffprobe check for audio
- `embed_subtitles()` — burns subtitles + watermark using FFmpeg filter_complex
- `subtitles_to_srt()` — converts subtitle list back to SRT format
- `reburn_video_task()` — re-embeds edited subtitles into video

### Backend models:
- `DownloadRequest(url, quality)`, `TrimRequest(start_time, end_time)`, `ProcessRequest(language, translation_service, ollama_model)`
- `SubtitleUpdate(id, text)`, `SubtitlesUpdateRequest(subtitles: List[SubtitleUpdate])`
- `Progress` class with constants for each processing stage

---

## Detailed Implementation Todos

### Phase 1: Fix Critical Bugs (Backend)

- [ ] **1.1 Fix YouTube/X download → translate pipeline**
  - Add `print(f"Processing video at: {video_path}")` at start of `process_video()`
  - Add `print(f"Audio stream check: {has_audio_stream(video_path)}")` before extraction
  - Verify `find_video_path()` returns the correct downloaded file path (not a stale trimmed file)
  - Test with a real YouTube URL end-to-end
  - If Whisper returns empty segments, log the audio file size after extraction
  - File: `app.py`, functions: `process_video()`, `find_video_path()`

- [ ] **1.2 Ensure trimmed video audio integrity**
  - After `trim_video()` call in `/trim/{job_id}` endpoint, verify with `has_audio_stream()` that trimmed file has audio
  - The re-encode trim uses `-c:a aac` — should work but verify
  - If trim produces audio-less file, log warning and retry with different codec
  - File: `app.py`, endpoint: `trim_video_endpoint()`

### Phase 2: Unified Layout — HTML Restructure

- [ ] **2.1 Replace wizard with single-page studio layout**
  - Remove the wizard step indicator (`<div class="wizard-steps" id="wizard-indicator">`)
  - Remove all 5 `<div data-step="N">` containers
  - Create new structure:
    ```html
    <div class="studio-layout">
      <div class="studio-main">
        <div class="video-panel">
          <video id="studio-video" controls></video>
          <div class="current-subtitle" id="current-subtitle"></div>
        </div>
        <div class="control-panel" id="control-panel">
          <!-- States rendered here -->
        </div>
      </div>
      <div class="trim-section" id="trim-section" style="display:none">
        <!-- Trim bar -->
      </div>
      <div class="export-section" id="export-section" style="display:none">
        <!-- Download buttons + stats -->
      </div>
    </div>
    ```
  - File: `static/index.html`

- [ ] **2.2 Right panel — Source state HTML**
  - Upload zone (drag & drop + click to browse)
  - OR divider
  - URL input with auto-detect icon (YouTube/X)
  - URL preview (thumbnail + title + duration)
  - Language toggle (English / Hebrew)
  - Translation service toggle (OpenAI / Ollama)
  - "Translate" button (disabled until video loaded)
  - File: `static/index.html`

- [ ] **2.3 Right panel — Processing state HTML**
  - Progress bar with percentage
  - Step indicators: Extract → Transcribe → Translate → Embed
  - Status text message
  - File: `static/index.html`

- [ ] **2.4 Right panel — Editor state HTML**
  - Header: "Subtitles" + segment count
  - Scrollable list container
  - Each item: timestamp badge + original text (italic) + editable textarea
  - Footer: "Save & Export" button + "Skip Editing" link
  - File: `static/index.html`

- [ ] **2.5 Trim bar (persistent, full-width)**
  - Header: "Trim Clip" label + Duration badge
  - Thumbnail strip or waveform
  - Trim slider with start/end handles
  - Time inputs (start/end) with precision controls
  - Hint text about keyboard controls
  - File: `static/index.html`

- [ ] **2.6 Export section (bottom, inline)**
  - Stats row: processing time, input tokens, output tokens, cost
  - Download buttons: Video with subtitles, Translated SRT, Original SRT, TXT variants
  - "Process Another Video" button
  - Reburn progress bar (shown during re-embedding)
  - File: `static/index.html`

### Phase 3: CSS — Dark Theme Studio Styling

- [ ] **3.1 Studio layout CSS**
  - `.studio-layout` — flex column for overall page
  - `.studio-main` — CSS Grid: `grid-template-columns: 1fr 400px`
  - `.video-panel` — relative positioning, black background, rounded corners
  - `.control-panel` — flex column, fill height, internal scroll
  - `.trim-section` — full width, padding, border-top separator
  - `.export-section` — full width, padding, border-top separator
  - Responsive: `@media (max-width: 900px)` → single column stack
  - File: `static/styles.css`

- [ ] **3.2 Control panel state styling**
  - `.panel-source`, `.panel-processing`, `.panel-editor` — show/hide states
  - Upload zone: dashed border, hover highlight
  - URL input: icon slot, loading spinner, error display
  - Language/service toggles: pill buttons with active state
  - Progress bar: gradient fill, step icons with active/completed states
  - File: `static/styles.css`

- [ ] **3.3 Subtitle editor styling**
  - `.subtitle-list` — flex column, overflow-y auto, max-height calc(100vh - Xpx)
  - `.subtitle-item` — border-left transparent (3px), hover bg, cursor pointer
  - `.subtitle-item.active` — border-left accent color, subtle glow bg
  - `.subtitle-time` — monospace, accent color, small font
  - `.subtitle-original` — muted, italic, 0.78rem
  - `.subtitle-text-input` — dark bg, accent border on focus, resize none
  - Auto-scroll: `scrollIntoView({ behavior: 'smooth', block: 'nearest' })`
  - File: `static/styles.css`

- [ ] **3.4 Trim bar styling**
  - Reuse existing trim slider CSS (handles, range, thumbnails)
  - Adapt to new layout (full width below grid)
  - Duration badge with accent dot indicator
  - File: `static/styles.css`

- [ ] **3.5 Export section styling**
  - Stats row: grid of stat cards
  - Download buttons: primary for video, secondary for SRT/TXT
  - Reburn progress: inline bar with status text
  - File: `static/styles.css`

### Phase 4: JavaScript — State Machine Refactor

- [ ] **4.1 Core state machine**
  - Replace `goToStep(1-5)` with `setPanelState(state)` where state = 'source' | 'processing' | 'editor' | 'export'
  - `setPanelState()` hides all panel states, shows the requested one
  - Video player and trim bar are NOT affected by panel state changes
  - Track `currentPanelState` variable
  - File: `static/script.js`

- [ ] **4.2 Video loading logic**
  - When upload completes or download finishes:
    1. Set video source: `studioVideo.src = /video-preview/{jobId}`
    2. Fetch duration: `GET /video-duration/{jobId}`
    3. Show trim section
    4. Initialize trim UI (handles at 0 and duration)
    5. Enable "Translate" button
  - File: `static/script.js`

- [ ] **4.3 Source panel logic**
  - Upload file: `POST /upload-only` → load video → enable translate
  - URL input: debounce → `GET /video-info?url=` → show preview
  - "Next/Translate" click with URL: `POST /download-url` → setPanelState('processing' with download mode) → poll `/download-status` → on complete: load video
  - "Translate" click: apply trim if needed → `POST /process/{jobId}` → setPanelState('processing')
  - File: `static/script.js`

- [ ] **4.4 Processing panel logic**
  - Reuse existing `startStatusPolling()`, `updateProgressFromStatus()`, `updateProgress()`, `updateSteps()`
  - On completion: store stats → `loadSubtitles()` → `setPanelState('editor')`
  - On error: show error inline in control panel (not a separate page)
  - File: `static/script.js`

- [ ] **4.5 Editor panel logic**
  - `renderSubtitleEditor()`: populate subtitle list in right panel
  - `editorVideo.addEventListener('timeupdate')`: find current subtitle → highlight → auto-scroll
  - Click subtitle item: `studioVideo.currentTime = sub.start` + highlight
  - Textarea input: track `hasEdits = true`
  - "Save & Export": if hasEdits → PUT subtitles → POST reburn → show export; else → show export directly
  - File: `static/script.js`

- [ ] **4.6 Export logic**
  - Show export section (stats + download buttons) below the studio layout
  - If reburning: show progress bar, disable video download until complete
  - Download buttons: same endpoints as current (`/download/{jobId}`, `/download-srt/{jobId}`, etc.)
  - "Process Another Video": `resetApp()` → clear state → setPanelState('source')
  - File: `static/script.js`

- [ ] **4.7 Trim integration**
  - Trim bar is always visible once video loads (independent of panel state)
  - Keep all existing trim precision controls (drag, keyboard arrows, editable inputs)
  - Before translation: if trim changed → `POST /trim/{jobId}` → wait → then process
  - If trim unchanged → `POST /skip-trim/{jobId}` → then process
  - File: `static/script.js`

### Phase 5: Download UX Improvements

- [ ] **5.1 Download progress in control panel**
  - When downloading from URL, show progress bar + status in the right panel
  - Video player shows placeholder until download completes
  - Once done: auto-load video + show trim bar
  - File: `static/script.js`

- [ ] **5.2 Better error handling**
  - Inline error display in control panel (not separate error page)
  - Parse yt-dlp errors: geo-restricted, private, unavailable
  - "Try Again" button that stays in source state
  - File: `static/script.js`

### Phase 6: Polish & Testing

- [ ] **6.1 Subtitle overlay**
  - `current-subtitle` div positioned over video during playback
  - Semi-transparent background, clean font, RTL support for Hebrew
  - Updates in real-time via `timeupdate` event
  - File: `static/styles.css` + `static/script.js`

- [ ] **6.2 End-to-end testing**
  - Test: Local upload → trim → translate → edit → export
  - Test: YouTube URL → download → trim → translate → edit → export
  - Test: X/Twitter URL → download → translate → export
  - Test: Skip trim → translate → skip edit → export
  - Test: Edit subtitles → reburn → download
  - Run the app: `python app.py` on port 8000

---

## Technical Reference

### Backend endpoints (no changes needed except bug fix logging):
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/upload-only` | Upload video file, returns job_id |
| POST | `/download-url` | Start yt-dlp download, returns job_id |
| GET | `/download-status/{id}` | Poll download progress |
| GET | `/video-info?url=` | Get video metadata without downloading |
| GET | `/video-preview/{id}` | Stream video for HTML5 player |
| GET | `/video-duration/{id}` | Get video duration + resolution |
| POST | `/trim/{id}` | Trim video (re-encode for accuracy) |
| POST | `/skip-trim/{id}` | Skip trimming, use original |
| POST | `/process/{id}` | Start translation pipeline |
| GET | `/status/{id}` | Poll processing status |
| GET | `/subtitles/{id}` | Get subtitle segments for editing |
| PUT | `/subtitles/{id}` | Update edited subtitle text |
| POST | `/reburn/{id}` | Re-embed edited subtitles |
| GET | `/download/{id}` | Download final video |
| GET | `/download-srt/{id}` | Download translated SRT |
| GET | `/download-srt-txt/{id}` | Download translated TXT |
| GET | `/download-transcription/{id}` | Download original SRT |
| GET | `/download-transcription-txt/{id}` | Download original TXT |

### Key technical decisions:
- **Re-encode trim**: `-ss` before `-i` + `-c:v libx264 -c:a aac` for accuracy without slowness
- **Thread safety**: `threading.Lock()` wraps all background task mutations to `jobs` dict
- **Subtitle burn style**: `FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H40000000,BackColour=&H80000000,Outline=1,Shadow=2,Bold=0,MarginV=30`
- **Python**: 3.14, runs with `python app.py` on port 8000
- **Dependencies**: fastapi, uvicorn, openai, yt-dlp, python-multipart, python-dotenv
