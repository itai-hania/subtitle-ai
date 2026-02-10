# Watermark Feature Specification

## Overview

Add a branded watermark to all translated videos. The watermark consists of:
- **Hebrew text**: ״המחנך הפיננסי״ (top line)
- **English text**: @FinancialEdux (bottom line)  
- **Logo**: The avatar image from `logo.jpg`

---

## Requirements

### Watermark Layout

```
┌─────────────────────────────────────────────────┐
│                                   ┌─────────────┤
│                                   │ @FinancialEduX  ┌──────┐
│                                   │ המחנך הפיננסי   │ LOGO │
│                                   └─────────────┴──────┘
│                                                 │
│                                                 │
│                                                 │
│                   [VIDEO CONTENT]               │
│                                                 │
│                                                 │
│                                                 │
│        ══════════════════════════               │
│              [SUBTITLES HERE]                   │
└─────────────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| **Position** | Upper-right corner |
| **Text arrangement** | English on top, Hebrew below |
| **Logo position** | Right of text block, vertically centered between both lines |
| **Opacity** | 70% (classic, visible but not distracting) |
| **Duration** | Throughout entire video |
| **Size** | Small & elegant (non-intrusive) |

### Text Styling

| Element | Value |
|---------|-------|
| Hebrew font | Arial/Helvetica (fallback-safe) |
| English font | Arial/Helvetica |
| Font size (Hebrew) | ~20px (scaled to video) |
| Font size (English) | ~16px |
| Text color | White (#FFFFFF) |
| Text shadow | 2px black shadow for visibility |

---

## Implementation Plan

### Files to Modify

#### [MODIFY] [app.py](file:///Users/itayy16/CursorProjects/SubtitleAI/app.py)

1. **Add watermark configuration constants** (near top of file)
   - Define watermark text, logo path, position, opacity, font settings

2. **Create new function `add_watermark()`** 
   - Uses FFmpeg's overlay filter to composite:
     - Text drawn using `drawtext` filter
     - Logo scaled and overlaid using `overlay` filter
   - Apply 70% opacity to both elements

3. **Modify `embed_subtitles()` function** (lines 351-379)
   - Integrate watermark filter chain with existing subtitle burning
   - Use FFmpeg filter complex to combine:
     - Subtitle filter
     - Watermark composite overlay
     - Logo overlay

### FFmpeg Filter Chain Strategy

```
[0:v] → subtitles → drawtext(Hebrew) → drawtext(English) → overlay(logo) → output
```

The FFmpeg command will use `-filter_complex` to chain:
1. `subtitles` filter (existing)
2. `drawtext` for Hebrew text (line 1)
3. `drawtext` for English text (line 2)
4. `scale` + `colorchannelmixer` (for logo opacity)
5. `overlay` for logo positioning

### Example FFmpeg Command Structure

```bash
ffmpeg -i video.mp4 -i logo.jpg \
  -filter_complex "
    [0:v]subtitles='subtitles.srt':force_style='...'[sub];
    [1:v]scale=60:-1,format=rgba,colorchannelmixer=aa=0.7[logo];
    [sub]drawtext=text='@FinancialEduX':fontsize=16:fontcolor=white@0.7:x=w-tw-80:y=15:shadowcolor=black:shadowx=2:shadowy=2[text1];
    [text1]drawtext=text='המחנך הפיננסי':fontfile=...:fontsize=20:fontcolor=white@0.7:x=w-tw-80:y=38:shadowcolor=black:shadowx=2:shadowy=2[text2];
    [text2][logo]overlay=W-w-10:10[out]
  " \
  -map "[out]" -map 0:a output.mp4
```

---

## Verification Plan

### Automated Testing

1. **Run the existing application** to ensure no regressions:
   ```bash
   cd /Users/itayy16/CursorProjects/SubtitleAI
   python -m pytest tests/ -v  # If tests exist
   ```

2. **Manual FFmpeg test** (validate filter chain works):
   ```bash
   ffmpeg -i test_video.mp4 -i logo.jpg \
     -filter_complex "[0:v]drawtext=text='Test':fontsize=24:fontcolor=white:x=10:y=10[v];[1:v]scale=50:-1[l];[v][l]overlay=W-w-10:10" \
     -y test_output.mp4
   ```

### Manual Verification

1. Upload a short test video through the web UI
2. Wait for processing to complete  
3. Download the resulting video
4. Verify:
   - [ ] Watermark appears in upper-right corner
   - [ ] Hebrew text ״המחנך הפיננסי״ is visible and readable
   - [ ] English text @FinancialEdux appears below Hebrew
   - [ ] Logo appears to the right of text
   - [ ] Watermark is at ~70% opacity (not too bright, not too faint)
   - [ ] Watermark persists throughout entire video
   - [ ] Subtitles still work correctly
   - [ ] Video/audio sync is maintained

---

## Notes

- The logo file is located at: `/Users/itayy16/CursorProjects/SubtitleAI/logo.jpg`
- Logo dimensions should be scaled proportionally (e.g., 50-60px height)
- Hebrew text requires proper RTL handling in FFmpeg (may need font with Hebrew support)
- Fallback: If Hebrew rendering fails, use the logo-only watermark
