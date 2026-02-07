# 🎬 SubtitleAI

**AI-powered video subtitle generator with automatic translation to Hebrew (and other languages).**

Upload a video, get back a subtitled version with accurate translations — all powered by OpenAI Whisper for transcription and GPT for translation.

---

## ✨ Features

- 🎙️ **Automatic Transcription** — Uses OpenAI Whisper API for accurate speech-to-text
- 🌐 **Multi-Language Translation** — Translates subtitles to Hebrew, Spanish, French, and more
- 📝 **RTL Support** — Proper Right-to-Left rendering for Hebrew subtitles
- ⬇️ **Multiple Download Formats** — Get your video with burned-in subtitles, SRT files, or plain text
- 🎨 **Clean Web Interface** — Simple drag-and-drop UI for easy use
- 💰 **Cost Tracking** — Displays token usage and estimated API costs

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg installed on your system
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/itai-hania/subtitle-ai.git
   cd subtitle-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file with your keys:
   # OPENAI_API_KEY=your_openai_api_key_here
   # X_USERNAME=your_twitter_username (optional)
   # X_PASSWORD=your_twitter_password (optional)
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:8000
   ```

---

## 📖 Usage

1. **Upload a Video** — Drag and drop or click to select a video file (MP4, MOV, AVI, MKV)
2. **Select Target Language** — Choose the language for subtitles (e.g., Hebrew)
3. **Process** — Wait for transcription and translation to complete
4. **Download** — Get your video with subtitles, or download SRT/TXT files separately

---

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload video and start processing |
| `/status/{job_id}` | GET | Check processing status |
| `/download/{job_id}` | GET | Download video with subtitles |
| `/download-srt/{job_id}` | GET | Download translated SRT file |
| `/download-transcription/{job_id}` | GET | Download original SRT file |
| `/download-srt-txt/{job_id}` | GET | Download translated plain text |
| `/download-transcription-txt/{job_id}` | GET | Download original plain text |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for Whisper and GPT |

### Translation Settings

The app uses `gpt-5-nano` with optimized settings:
- **Chunk size**: 50 segments per batch
- **Reasoning effort**: Low (minimizes hidden tokens)
- **Max tokens**: 10,000 per request

---

## 📁 Project Structure

```
subtitle-ai/
├── app.py              # FastAPI backend application
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not tracked by git)
├── .gitignore          # Git ignore rules
├── static/
│   ├── index.html      # Web UI
│   ├── script.js       # Frontend JavaScript
│   └── styles.css      # Styling
├── uploads/            # Uploaded videos (gitignored)
├── outputs/            # Processed videos (gitignored)
└── logs/               # Translation logs (gitignored)
```

---

## 🔧 Technical Details

- **Backend**: FastAPI (Python)
- **Transcription**: OpenAI Whisper API
- **Translation**: OpenAI GPT-5-nano
- **Video Processing**: FFmpeg
- **Frontend**: Vanilla HTML/CSS/JavaScript

---

## 📄 License

MIT License — feel free to use and modify.

---

## 🙏 Acknowledgments

- [OpenAI](https://openai.com) for Whisper and GPT APIs
- [FFmpeg](https://ffmpeg.org) for video processing
