/**
 * SubtitleAI Studio — Single-page state-driven frontend
 */

// ============================================
// State
// ============================================
let currentPanelState = 'source';
let selectedFile = null;
let videoUrl = null;
let videoInfo = null;
let currentJobId = null;
let videoLoaded = false; // true once video is in the player
let statusInterval = null;
let downloadInterval = null;
let reburnInterval = null;
let trimStart = 0;
let trimEnd = 0;
let videoDuration = 0;
let subtitles = [];
let hasEdits = false;

// ============================================
// DOM Elements
// ============================================

// Video
const studioVideo = document.getElementById('studio-video');
const videoPlaceholder = document.getElementById('video-placeholder');
const currentSubtitleEl = document.getElementById('current-subtitle');

// Control Panel states
const panelSource = document.getElementById('panel-source');
const panelDownloading = document.getElementById('panel-downloading');
const panelProcessing = document.getElementById('panel-processing');
const panelEditor = document.getElementById('panel-editor');

// Source panel
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const removeFileBtn = document.getElementById('remove-file');
const videoUrlInput = document.getElementById('video-url');
const urlSourceIcon = document.getElementById('url-source-icon');
const urlPreview = document.getElementById('url-preview');
const urlThumbnail = document.getElementById('url-thumbnail');
const urlTitle = document.getElementById('url-title');
const urlDuration = document.getElementById('url-duration');
const removeUrlBtn = document.getElementById('remove-url');
const urlLoading = document.getElementById('url-loading');
const urlError = document.getElementById('url-error');
const languageToggle = document.getElementById('language-toggle');
const serviceToggle = document.getElementById('service-toggle');
const ollamaOptions = document.getElementById('ollama-options');
const ollamaModel = document.getElementById('ollama-model');
const btnTranslate = document.getElementById('btn-translate');

// Downloading panel
const downloadProgressBar = document.getElementById('download-progress-bar');
const downloadStatusText = document.getElementById('download-status-text');

// Processing panel
const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');
const processingError = document.getElementById('processing-error');
const processingErrorText = document.getElementById('processing-error-text');
const processingRetryBtn = document.getElementById('processing-retry-btn');

// Editor panel
const subtitleList = document.getElementById('subtitle-list');
const subtitleCount = document.getElementById('subtitle-count');
const btnSkipEdit = document.getElementById('btn-skip-edit');
const btnSaveExport = document.getElementById('btn-save-export');

// Trim section
const trimSection = document.getElementById('trim-section');
const trimSlider = document.getElementById('trim-slider');
const trimRange = document.getElementById('trim-range');
const trimHandleStart = document.getElementById('trim-handle-start');
const trimHandleEnd = document.getElementById('trim-handle-end');
const trimStartTime = document.getElementById('trim-start-time');
const trimEndTime = document.getElementById('trim-end-time');
const trimDurationEl = document.getElementById('trim-duration');
const trimThumbnails = document.getElementById('trim-thumbnails');
const trimPlayhead = document.getElementById('trim-playhead');

// Export section
const exportSection = document.getElementById('export-section');
const elapsedTimeEl = document.getElementById('elapsed-time');
const inputTokensEl = document.getElementById('input-tokens');
const outputTokensEl = document.getElementById('output-tokens');
const totalCostEl = document.getElementById('total-cost');
const reburnProgress = document.getElementById('reburn-progress');
const reburnProgressBar = document.getElementById('reburn-progress-bar');
const reburnStatusText = document.getElementById('reburn-status-text');
const downloadVideoBtn = document.getElementById('download-video-btn');
const downloadSrtBtn = document.getElementById('download-srt-btn');
const downloadSrtTxtBtn = document.getElementById('download-srt-txt-btn');
const downloadTranscriptionBtn = document.getElementById('download-transcription-btn');
const downloadTranscriptionTxtBtn = document.getElementById('download-transcription-txt-btn');
const newVideoBtn = document.getElementById('new-video-btn');

// ============================================
// Panel State Machine
// ============================================

function setPanelState(state) {
    currentPanelState = state;
    // Hide all panel states
    [panelSource, panelDownloading, panelProcessing, panelEditor].forEach(el => {
        el.style.display = 'none';
    });
    // Show requested
    switch (state) {
        case 'source':
            panelSource.style.display = 'flex';
            break;
        case 'downloading':
            panelDownloading.style.display = 'flex';
            break;
        case 'processing':
            panelProcessing.style.display = 'flex';
            processingError.style.display = 'none';
            break;
        case 'editor':
            panelEditor.style.display = 'flex';
            break;
    }
}

// ============================================
// Utilities
// ============================================

function clearAllIntervals() {
    if (statusInterval) { clearInterval(statusInterval); statusInterval = null; }
    if (downloadInterval) { clearInterval(downloadInterval); downloadInterval = null; }
    if (reburnInterval) { clearInterval(reburnInterval); reburnInterval = null; }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatTimeFull(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatTimePrecise(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toFixed(1).padStart(4, '0')}`;
}

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

function showError(message) {
    // Inline error in processing panel
    processingErrorText.textContent = message;
    processingError.style.display = 'block';
    setPanelState('processing');
    // Hide progress UI
    progressBar.parentElement.style.display = 'none';
    statusText.style.display = 'none';
    document.querySelector('.progress-steps').style.display = 'none';
}

// ============================================
// Video Loading
// ============================================

function loadVideoPreview() {
    studioVideo.src = `/video-preview/${currentJobId}`;
    videoPlaceholder.style.display = 'none';
}

async function fetchAndShowDuration() {
    const resp = await fetch(`/video-duration/${currentJobId}`);
    const data = await resp.json();
    videoDuration = data.duration;
    trimStart = 0;
    trimEnd = videoDuration;
    updateTrimUI();
    trimSection.style.display = 'block';
}

// ============================================
// Source Panel — File Upload
// ============================================

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    const validTypes = ['.mp4', '.mov', '.avi', '.mkv'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validTypes.includes(ext)) {
        alert('Please select a valid video file (MP4, MOV, AVI, or MKV)');
        return;
    }

    selectedFile = file;
    videoUrl = null;
    videoInfo = null;
    videoLoaded = false;
    videoUrlInput.value = '';
    urlPreview.style.display = 'none';
    urlSourceIcon.textContent = '';

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    dropZone.style.display = 'none';
    filePreview.style.display = 'block';
    updateTranslateButton();
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    dropZone.style.display = 'block';
    filePreview.style.display = 'none';
    updateTranslateButton();
}

// ============================================
// Source Panel — URL Input
// ============================================

const handleUrlInput = debounce(async () => {
    const url = videoUrlInput.value.trim();
    if (!url) {
        urlSourceIcon.textContent = '';
        urlPreview.style.display = 'none';
        urlError.style.display = 'none';
        videoUrl = null;
        videoInfo = null;
        updateTranslateButton();
        return;
    }

    // Detect source icon
    if (url.includes('youtube.com') || url.includes('youtu.be')) {
        urlSourceIcon.textContent = '▶️';
    } else if (url.includes('twitter.com') || url.includes('x.com')) {
        urlSourceIcon.textContent = '𝕏';
    } else {
        urlSourceIcon.textContent = '❓';
    }

    urlLoading.style.display = 'flex';
    urlError.style.display = 'none';
    urlPreview.style.display = 'none';

    try {
        const response = await fetch(`/video-info?url=${encodeURIComponent(url)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Failed to fetch video info');

        videoUrl = url;
        videoInfo = data;
        selectedFile = null;
        clearFile();

        urlThumbnail.src = data.thumbnail || '';
        urlTitle.textContent = data.title;
        urlDuration.textContent = `Duration: ${formatTimeFull(data.duration)}`;
        urlPreview.style.display = 'flex';

    } catch (error) {
        urlError.textContent = error.message;
        urlError.style.display = 'block';
        videoUrl = null;
        videoInfo = null;
    } finally {
        urlLoading.style.display = 'none';
        updateTranslateButton();
    }
}, 500);

videoUrlInput.addEventListener('input', handleUrlInput);

removeUrlBtn.addEventListener('click', () => {
    videoUrlInput.value = '';
    urlSourceIcon.textContent = '';
    urlPreview.style.display = 'none';
    urlError.style.display = 'none';
    videoUrl = null;
    videoInfo = null;
    updateTranslateButton();
});

// ============================================
// Source Panel — Toggles
// ============================================

function setActiveToggle(container, activeBtn) {
    container.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
    activeBtn.classList.add('active');
}

languageToggle.addEventListener('click', (e) => {
    if (e.target.classList.contains('toggle-btn')) setActiveToggle(languageToggle, e.target);
});

serviceToggle.addEventListener('click', (e) => {
    if (e.target.classList.contains('toggle-btn')) {
        setActiveToggle(serviceToggle, e.target);
        const isOllama = e.target.dataset.value === 'ollama';
        ollamaOptions.style.display = isOllama ? 'flex' : 'none';
        if (isOllama) loadOllamaModels();
    }
});

async function loadOllamaModels() {
    try {
        const resp = await fetch('/ollama-models');
        const data = await resp.json();
        if (data.models && data.models.length > 0) {
            ollamaModel.innerHTML = '';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                ollamaModel.appendChild(opt);
            });
        }
    } catch (e) {
        console.error('Failed to load Ollama models:', e);
    }
}

function getSelectedLanguage() {
    const btn = languageToggle.querySelector('.toggle-btn.active');
    return btn ? btn.dataset.value : 'English';
}

function getSelectedService() {
    const btn = serviceToggle.querySelector('.toggle-btn.active');
    return btn ? btn.dataset.value : 'openai';
}

// ============================================
// Translate Button
// ============================================

function updateTranslateButton() {
    if (videoLoaded) {
        btnTranslate.disabled = false;
        btnTranslate.innerHTML = '<span class="btn-icon">✨</span> Translate';
    } else {
        btnTranslate.disabled = !(selectedFile || videoUrl);
        if (selectedFile) {
            btnTranslate.innerHTML = '<span class="btn-icon">📤</span> Upload & Preview';
        } else if (videoUrl) {
            btnTranslate.innerHTML = '<span class="btn-icon">⬇️</span> Download & Preview';
        } else {
            btnTranslate.innerHTML = '<span class="btn-icon">✨</span> Translate';
        }
    }
}

btnTranslate.addEventListener('click', async () => {
    if (videoLoaded) {
        // Phase 2: start translation
        await applyTrimAndTranslate();
    } else if (selectedFile) {
        // Phase 1: upload and preview
        await uploadAndPreview();
    } else if (videoUrl) {
        // Phase 1: download and preview
        await downloadAndPreview();
    }
});

async function uploadAndPreview() {
    btnTranslate.disabled = true;
    btnTranslate.innerHTML = '<span class="spinner"></span> Uploading...';

    try {
        const formData = new FormData();
        formData.append('video', selectedFile);

        const resp = await fetch('/upload-only', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await resp.json();
        currentJobId = data.job_id;

        loadVideoPreview();
        await fetchAndShowDuration();
        videoLoaded = true;
        updateTranslateButton();

    } catch (error) {
        showError(error.message);
    } finally {
        btnTranslate.disabled = false;
        updateTranslateButton();
    }
}

async function downloadAndPreview() {
    btnTranslate.disabled = true;
    setPanelState('downloading');
    downloadProgressBar.style.width = '0%';
    downloadStatusText.textContent = 'Starting download...';

    try {
        const resp = await fetch('/download-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: videoUrl, quality: '720p' })
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Download failed');
        }
        const data = await resp.json();
        currentJobId = data.job_id;

        // Poll download status
        await pollDownloadStatus();

        // Download complete — load video, go back to source panel
        loadVideoPreview();
        await fetchAndShowDuration();
        videoLoaded = true;
        setPanelState('source');
        updateTranslateButton();

    } catch (error) {
        showError(error.message);
    } finally {
        btnTranslate.disabled = false;
        updateTranslateButton();
    }
}

function pollDownloadStatus() {
    return new Promise((resolve, reject) => {
        downloadInterval = setInterval(async () => {
            try {
                const resp = await fetch(`/download-status/${currentJobId}`);
                const data = await resp.json();

                downloadProgressBar.style.width = `${data.progress}%`;
                downloadStatusText.textContent = data.download_status === 'processing'
                    ? 'Processing video...'
                    : `Downloading: ${data.progress}%`;

                if (data.status === 'downloaded') {
                    clearInterval(downloadInterval);
                    downloadInterval = null;
                    resolve();
                } else if (data.status === 'error') {
                    clearInterval(downloadInterval);
                    downloadInterval = null;
                    reject(new Error(data.error || 'Download failed'));
                }
            } catch (err) {
                clearInterval(downloadInterval);
                downloadInterval = null;
                reject(err);
            }
        }, 1000);
    });
}

// ============================================
// Trim + Translate Flow
// ============================================

async function applyTrimAndTranslate() {
    // Check if trim was changed
    const trimChanged = Math.abs(trimStart) > 0.5 || Math.abs(trimEnd - videoDuration) > 0.5;

    if (trimChanged) {
        setPanelState('processing');
        updateProgress(5, 'Trimming video...');
        try {
            const resp = await fetch(`/trim/${currentJobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start_time: trimStart, end_time: trimEnd })
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Trim failed');
            }
        } catch (error) {
            showError(error.message);
            return;
        }
    } else {
        try {
            await fetch(`/skip-trim/${currentJobId}`, { method: 'POST' });
        } catch (e) {
            // Non-critical
        }
    }

    await startTranslation();
}

async function startTranslation() {
    setPanelState('processing');
    updateProgress(5, 'Starting translation...');
    updateSteps('extract');

    // Show progress UI (might have been hidden by error)
    progressBar.parentElement.style.display = '';
    statusText.style.display = '';
    document.querySelector('.progress-steps').style.display = '';

    try {
        const resp = await fetch(`/process/${currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                language: getSelectedLanguage(),
                translation_service: getSelectedService(),
                ollama_model: ollamaModel.value
            })
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Processing failed');
        }
        const data = await resp.json();
        currentJobId = data.job_id;

        startStatusPolling();
    } catch (error) {
        showError(error.message);
    }
}

function startStatusPolling() {
    statusInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/status/${currentJobId}`);
            const data = await resp.json();

            updateProgressFromStatus(data);

            if (data.status === 'completed') {
                clearInterval(statusInterval);
                statusInterval = null;

                // Store stats
                if (data.elapsed_time) elapsedTimeEl.textContent = data.elapsed_time;
                if (data.token_usage) {
                    inputTokensEl.textContent = data.token_usage.prompt_tokens?.toLocaleString() || '--';
                    outputTokensEl.textContent = data.token_usage.completion_tokens?.toLocaleString() || '--';
                    if (data.token_usage.total_cost !== undefined) {
                        totalCostEl.textContent = '$' + data.token_usage.total_cost.toFixed(4);
                    }
                }

                // Hide trim section — no longer needed
                trimSection.style.display = 'none';

                // Reload video with burned-in subtitles
                studioVideo.src = `/video-preview/${currentJobId}?t=${Date.now()}`;

                await loadSubtitles();
                setPanelState('editor');

            } else if (data.status === 'error') {
                clearInterval(statusInterval);
                statusInterval = null;
                showError(data.error || 'An error occurred during processing');
            }
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 2000);
}

function updateProgressFromStatus(data) {
    const msgs = {
        'queued': 'Waiting in queue...',
        'extracting_audio': 'Extracting audio from video...',
        'transcribing': 'Transcribing speech with AI...',
        'translating': 'Translating subtitles...',
        'embedding_subtitles': 'Embedding subtitles into video...',
        'completed': 'Processing complete!'
    };
    const stepMap = {
        'queued': 'extract',
        'extracting_audio': 'extract',
        'transcribing': 'transcribe',
        'translating': 'translate',
        'embedding_subtitles': 'embed',
        'completed': 'embed'
    };
    updateProgress(data.progress, msgs[data.status] || data.status);
    updateSteps(stepMap[data.status]);
}

function updateProgress(percent, message) {
    progressBar.style.width = percent + '%';
    statusText.textContent = message;
}

function updateSteps(currentStepName) {
    const stepIds = {
        'extract': 'step-extract',
        'transcribe': 'step-transcribe',
        'translate': 'step-translate-progress',
        'embed': 'step-embed'
    };
    const order = ['extract', 'transcribe', 'translate', 'embed'];
    const idx = order.indexOf(currentStepName);

    order.forEach((s, i) => {
        const el = document.getElementById(stepIds[s]);
        if (el) {
            el.classList.remove('active', 'completed');
            if (i < idx) el.classList.add('completed');
            else if (i === idx) el.classList.add('active');
        }
    });
}

// ============================================
// Subtitle Editor
// ============================================

async function loadSubtitles() {
    try {
        const resp = await fetch(`/subtitles/${currentJobId}`);
        const data = await resp.json();
        subtitles = data.subtitles;
        renderSubtitleEditor();
    } catch (e) {
        console.error('Failed to load subtitles:', e);
    }
}

function renderSubtitleEditor() {
    subtitleCount.textContent = `${subtitles.length} segments`;

    subtitleList.innerHTML = '';
    subtitles.forEach(sub => {
        const item = document.createElement('div');
        item.className = 'subtitle-item';
        item.dataset.id = sub.id;

        const originalDiv = document.createElement('div');
        originalDiv.className = 'subtitle-original';
        originalDiv.dir = 'auto';
        originalDiv.textContent = sub.original_text;

        const textarea = document.createElement('textarea');
        textarea.className = 'subtitle-text-input';
        textarea.rows = 2;
        textarea.dir = 'auto';
        textarea.dataset.id = sub.id;
        textarea.value = sub.text;

        const timeDiv = document.createElement('div');
        timeDiv.className = 'subtitle-time';
        timeDiv.textContent = `${formatTime(sub.start)} → ${formatTime(sub.end)}`;

        item.appendChild(timeDiv);
        item.appendChild(originalDiv);
        item.appendChild(textarea);

        item.addEventListener('click', (e) => {
            if (e.target.tagName !== 'TEXTAREA') {
                studioVideo.currentTime = sub.start;
                highlightSubtitle(sub.id);
            }
        });

        textarea.addEventListener('input', () => {
            hasEdits = true;
            const idx = subtitles.findIndex(s => s.id === sub.id);
            if (idx !== -1) subtitles[idx].text = textarea.value;
        });
        textarea.addEventListener('focus', () => highlightSubtitle(sub.id));

        subtitleList.appendChild(item);
    });

    // Set up subtitle overlay sync
    setupSubtitleSync();
}

function setupSubtitleSync() {
    // Remove previous listener to avoid duplicates
    studioVideo.removeEventListener('timeupdate', onVideoTimeUpdate);
    studioVideo.addEventListener('timeupdate', onVideoTimeUpdate);
}

function onVideoTimeUpdate() {
    const t = studioVideo.currentTime;
    const current = subtitles.find(s => t >= s.start && t <= s.end);

    if (current) {
        // Only show overlay subtitle when NOT in editor mode (burned video already has them)
        if (currentPanelState !== 'editor') {
            currentSubtitleEl.textContent = current.text;
            currentSubtitleEl.dir = 'auto';
            currentSubtitleEl.style.display = 'block';
        } else {
            currentSubtitleEl.style.display = 'none';
        }
        if (currentPanelState === 'editor') highlightSubtitle(current.id);
    } else {
        currentSubtitleEl.style.display = 'none';
    }

    // Update trim playhead
    if (videoDuration > 0) {
        trimPlayhead.style.left = `${(t / videoDuration) * 100}%`;
    }
}

function highlightSubtitle(id) {
    document.querySelectorAll('.subtitle-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.id) === id);
    });
    // No auto-scroll — let user scroll freely while editing
}

// Editor actions
btnSkipEdit.addEventListener('click', () => {
    showExportSection();
});

btnSaveExport.addEventListener('click', async () => {
    if (hasEdits) {
        await saveSubtitlesAndReburn();
    } else {
        showExportSection();
    }
});

async function saveSubtitlesAndReburn() {
    btnSaveExport.disabled = true;
    btnSaveExport.innerHTML = '<span class="spinner"></span> Saving...';

    try {
        const updates = subtitles.map(s => ({ id: s.id, text: s.text }));
        await fetch(`/subtitles/${currentJobId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subtitles: updates })
        });

        await fetch(`/reburn/${currentJobId}`, { method: 'POST' });

        showExportSection();

        // Show reburn progress
        reburnProgress.style.display = 'block';
        downloadVideoBtn.disabled = true;
        await pollReburnStatus();
    } catch (error) {
        showError(error.message);
    } finally {
        btnSaveExport.disabled = false;
        btnSaveExport.innerHTML = '<span class="btn-icon">💾</span> Save & Export';
    }
}

function pollReburnStatus() {
    return new Promise((resolve, reject) => {
        reburnInterval = setInterval(async () => {
            try {
                const resp = await fetch(`/status/${currentJobId}`);
                const data = await resp.json();

                reburnProgressBar.style.width = `${data.progress}%`;
                reburnStatusText.textContent = data.status === 'reburning'
                    ? `Re-embedding subtitles: ${data.progress}%`
                    : 'Finishing up...';

                if (data.status === 'completed') {
                    clearInterval(reburnInterval);
                    reburnInterval = null;
                    reburnProgress.style.display = 'none';
                    downloadVideoBtn.disabled = false;
                    // Reload video with new burned subtitles
                    studioVideo.src = `/video-preview/${currentJobId}?t=${Date.now()}`;
                    resolve();
                } else if (data.status === 'error') {
                    clearInterval(reburnInterval);
                    reburnInterval = null;
                    reject(new Error(data.error || 'Re-burn failed'));
                }
            } catch (err) {
                clearInterval(reburnInterval);
                reburnInterval = null;
                reject(err);
            }
        }, 1000);
    });
}

// ============================================
// Export Section
// ============================================

function showExportSection() {
    exportSection.style.display = 'block';
    // Keep editor visible so user can still view subtitles
}

downloadVideoBtn.addEventListener('click', () => {
    if (currentJobId) window.location.href = `/download/${currentJobId}`;
});
downloadSrtBtn.addEventListener('click', () => {
    if (currentJobId) window.location.href = `/download-srt/${currentJobId}`;
});
downloadSrtTxtBtn.addEventListener('click', () => {
    if (currentJobId) window.location.href = `/download-srt-txt/${currentJobId}`;
});
downloadTranscriptionBtn.addEventListener('click', () => {
    if (currentJobId) window.location.href = `/download-transcription/${currentJobId}`;
});
downloadTranscriptionTxtBtn.addEventListener('click', () => {
    if (currentJobId) window.location.href = `/download-transcription-txt/${currentJobId}`;
});

newVideoBtn.addEventListener('click', resetApp);

processingRetryBtn.addEventListener('click', resetApp);

// ============================================
// Trim Controls
// ============================================

function updateTrimUI() {
    const d = videoDuration || 1;
    const sp = (trimStart / d) * 100;
    const ep = (trimEnd / d) * 100;

    trimHandleStart.style.left = `${sp}%`;
    trimHandleEnd.style.left = `${ep}%`;
    trimRange.style.left = `${sp}%`;
    trimRange.style.right = `${100 - ep}%`;

    trimStartTime.value = formatTimePrecise(trimStart);
    trimEndTime.value = formatTimePrecise(trimEnd);
    trimDurationEl.textContent = `Duration: ${formatTimePrecise(trimEnd - trimStart)}`;
}

// Trim drag
let draggingHandle = null;

function handleTrimDrag(e) {
    if (!draggingHandle) return;
    const rect = trimSlider.getBoundingClientRect();
    let pct = (e.clientX - rect.left) / rect.width;
    pct = Math.max(0, Math.min(1, pct));
    let t = Math.round(pct * videoDuration * 10) / 10;

    if (draggingHandle === 'start') {
        trimStart = Math.max(0, Math.min(t, trimEnd - 0.1));
        studioVideo.currentTime = trimStart;
    } else {
        trimEnd = Math.min(videoDuration, Math.max(t, trimStart + 0.1));
        studioVideo.currentTime = trimEnd;
    }
    updateTrimUI();
}

function stopTrimDrag() {
    draggingHandle = null;
    document.removeEventListener('mousemove', handleTrimDrag);
    document.removeEventListener('mouseup', stopTrimDrag);
}

function startTrimDrag(handle, e) {
    e.preventDefault();
    stopTrimDrag();
    studioVideo.pause();
    draggingHandle = handle;
    document.addEventListener('mousemove', handleTrimDrag);
    document.addEventListener('mouseup', stopTrimDrag);
}

trimHandleStart.addEventListener('mousedown', (e) => startTrimDrag('start', e));
trimHandleEnd.addEventListener('mousedown', (e) => startTrimDrag('end', e));

// Click slider to seek
trimSlider.addEventListener('click', (e) => {
    if (e.target.closest('.trim-handle')) return;
    const rect = trimSlider.getBoundingClientRect();
    let pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    studioVideo.currentTime = pct * videoDuration;
});

// Keyboard controls
trimHandleStart.setAttribute('tabindex', '0');
trimHandleEnd.setAttribute('tabindex', '0');

function handleTrimKeydown(handle, e) {
    const step = e.shiftKey ? 1.0 : 0.1;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const delta = e.key === 'ArrowRight' ? step : -step;
        if (handle === 'start') {
            trimStart = Math.max(0, Math.min(trimStart + delta, trimEnd - 0.1));
            studioVideo.currentTime = trimStart;
        } else {
            trimEnd = Math.min(videoDuration, Math.max(trimEnd + delta, trimStart + 0.1));
            studioVideo.currentTime = trimEnd;
        }
        trimStart = Math.round(trimStart * 10) / 10;
        trimEnd = Math.round(trimEnd * 10) / 10;
        updateTrimUI();
    }
}

trimHandleStart.addEventListener('keydown', (e) => handleTrimKeydown('start', e));
trimHandleEnd.addEventListener('keydown', (e) => handleTrimKeydown('end', e));

// Time inputs
function parseTimeInput(value) {
    value = value.trim();
    const match = value.match(/^(\d+):(\d+\.?\d*)$/);
    if (match) return parseInt(match[1]) * 60 + parseFloat(match[2]);
    const num = parseFloat(value);
    return isNaN(num) ? null : num;
}

trimStartTime.addEventListener('change', () => {
    const t = parseTimeInput(trimStartTime.value);
    if (t !== null && t >= 0 && t < trimEnd) {
        trimStart = Math.round(t * 10) / 10;
        studioVideo.currentTime = trimStart;
        updateTrimUI();
    } else {
        trimStartTime.value = formatTimePrecise(trimStart);
    }
});

trimEndTime.addEventListener('change', () => {
    const t = parseTimeInput(trimEndTime.value);
    if (t !== null && t > trimStart && t <= videoDuration) {
        trimEnd = Math.round(t * 10) / 10;
        studioVideo.currentTime = trimEnd;
        updateTrimUI();
    } else {
        trimEndTime.value = formatTimePrecise(trimEnd);
    }
});

// Thumbnail generation
studioVideo.addEventListener('loadedmetadata', () => {
    if (!videoDuration || isNaN(studioVideo.duration)) {
        videoDuration = studioVideo.duration;
        trimEnd = videoDuration;
        updateTrimUI();
    }
    generateThumbnails();
});

function generateThumbnails() {
    if (!studioVideo.duration || studioVideo.duration === Infinity) return;

    trimThumbnails.innerHTML = '';
    const count = 15;
    const interval = studioVideo.duration / count;

    const thumbVideo = document.createElement('video');
    thumbVideo.src = studioVideo.src;
    thumbVideo.crossOrigin = 'anonymous';
    thumbVideo.muted = true;
    thumbVideo.preload = 'auto';

    for (let i = 0; i < count; i++) {
        const canvas = document.createElement('canvas');
        canvas.className = 'trim-thumbnail';
        canvas.width = 120;
        canvas.height = 68;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, 120, 68);
        trimThumbnails.appendChild(canvas);
    }

    let generated = 0;

    thumbVideo.addEventListener('loadeddata', () => captureFrame(0));

    function captureFrame(index) {
        if (index >= count) { thumbVideo.remove(); return; }
        thumbVideo.currentTime = index * interval;
    }

    thumbVideo.addEventListener('seeked', () => {
        const index = Math.round(thumbVideo.currentTime / interval);
        if (index < count) {
            const canvas = trimThumbnails.children[index];
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(thumbVideo, 0, 0, 120, 68);
            }
        }
        generated++;
        if (generated < count) captureFrame(generated);
        else thumbVideo.remove();
    });
}

// ============================================
// Reset
// ============================================

function resetApp() {
    selectedFile = null;
    videoUrl = null;
    videoInfo = null;
    currentJobId = null;
    videoLoaded = false;
    trimStart = 0;
    trimEnd = 0;
    videoDuration = 0;
    subtitles = [];
    hasEdits = false;

    clearAllIntervals();

    // Reset video
    studioVideo.removeAttribute('src');
    studioVideo.load();
    videoPlaceholder.style.display = 'flex';
    currentSubtitleEl.style.display = 'none';

    // Reset source panel
    clearFile();
    videoUrlInput.value = '';
    urlSourceIcon.textContent = '';
    urlPreview.style.display = 'none';
    urlError.style.display = 'none';

    // Reset progress
    progressBar.style.width = '0%';
    downloadProgressBar.style.width = '0%';
    reburnProgress.style.display = 'none';

    // Reset progress UI visibility
    progressBar.parentElement.style.display = '';
    statusText.style.display = '';
    const stepsEl = document.querySelector('.progress-steps');
    if (stepsEl) stepsEl.style.display = '';

    // Hide sections
    trimSection.style.display = 'none';
    exportSection.style.display = 'none';

    setPanelState('source');
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('SubtitleAI Studio initialized');
    setPanelState('source');
});
