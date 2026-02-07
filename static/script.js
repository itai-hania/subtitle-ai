/**
 * SubtitleAI - Wizard-based Frontend JavaScript
 * Handles video download, trimming, translation, subtitle editing, and export
 */

// ============================================
// State
// ============================================
let currentStep = 1;
let selectedFile = null;
let videoUrl = null;
let videoInfo = null;
let currentJobId = null;
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

// Wizard
const wizardIndicator = document.getElementById('wizard-indicator');
const wizardSteps = document.querySelectorAll('.wizard-step');

// Step 1: Source
const stepSource = document.getElementById('step-source');
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
const step1Next = document.getElementById('step1-next');

// Step 2: Trim
const stepTrim = document.getElementById('step-trim');
const trimVideo = document.getElementById('trim-video');
const trimSlider = document.getElementById('trim-slider');
const trimRange = document.getElementById('trim-range');
const trimHandleStart = document.getElementById('trim-handle-start');
const trimHandleEnd = document.getElementById('trim-handle-end');
const trimStartTime = document.getElementById('trim-start-time');
const trimEndTime = document.getElementById('trim-end-time');
const trimDurationEl = document.getElementById('trim-duration');
const downloadProgress = document.getElementById('download-progress');
const downloadProgressBar = document.getElementById('download-progress-bar');
const downloadStatusText = document.getElementById('download-status-text');
const step2Back = document.getElementById('step2-back');
const step2Skip = document.getElementById('step2-skip');
const step2Next = document.getElementById('step2-next');

// Step 3: Translate
const stepTranslate = document.getElementById('step-translate');
const languageToggle = document.getElementById('language-toggle');
const serviceToggle = document.getElementById('service-toggle');
const ollamaOptions = document.getElementById('ollama-options');
const ollamaModel = document.getElementById('ollama-model');
const processingSection = document.getElementById('processing-section');
const translateNav = document.getElementById('translate-nav');
const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');
const step3Back = document.getElementById('step3-back');
const startTranslationBtn = document.getElementById('start-translation');

// Step 4: Edit
const stepEdit = document.getElementById('step-edit');
const editorVideo = document.getElementById('editor-video');
const currentSubtitle = document.getElementById('current-subtitle');
const timeline = document.getElementById('timeline');
const timelinePlayhead = document.getElementById('timeline-playhead');
const subtitleList = document.getElementById('subtitle-list');
const step4Skip = document.getElementById('step4-skip');
const step4Next = document.getElementById('step4-next');

// Step 5: Export
const stepExport = document.getElementById('step-export');
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

// Error section
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const retryBtn = document.getElementById('retry-btn');

// ============================================
// Wizard Navigation
// ============================================

function goToStep(step) {
    currentStep = step;
    clearAllIntervals();
    
    // Update wizard indicator
    wizardSteps.forEach((el, index) => {
        const stepNum = index + 1;
        el.classList.remove('active', 'completed');
        if (stepNum < step) {
            el.classList.add('completed');
        } else if (stepNum === step) {
            el.classList.add('active');
        }
    });
    
    // Hide all steps
    [stepSource, stepTrim, stepTranslate, stepEdit, stepExport, errorSection].forEach(el => {
        el.style.display = 'none';
    });
    
    // Show current step
    switch (step) {
        case 1:
            stepSource.style.display = 'block';
            break;
        case 2:
            stepTrim.style.display = 'block';
            break;
        case 3:
            stepTranslate.style.display = 'block';
            break;
        case 4:
            stepEdit.style.display = 'block';
            break;
        case 5:
            stepExport.style.display = 'block';
            break;
    }
}

function showError(message) {
    clearAllIntervals();
    errorMessage.textContent = message;
    [stepSource, stepTrim, stepTranslate, stepEdit, stepExport].forEach(el => {
        el.style.display = 'none';
    });
    errorSection.style.display = 'block';
}

// ============================================
// Utility Functions
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
    if (hrs > 0) {
        return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================
// Step 1: Source Selection
// ============================================

// File upload handlers
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
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    const validTypes = ['.mp4', '.mov', '.avi', '.mkv'];
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(extension)) {
        alert('Please select a valid video file (MP4, MOV, AVI, or MKV)');
        return;
    }
    
    selectedFile = file;
    videoUrl = null;
    videoInfo = null;
    
    // Clear URL input
    videoUrlInput.value = '';
    urlPreview.style.display = 'none';
    urlSourceIcon.textContent = '';
    
    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    dropZone.style.display = 'none';
    filePreview.style.display = 'block';
    
    updateStep1NextButton();
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    dropZone.style.display = 'block';
    filePreview.style.display = 'none';
    updateStep1NextButton();
}

// URL input handlers
const handleUrlInput = debounce(async () => {
    const url = videoUrlInput.value.trim();
    
    if (!url) {
        urlSourceIcon.textContent = '';
        urlPreview.style.display = 'none';
        urlError.style.display = 'none';
        videoUrl = null;
        videoInfo = null;
        updateStep1NextButton();
        return;
    }
    
    // Detect source
    if (url.includes('youtube.com') || url.includes('youtu.be')) {
        urlSourceIcon.textContent = '▶️';
    } else if (url.includes('twitter.com') || url.includes('x.com')) {
        urlSourceIcon.textContent = '𝕏';
    } else {
        urlSourceIcon.textContent = '❓';
    }
    
    // Fetch video info
    urlLoading.style.display = 'flex';
    urlError.style.display = 'none';
    urlPreview.style.display = 'none';
    
    try {
        const response = await fetch(`/video-info?url=${encodeURIComponent(url)}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to fetch video info');
        }
        
        videoUrl = url;
        videoInfo = data;
        selectedFile = null;
        
        // Clear file selection
        clearFile();
        
        // Show preview
        urlThumbnail.src = data.thumbnail || '';
        urlTitle.textContent = data.title;
        urlDuration.textContent = `Duration: ${formatTimeFull(data.duration)}`;
        urlPreview.style.display = 'flex';
        
        if (data.source === 'youtube') {
            urlSourceIcon.textContent = '▶️';
        } else if (data.source === 'twitter') {
            urlSourceIcon.textContent = '𝕏';
        }
        
    } catch (error) {
        urlError.textContent = error.message;
        urlError.style.display = 'block';
        videoUrl = null;
        videoInfo = null;
    } finally {
        urlLoading.style.display = 'none';
        updateStep1NextButton();
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
    updateStep1NextButton();
});

function updateStep1NextButton() {
    step1Next.disabled = !(selectedFile || videoUrl);
}

step1Next.addEventListener('click', async () => {
    if (selectedFile) {
        // Upload file and get job ID
        await uploadFile();
    } else if (videoUrl) {
        // Download from URL
        await downloadFromUrl();
    }
});

async function uploadFile() {
    step1Next.disabled = true;
    step1Next.innerHTML = '<span class="spinner"></span> Uploading...';
    
    const formData = new FormData();
    formData.append('video', selectedFile);
    
    try {
        const response = await fetch('/upload-only', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Get video duration
        const durationResponse = await fetch(`/video-duration/${currentJobId}`);
        const durationData = await durationResponse.json();
        videoDuration = durationData.duration;
        
        // Set up trim video preview
        trimVideo.src = `/video-preview/${currentJobId}`;
        trimStart = 0;
        trimEnd = videoDuration;
        updateTrimUI();
        
        goToStep(2);
        
    } catch (error) {
        showError(error.message);
    } finally {
        step1Next.disabled = false;
        step1Next.innerHTML = 'Next <span class="btn-arrow">→</span>';
    }
}

async function downloadFromUrl() {
    goToStep(2);
    
    // Show download progress
    downloadProgress.style.display = 'block';
    step2Next.disabled = true;
    step2Skip.disabled = true;
    
    try {
        const response = await fetch('/download-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: videoUrl, quality: '720p' })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Download failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Poll download status
        await pollDownloadStatus();
        
    } catch (error) {
        showError(error.message);
    }
}

async function pollDownloadStatus() {
    return new Promise((resolve, reject) => {
        downloadInterval = setInterval(async () => {
            try {
                const response = await fetch(`/download-status/${currentJobId}`);
                const data = await response.json();
                
                downloadProgressBar.style.width = `${data.progress}%`;
                downloadStatusText.textContent = data.download_status === 'processing' 
                    ? 'Processing video...' 
                    : `Downloading: ${data.progress}%`;
                
                if (data.status === 'downloaded') {
                    clearInterval(downloadInterval);
                    downloadInterval = null;
                    downloadProgress.style.display = 'none';
                    
                    // Get video duration
                    const durationResponse = await fetch(`/video-duration/${currentJobId}`);
                    const durationData = await durationResponse.json();
                    videoDuration = durationData.duration;
                    
                    // Set up trim video preview
                    trimVideo.src = `/video-preview/${currentJobId}`;
                    trimStart = 0;
                    trimEnd = videoDuration;
                    updateTrimUI();
                    
                    step2Next.disabled = false;
                    step2Skip.disabled = false;
                    
                    resolve();
                } else if (data.status === 'error') {
                    clearInterval(downloadInterval);
                    downloadInterval = null;
                    reject(new Error(data.error || 'Download failed'));
                }
            } catch (error) {
                clearInterval(downloadInterval);
                downloadInterval = null;
                reject(error);
            }
        }, 1000);
    });
}

// ============================================
// Step 2: Trim Video
// ============================================

function updateTrimUI() {
    const duration = videoDuration || 1;
    const startPercent = (trimStart / duration) * 100;
    const endPercent = (trimEnd / duration) * 100;
    
    trimHandleStart.style.left = `${startPercent}%`;
    trimHandleEnd.style.left = `${endPercent}%`;
    trimRange.style.left = `${startPercent}%`;
    trimRange.style.right = `${100 - endPercent}%`;
    
    trimStartTime.textContent = formatTime(trimStart);
    trimEndTime.textContent = formatTime(trimEnd);
    trimDurationEl.textContent = `Duration: ${formatTime(trimEnd - trimStart)}`;
}

// Trim slider interaction
let draggingHandle = null;

function handleTrimDrag(e) {
    if (!draggingHandle) return;
    
    const rect = trimSlider.getBoundingClientRect();
    let percent = (e.clientX - rect.left) / rect.width;
    percent = Math.max(0, Math.min(1, percent));
    const time = percent * videoDuration;
    
    if (draggingHandle === 'start') {
        trimStart = Math.min(time, trimEnd - 1);
    } else {
        trimEnd = Math.max(time, trimStart + 1);
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
    // Clean up any previous drag listeners
    stopTrimDrag();
    draggingHandle = handle;
    document.addEventListener('mousemove', handleTrimDrag);
    document.addEventListener('mouseup', stopTrimDrag);
}

trimHandleStart.addEventListener('mousedown', (e) => startTrimDrag('start', e));
trimHandleEnd.addEventListener('mousedown', (e) => startTrimDrag('end', e));

// Sync video with trim handles
trimVideo.addEventListener('loadedmetadata', () => {
    if (!videoDuration || isNaN(trimVideo.duration)) {
        videoDuration = trimVideo.duration;
        trimEnd = videoDuration;
        updateTrimUI();
    }
});

trimVideo.addEventListener('error', () => {
    showError('Failed to load video preview. The file may be corrupt or unsupported.');
});

editorVideo.addEventListener('error', () => {
    console.error('Editor video failed to load');
});

step2Back.addEventListener('click', () => {
    goToStep(1);
});

step2Skip.addEventListener('click', async () => {
    await skipTrim();
});

step2Next.addEventListener('click', async () => {
    await applyTrim();
});

async function skipTrim() {
    try {
        await fetch(`/skip-trim/${currentJobId}`, { method: 'POST' });
        goToStep(3);
    } catch (error) {
        showError(error.message);
    }
}

async function applyTrim() {
    // Check if trim is actually needed
    if (Math.abs(trimStart - 0) < 0.5 && Math.abs(trimEnd - videoDuration) < 0.5) {
        await skipTrim();
        return;
    }
    
    step2Next.disabled = true;
    step2Next.innerHTML = '<span class="spinner"></span> Trimming...';
    
    try {
        const response = await fetch(`/trim/${currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_time: trimStart,
                end_time: trimEnd
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Trim failed');
        }
        
        goToStep(3);
        
    } catch (error) {
        showError(error.message);
    } finally {
        step2Next.disabled = false;
        step2Next.innerHTML = 'Trim & Continue <span class="btn-arrow">→</span>';
    }
}

// ============================================
// Step 3: Translation
// ============================================

function setActiveToggle(container, activeBtn) {
    container.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    activeBtn.classList.add('active');
}

languageToggle.addEventListener('click', (e) => {
    if (e.target.classList.contains('toggle-btn')) {
        setActiveToggle(languageToggle, e.target);
    }
});

serviceToggle.addEventListener('click', (e) => {
    if (e.target.classList.contains('toggle-btn')) {
        setActiveToggle(serviceToggle, e.target);
        const isOllama = e.target.dataset.value === 'ollama';
        ollamaOptions.style.display = isOllama ? 'flex' : 'none';
        
        if (isOllama) {
            loadOllamaModels();
        }
    }
});

async function loadOllamaModels() {
    try {
        const response = await fetch('/ollama-models');
        const data = await response.json();
        
        if (data.models && data.models.length > 0) {
            ollamaModel.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                ollamaModel.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load Ollama models:', error);
    }
}

function getSelectedLanguage() {
    const activeBtn = languageToggle.querySelector('.toggle-btn.active');
    return activeBtn ? activeBtn.dataset.value : 'English';
}

function getSelectedService() {
    const activeBtn = serviceToggle.querySelector('.toggle-btn.active');
    return activeBtn ? activeBtn.dataset.value : 'openai';
}

step3Back.addEventListener('click', () => {
    goToStep(2);
});

startTranslationBtn.addEventListener('click', async () => {
    await startTranslation();
});

async function startTranslation() {
    processingSection.style.display = 'block';
    translateNav.style.display = 'none';
    updateProgress(5, 'Starting translation...');
    updateSteps('upload');
    
    try {
        // Use the process endpoint for already downloaded/uploaded videos
        const response = await fetch(`/process/${currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                language: getSelectedLanguage(),
                translation_service: getSelectedService(),
                ollama_model: ollamaModel.value
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Start polling for status
        startStatusPolling();
        
    } catch (error) {
        showError(error.message);
    }
}

function startStatusPolling() {
    statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${currentJobId}`);
            const data = await response.json();
            
            updateProgressFromStatus(data);
            
            if (data.status === 'completed') {
                clearInterval(statusInterval);
                
                // Store stats
                if (data.elapsed_time) {
                    elapsedTimeEl.textContent = data.elapsed_time;
                }
                if (data.token_usage) {
                    inputTokensEl.textContent = data.token_usage.prompt_tokens?.toLocaleString() || '--';
                    outputTokensEl.textContent = data.token_usage.completion_tokens?.toLocaleString() || '--';
                    if (data.token_usage.total_cost !== undefined) {
                        totalCostEl.textContent = '$' + data.token_usage.total_cost.toFixed(4);
                    }
                }
                
                // Load subtitles for editor
                await loadSubtitles();
                
                // Go to edit step
                goToStep(4);
                
            } else if (data.status === 'error') {
                clearInterval(statusInterval);
                showError(data.error || 'An error occurred during processing');
            }
            
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 2000);
}

function updateProgressFromStatus(data) {
    const statusMessages = {
        'queued': 'Waiting in queue...',
        'extracting_audio': 'Extracting audio from video...',
        'transcribing': 'Transcribing speech with AI...',
        'translating': 'Translating subtitles...',
        'embedding_subtitles': 'Embedding subtitles into video...',
        'completed': 'Processing complete!'
    };
    
    const stepMapping = {
        'queued': 'upload',
        'extracting_audio': 'extract',
        'transcribing': 'transcribe',
        'translating': 'translate',
        'embedding_subtitles': 'embed',
        'completed': 'embed'
    };
    
    updateProgress(data.progress, statusMessages[data.status] || data.status);
    updateSteps(stepMapping[data.status]);
}

function updateProgress(percent, message) {
    progressBar.style.width = percent + '%';
    statusText.textContent = message;
}

function updateSteps(currentStepName) {
    const stepIds = {
        'upload': 'step-upload',
        'extract': 'step-extract',
        'transcribe': 'step-transcribe',
        'translate': 'step-translate-progress',
        'embed': 'step-embed'
    };
    const steps = ['upload', 'extract', 'transcribe', 'translate', 'embed'];
    const currentIndex = steps.indexOf(currentStepName);
    
    steps.forEach((step, index) => {
        const stepEl = document.getElementById(stepIds[step]);
        if (stepEl) {
            stepEl.classList.remove('active', 'completed');
            
            if (index < currentIndex) {
                stepEl.classList.add('completed');
            } else if (index === currentIndex) {
                stepEl.classList.add('active');
            }
        }
    });
}

// ============================================
// Step 4: Subtitle Editor
// ============================================

async function loadSubtitles() {
    try {
        const response = await fetch(`/subtitles/${currentJobId}`);
        const data = await response.json();
        
        subtitles = data.subtitles;
        renderSubtitleEditor();
        
    } catch (error) {
        console.error('Failed to load subtitles:', error);
    }
}

function renderSubtitleEditor() {
    // Set video source
    editorVideo.src = `/video-preview/${currentJobId}`;
    
    // Get video duration for timeline
    const totalDuration = subtitles.length > 0 
        ? subtitles[subtitles.length - 1].end 
        : 60;
    
    // Render timeline blocks
    timeline.innerHTML = '';
    subtitles.forEach((sub, index) => {
        const block = document.createElement('div');
        block.className = 'timeline-block';
        block.dataset.id = sub.id;
        
        const left = (sub.start / totalDuration) * 100;
        const width = ((sub.end - sub.start) / totalDuration) * 100;
        
        block.style.left = `${left}%`;
        block.style.width = `${Math.max(width, 0.5)}%`;
        block.textContent = sub.id;
        
        block.addEventListener('click', () => {
            editorVideo.currentTime = sub.start;
            highlightSubtitle(sub.id);
        });
        
        timeline.appendChild(block);
    });
    
    // Render subtitle list
    subtitleList.innerHTML = '';
    subtitles.forEach((sub, index) => {
        const item = document.createElement('div');
        item.className = 'subtitle-item';
        item.dataset.id = sub.id;
        
        item.innerHTML = `
            <div class="subtitle-time">
                ${formatTime(sub.start)} - ${formatTime(sub.end)}
            </div>
            <div class="subtitle-content">
                <div class="subtitle-original">${sub.original_text}</div>
                <textarea class="subtitle-text-input" rows="2" data-id="${sub.id}">${sub.text}</textarea>
            </div>
        `;
        
        const textarea = item.querySelector('textarea');
        textarea.addEventListener('input', () => {
            hasEdits = true;
            // Update subtitle in memory
            const idx = subtitles.findIndex(s => s.id === sub.id);
            if (idx !== -1) {
                subtitles[idx].text = textarea.value;
            }
        });
        
        textarea.addEventListener('focus', () => {
            highlightSubtitle(sub.id);
        });
        
        subtitleList.appendChild(item);
    });
    
    // Update current subtitle display on video timeupdate
    editorVideo.addEventListener('timeupdate', () => {
        const currentTime = editorVideo.currentTime;
        const totalDuration = subtitles.length > 0 
            ? subtitles[subtitles.length - 1].end 
            : 60;
        
        // Update playhead position
        const playheadPos = (currentTime / totalDuration) * 100;
        timelinePlayhead.style.left = `${playheadPos}%`;
        
        // Find current subtitle
        const current = subtitles.find(s => currentTime >= s.start && currentTime <= s.end);
        if (current) {
            currentSubtitle.textContent = current.text;
            currentSubtitle.style.display = 'block';
            highlightSubtitle(current.id);
        } else {
            currentSubtitle.style.display = 'none';
        }
    });
}

function highlightSubtitle(id) {
    // Highlight in timeline
    document.querySelectorAll('.timeline-block').forEach(block => {
        block.classList.toggle('active', parseInt(block.dataset.id) === id);
    });
    
    // Highlight in list
    document.querySelectorAll('.subtitle-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.id) === id);
    });
    
    // Scroll into view
    const activeItem = document.querySelector(`.subtitle-item[data-id="${id}"]`);
    if (activeItem) {
        activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

step4Skip.addEventListener('click', () => {
    goToStep(5);
});

step4Next.addEventListener('click', async () => {
    if (hasEdits) {
        await saveSubtitlesAndReburn();
    } else {
        goToStep(5);
    }
});

async function saveSubtitlesAndReburn() {
    step4Next.disabled = true;
    step4Next.innerHTML = '<span class="spinner"></span> Saving...';
    
    try {
        // Save subtitle edits
        const updates = subtitles.map(s => ({ id: s.id, text: s.text }));
        
        await fetch(`/subtitles/${currentJobId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subtitles: updates })
        });
        
        // Start re-burn
        await fetch(`/reburn/${currentJobId}`, { method: 'POST' });
        
        goToStep(5);
        
        // Show reburn progress
        reburnProgress.style.display = 'block';
        downloadVideoBtn.disabled = true;
        
        // Poll reburn status
        await pollReburnStatus();
        
    } catch (error) {
        showError(error.message);
    } finally {
        step4Next.disabled = false;
        step4Next.innerHTML = 'Save & Continue <span class="btn-arrow">→</span>';
    }
}

async function pollReburnStatus() {
    return new Promise((resolve, reject) => {
        reburnInterval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${currentJobId}`);
                const data = await response.json();
                
                reburnProgressBar.style.width = `${data.progress}%`;
                reburnStatusText.textContent = data.status === 'reburning' 
                    ? `Re-embedding subtitles: ${data.progress}%`
                    : 'Finishing up...';
                
                if (data.status === 'completed') {
                    clearInterval(reburnInterval);
                    reburnInterval = null;
                    reburnProgress.style.display = 'none';
                    downloadVideoBtn.disabled = false;
                    resolve();
                } else if (data.status === 'error') {
                    clearInterval(reburnInterval);
                    reburnInterval = null;
                    reject(new Error(data.error || 'Re-burn failed'));
                }
            } catch (error) {
                clearInterval(reburnInterval);
                reburnInterval = null;
                reject(error);
            }
        }, 1000);
    });
}

// ============================================
// Step 5: Export
// ============================================

downloadVideoBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/download/${currentJobId}`;
    }
});

downloadSrtBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/download-srt/${currentJobId}`;
    }
});

downloadSrtTxtBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/download-srt-txt/${currentJobId}`;
    }
});

downloadTranscriptionBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/download-transcription/${currentJobId}`;
    }
});

downloadTranscriptionTxtBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/download-transcription-txt/${currentJobId}`;
    }
});

newVideoBtn.addEventListener('click', () => {
    resetApp();
});

retryBtn.addEventListener('click', () => {
    resetApp();
});

function resetApp() {
    // Clear state
    currentStep = 1;
    selectedFile = null;
    videoUrl = null;
    videoInfo = null;
    currentJobId = null;
    trimStart = 0;
    trimEnd = 0;
    videoDuration = 0;
    subtitles = [];
    hasEdits = false;
    
    clearAllIntervals();
    
    // Reset UI
    clearFile();
    videoUrlInput.value = '';
    urlSourceIcon.textContent = '';
    urlPreview.style.display = 'none';
    urlError.style.display = 'none';
    processingSection.style.display = 'none';
    translateNav.style.display = 'flex';
    progressBar.style.width = '0%';
    downloadProgress.style.display = 'none';
    reburnProgress.style.display = 'none';
    
    // Reset wizard steps
    wizardSteps.forEach(step => {
        step.classList.remove('active', 'completed');
    });
    
    goToStep(1);
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('SubtitleAI Wizard initialized');
    goToStep(1);
});
