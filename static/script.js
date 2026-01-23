/**
 * SubtitleAI - Frontend JavaScript
 * Handles file upload, processing, and download functionality
 */

// State
let selectedFile = null;
let currentJobId = null;
let statusInterval = null;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const removeFileBtn = document.getElementById('remove-file');
const processBtn = document.getElementById('process-btn');
const languageToggle = document.getElementById('language-toggle');
const serviceToggle = document.getElementById('service-toggle');
const ollamaOptions = document.getElementById('ollama-options');
const ollamaModel = document.getElementById('ollama-model');

// Sections
const uploadSection = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const successSection = document.getElementById('success-section');
const errorSection = document.getElementById('error-section');

// Progress elements
const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');

// Download buttons
const downloadVideoBtn = document.getElementById('download-video-btn');
const downloadSrtBtn = document.getElementById('download-srt-btn');
const downloadSrtTxtBtn = document.getElementById('download-srt-txt-btn');
const downloadTranscriptionBtn = document.getElementById('download-transcription-btn');
const downloadTranscriptionTxtBtn = document.getElementById('download-transcription-txt-btn');
const newVideoBtn = document.getElementById('new-video-btn');
const retryBtn = document.getElementById('retry-btn');

// Stats elements
const elapsedTimeEl = document.getElementById('elapsed-time');
const inputTokensEl = document.getElementById('input-tokens');
const outputTokensEl = document.getElementById('output-tokens');
const totalCostEl = document.getElementById('total-cost');

// ============================================
// Event Listeners
// ============================================

// Drag and Drop
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
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
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

// Toggle buttons
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

// Process button
processBtn.addEventListener('click', startProcessing);

// Download buttons
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

// New video / Retry buttons
newVideoBtn.addEventListener('click', resetToUpload);
retryBtn.addEventListener('click', resetToUpload);

// ============================================
// Functions
// ============================================

function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['.mp4', '.mov', '.avi', '.mkv'];
    const extension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(extension)) {
        alert('Please select a valid video file (MP4, MOV, AVI, or MKV)');
        return;
    }

    selectedFile = file;

    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    dropZone.style.display = 'none';
    filePreview.style.display = 'block';
    processBtn.disabled = false;
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';

    dropZone.style.display = 'block';
    filePreview.style.display = 'none';
    processBtn.disabled = true;
}

function setActiveToggle(container, activeBtn) {
    container.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    activeBtn.classList.add('active');
}

function getSelectedLanguage() {
    const activeBtn = languageToggle.querySelector('.toggle-btn.active');
    return activeBtn ? activeBtn.dataset.value : 'English';
}

function getSelectedService() {
    const activeBtn = serviceToggle.querySelector('.toggle-btn.active');
    return activeBtn ? activeBtn.dataset.value : 'openai';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

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

async function startProcessing() {
    if (!selectedFile) return;

    // Show progress section
    showSection('progress');
    updateProgress(5, 'Uploading video...');
    updateSteps('upload');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('video', selectedFile);
        formData.append('language', getSelectedLanguage());
        formData.append('translation_service', getSelectedService());
        formData.append('ollama_model', ollamaModel.value);

        // Upload
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
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

                // Update stats display
                if (data.elapsed_time) {
                    elapsedTimeEl.textContent = data.elapsed_time;
                }
                if (data.token_usage) {
                    inputTokensEl.textContent = data.token_usage.prompt_tokens.toLocaleString();
                    outputTokensEl.textContent = data.token_usage.completion_tokens.toLocaleString();
                    if (data.token_usage.total_cost !== undefined) {
                        totalCostEl.textContent = '$' + data.token_usage.total_cost.toFixed(4);
                    }
                }

                showSection('success');
            } else if (data.status === 'error') {
                clearInterval(statusInterval);
                showError(data.error || 'An error occurred during processing');
            }

        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 3000);
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

function updateSteps(currentStep) {
    const steps = ['upload', 'extract', 'transcribe', 'translate', 'embed'];
    const currentIndex = steps.indexOf(currentStep);

    steps.forEach((step, index) => {
        const stepEl = document.getElementById(`step-${step}`);
        stepEl.classList.remove('active', 'completed');

        if (index < currentIndex) {
            stepEl.classList.add('completed');
        } else if (index === currentIndex) {
            stepEl.classList.add('active');
        }
    });
}

function showSection(section) {
    uploadSection.style.display = 'none';
    progressSection.style.display = 'none';
    successSection.style.display = 'none';
    errorSection.style.display = 'none';

    switch (section) {
        case 'upload':
            uploadSection.style.display = 'block';
            break;
        case 'progress':
            progressSection.style.display = 'block';
            break;
        case 'success':
            successSection.style.display = 'block';
            break;
        case 'error':
            errorSection.style.display = 'block';
            break;
    }
}

function showError(message) {
    document.getElementById('error-message').textContent = message;
    showSection('error');
}

function resetToUpload() {
    if (statusInterval) {
        clearInterval(statusInterval);
    }

    currentJobId = null;
    clearFile();
    showSection('upload');

    // Reset progress
    progressBar.style.width = '0%';
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('SubtitleAI initialized');
});
