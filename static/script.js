/**
 * Frontend Interactive Controller for Federated Video Anomaly Detection App.
 * Handles drag & drop, file validation, Direct Video URL submission, API request,
 * progress updates, rendering top-3 predictions, timeline frame thumbnails, and video seeking.
 */

let activeMode = 'upload'; // 'upload' or 'url'
let selectedFile = null;
let currentVideoObjectUrl = null;

// Allowed extensions & size limit (500 MB)
const ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv'];
const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024;

document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    setupFileInput();
});

/**
 * Switch between Upload File tab and Direct Video URL tab.
 */
function switchTab(mode) {
    activeMode = mode;
    hideError();

    const uploadBtn = document.getElementById('tabUploadBtn');
    const urlBtn = document.getElementById('tabUrlBtn');
    const panelUpload = document.getElementById('panelUpload');
    const panelUrl = document.getElementById('panelUrl');

    if (mode === 'upload') {
        uploadBtn.classList.add('active');
        if (urlBtn) urlBtn.classList.remove('active');
        panelUpload.classList.add('active');
        if (panelUrl) panelUrl.classList.remove('active');
    } else {
        if (urlBtn) urlBtn.classList.add('active');
        uploadBtn.classList.remove('active');
        if (panelUrl) panelUrl.classList.add('active');
        panelUpload.classList.remove('active');
    }
}

/**
 * Drag and Drop Setup for Upload Zone
 */
function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });
}

function setupFileInput() {
    const fileInput = document.getElementById('videoFileInput');
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
}

/**
 * Validate and handle selected file
 */
function handleFileSelection(file) {
    hideError();

    // Check extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
        showError(`Invalid file format '.${ext}'. Only .mp4, .avi, .mov, and .mkv videos are supported.`);
        return;
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE_BYTES) {
        showError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds maximum allowed limit of 500 MB.`);
        return;
    }

    selectedFile = file;

    // Show preview file card
    document.getElementById('selectedFileName').textContent = file.name;
    document.getElementById('selectedFileSize').textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    document.getElementById('selectedFileCard').classList.remove('hidden');

    // Create object URL for local video preview
    if (currentVideoObjectUrl) {
        URL.revokeObjectURL(currentVideoObjectUrl);
    }
    currentVideoObjectUrl = URL.createObjectURL(file);
    const videoPlayer = document.getElementById('outputVideoPlayer');
    videoPlayer.src = currentVideoObjectUrl;
    videoPlayer.load();
}

function clearSelectedFile() {
    selectedFile = null;
    document.getElementById('videoFileInput').value = '';
    document.getElementById('selectedFileCard').classList.add('hidden');
    if (currentVideoObjectUrl) {
        URL.revokeObjectURL(currentVideoObjectUrl);
        currentVideoObjectUrl = null;
    }
}

/**
 * Main Inference Submission Handler
 */
async function submitInference() {
    hideError();

    const predictBtn = document.getElementById('predictBtn');
    let bodyData = null;
    let headers = {};

    if (activeMode === 'upload') {
        if (!selectedFile) {
            showError('Please select or drag & drop a video file first.');
            return;
        }
        bodyData = new FormData();
        bodyData.append('video', selectedFile);
    } else {
        const urlInput = document.getElementById('videoUrlInput').value.trim();
        if (!urlInput) {
            showError('Please enter a Video or Cloud Storage URL.');
            return;
        }

        // Basic URL structure validation
        try {
            const parsedUrl = new URL(urlInput);
            if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
                showError('Invalid URL scheme. Must start with http:// or https://');
                return;
            }
        } catch (e) {
            showError('Invalid URL structure. Please enter a valid URL.');
            return;
        }

        bodyData = JSON.stringify({ video_url: urlInput });
        headers['Content-Type'] = 'application/json';
    }

    // UI Loading State
    predictBtn.disabled = true;
    showStatus('Processing Video...', 'Downloading asset and extracting temporal frames', 25);
    document.getElementById('resultsSection').classList.add('hidden');

    try {
        updateProgress('Extracting Frames', 'Sampling video frames at stride 10', 45);
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: headers,
            body: bodyData,
        });

        updateProgress('Running SE-ResNet18 Model', 'Analyzing temporal sequence windows', 80);

        const data = await response.json();

        if (!response.ok || data.status !== 'success') {
            throw new Error(data.message || 'An error occurred during prediction.');
        }

        updateProgress('Finalizing', 'Aggregating predictions and anomaly timeline', 100);

        setTimeout(() => {
            hideStatus();
            renderResults(data);
            predictBtn.disabled = false;
        }, 500);

    } catch (err) {
        hideStatus();
        showError(err.message || 'Failed to communicate with inference server.');
        predictBtn.disabled = false;
    }
}

/**
 * Render Inference Prediction Results
 */
function renderResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const predictedClassTitle = document.getElementById('predictedClassTitle');
    const confidenceValue = document.getElementById('confidenceValue');
    const anomalyStatusBadge = document.getElementById('anomalyStatusBadge');
    const resultHeaderCard = document.getElementById('resultHeaderCard');
    const top3List = document.getElementById('top3List');
    const timelineTrack = document.getElementById('timelineTrack');
    const videoPlayer = document.getElementById('outputVideoPlayer');

    // Load Video Source from Server Response (Works for both File Upload and Direct URL!)
    if (data.video_url) {
        videoPlayer.src = data.video_url;
        videoPlayer.load();
    }

    // Primary Prediction
    predictedClassTitle.textContent = data.prediction;
    confidenceValue.textContent = data.confidence_percentage;

    if (data.is_anomaly) {
        resultHeaderCard.classList.remove('normal-state');
        anomalyStatusBadge.className = 'result-badge';
        anomalyStatusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ANOMALY DETECTED`;
    } else {
        resultHeaderCard.classList.add('normal-state');
        anomalyStatusBadge.className = 'result-badge normal-badge';
        anomalyStatusBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> NORMAL PATTERN`;
    }

    // Render Top-3 Predictions
    top3List.innerHTML = '';
    data.top3.forEach((item) => {
        const pct = (item.confidence * 100).toFixed(1);
        const el = document.createElement('div');
        el.className = 'top3-item';
        el.innerHTML = `
            <div class="top3-item-header">
                <span>${item.class}</span>
                <span>${pct}%</span>
            </div>
            <div class="top3-bar-bg">
                <div class="top3-bar-fill" style="width: ${pct}%"></div>
            </div>
        `;
        top3List.appendChild(el);
    });

    // Render Video Metadata
    if (data.video_metadata) {
        document.getElementById('metaDuration').textContent = `${data.video_metadata.duration_sec}s`;
        document.getElementById('metaSequences').textContent = `${data.video_metadata.num_sequences} sequences`;
        document.getElementById('metaFps').textContent = `${data.video_metadata.fps} fps`;
    }

    // Render Timeline Segments with Frame Thumbnails
    timelineTrack.innerHTML = '';
    if (data.timeline && data.timeline.length > 0) {
        data.timeline.forEach((seg) => {
            const segEl = document.createElement('div');
            const anomalyClass = seg.is_anomaly ? 'is-anomaly' : 'is-normal';
            segEl.className = `timeline-segment ${anomalyClass}`;
            segEl.title = `Click to seek video to ${seg.time_range} (${seg.predicted_class} - ${(seg.confidence * 100).toFixed(1)}%)`;

            const imgHtml = seg.thumbnail 
                ? `<img src="${seg.thumbnail}" class="segment-thumb" alt="Frame Thumbnail">` 
                : `<div class="segment-thumb" style="background:#111;"></div>`;

            segEl.innerHTML = `
                ${imgHtml}
                <div class="segment-time">${seg.time_range}</div>
                <div class="segment-class">${seg.predicted_class}</div>
                <div class="segment-conf">${(seg.confidence * 100).toFixed(0)}%</div>
            `;

            // Click segment to seek video player
            segEl.addEventListener('click', () => {
                videoPlayer.currentTime = seg.start_time;
                videoPlayer.play();
            });

            timelineTrack.appendChild(segEl);
        });
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/* UI Helper Functions */
function showStatus(title, message, progressPct) {
    const statusSection = document.getElementById('statusSection');
    document.getElementById('statusTitle').textContent = title;
    document.getElementById('statusMessage').textContent = message;
    document.getElementById('progressBarFill').style.width = `${progressPct}%`;
    statusSection.classList.remove('hidden');
}

function updateProgress(title, message, progressPct) {
    document.getElementById('statusTitle').textContent = title;
    document.getElementById('statusMessage').textContent = message;
    document.getElementById('progressBarFill').style.width = `${progressPct}%`;
}

function hideStatus() {
    document.getElementById('statusSection').classList.add('hidden');
}

function showError(msg) {
    const errorBanner = document.getElementById('errorBanner');
    document.getElementById('errorMessage').textContent = msg;
    errorBanner.classList.remove('hidden');
}

function hideError() {
    document.getElementById('errorBanner').classList.add('hidden');
}
