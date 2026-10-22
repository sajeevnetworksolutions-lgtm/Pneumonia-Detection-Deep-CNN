/**
 * AURA-PNEUMO CAD v2.4 - Radiologist Workstation Controller
 * Manages DICOM viewport manipulations, inference calls, Grad-CAM overlays,
 * multi-dataset cohorts, parameter experimentation, dynamic threshold audits,
 * and clinical report generation.
 */

const state = {
    currentSampleId: "CASE-4412-LOBAR",
    currentImagePath: null,
    uploadedFile: null,
    selectedOptimizer: "rmsprop",
    selectedDropout: "proposed",
    threshold: 0.50,
    colormap: "turbo",
    alpha: 0.55,
    viewMode: "blended", // 'blended' | 'original' | 'heatmap'
    
    // Viewport image adjustments
    brightness: 0,
    contrast: 100,
    inverted: false,
    zoom: 1.0,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    showReticle: false,

    // Last inference payload
    lastInference: null
};

// DOM Elements
const cxrImage = document.getElementById("cxrImage");
const dicomViewport = document.getElementById("dicomViewport");
const reticleOverlay = document.getElementById("reticleOverlay");
const focalTargetBox = document.getElementById("focalTargetBox");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValueDisplay = document.getElementById("thresholdValueDisplay");
const brightnessSlider = document.getElementById("brightnessSlider");
const contrastSlider = document.getElementById("contrastSlider");
const alphaSlider = document.getElementById("alphaSlider");
const colormapSelect = document.getElementById("colormapSelect");

// Tab Switching
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        const targetPane = document.getElementById(btn.dataset.tab);
        if (targetPane) targetPane.classList.add("active");
    });
});

// Theme Toggle
const themeToggleBtn = document.getElementById("themeToggleBtn");
if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
        document.body.classList.toggle("clinical-bright");
        const isBright = document.body.classList.contains("clinical-bright");
        themeToggleBtn.innerHTML = isBright ? "🌙 Darkroom" : "☀️ Clinical Bright";
    });
}

// Viewport Rendering & Filters
function applyViewportTransform() {
    if (!cxrImage) return;
    
    const filterStr = `brightness(${100 + state.brightness}%) contrast(${state.contrast}%) ${state.inverted ? 'invert(100%)' : ''}`;
    const transformStr = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    
    cxrImage.style.filter = filterStr;
    cxrImage.style.transform = transformStr;
}

// Pan & Drag Handlers
dicomViewport.addEventListener("mousedown", (e) => {
    state.isDragging = true;
    state.dragStartX = e.clientX - state.panX;
    state.dragStartY = e.clientY - state.panY;
});

window.addEventListener("mousemove", (e) => {
    if (!state.isDragging) return;
    state.panX = e.clientX - state.dragStartX;
    state.panY = e.clientY - state.dragStartY;
    applyViewportTransform();
});

window.addEventListener("mouseup", () => {
    state.isDragging = false;
});

// Zoom Wheel
dicomViewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 0.15 : -0.15;
    state.zoom = Math.min(Math.max(state.zoom + zoomFactor, 0.8), 3.5);
    applyViewportTransform();
});

// Reset Viewport
document.getElementById("btnResetView").addEventListener("click", () => {
    state.brightness = 0;
    state.contrast = 100;
    state.inverted = false;
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    
    if (brightnessSlider) brightnessSlider.value = 0;
    if (contrastSlider) contrastSlider.value = 100;
    document.getElementById("btnInvert").classList.remove("active");
    applyViewportTransform();
});

// Invert Toggle
document.getElementById("btnInvert").addEventListener("click", function() {
    state.inverted = !state.inverted;
    this.classList.toggle("active", state.inverted);
    applyViewportTransform();
});

// Grid / Reticle Toggle
document.getElementById("btnReticle").addEventListener("click", function() {
    state.showReticle = !state.showReticle;
    this.classList.toggle("active", state.showReticle);
    reticleOverlay.classList.toggle("active", state.showReticle);
});

// Brightness & Contrast Sliders
if (brightnessSlider) {
    brightnessSlider.addEventListener("input", (e) => {
        state.brightness = parseInt(e.target.value);
        document.getElementById("brightnessVal").innerText = `${state.brightness > 0 ? '+' : ''}${state.brightness}%`;
        applyViewportTransform();
    });
}

if (contrastSlider) {
    contrastSlider.addEventListener("input", (e) => {
        state.contrast = parseInt(e.target.value);
        document.getElementById("contrastVal").innerText = `${state.contrast}%`;
        applyViewportTransform();
    });
}

// Colormap & Alpha
if (colormapSelect) {
    colormapSelect.addEventListener("change", (e) => {
        state.colormap = e.target.value;
        runInference();
    });
}

if (alphaSlider) {
    alphaSlider.addEventListener("input", (e) => {
        state.alpha = parseFloat(e.target.value);
        document.getElementById("alphaVal").innerText = `${Math.round(state.alpha * 100)}%`;
        runInference();
    });
}

// View Mode Selector (Blended / Original / Heatmap)
document.querySelectorAll(".viewmode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".viewmode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.viewMode = btn.dataset.mode;
        updateViewportImage();
    });
});

function updateViewportImage() {
    if (!state.lastInference) return;
    
    if (state.viewMode === "blended") {
        cxrImage.src = state.lastInference.images.blended;
    } else if (state.viewMode === "original") {
        cxrImage.src = state.lastInference.images.original;
    } else if (state.viewMode === "heatmap") {
        cxrImage.src = state.lastInference.images.heatmap;
    }
}

// Decision Threshold Slider
if (thresholdSlider) {
    thresholdSlider.addEventListener("input", (e) => {
        state.threshold = parseFloat(e.target.value);
        thresholdValueDisplay.innerText = state.threshold.toFixed(2);
        runInference();
    });
}

// Run Inference API
async function runInference() {
    const formData = new FormData();
    formData.append("threshold", state.threshold);
    formData.append("colormap", state.colormap);
    formData.append("alpha", state.alpha);
    formData.append("optimizer_choice", state.selectedOptimizer || "rmsprop");

    if (state.uploadedFile) {
        formData.append("file", state.uploadedFile);
    } else if (state.currentImagePath) {
        formData.append("image_path", state.currentImagePath);
    } else {
        formData.append("sample_id", state.currentSampleId);
    }

    try {
        const response = await fetch("/api/diagnose", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Diagnostic inference failed");
        const data = await response.json();
        
        state.lastInference = data;
        updateUIWithResults(data);
    } catch (err) {
        console.error("Inference Error:", err);
    }
}

// Update UI Components
function updateUIWithResults(data) {
    updateViewportImage();

    // Priority Banner
    const banner = document.getElementById("triageBanner");
    const bannerTitle = document.getElementById("triageTitle");
    const bannerDesc = document.getElementById("triageDesc");
    
    banner.className = `triage-banner ${data.triage_color === 'red' ? 'pneumonia' : data.triage_color === 'yellow' ? 'warning' : 'normal'}`;
    bannerTitle.innerText = data.triage_tier;
    bannerDesc.innerText = data.recommendation;

    // Gauge Needle & Reading
    const probPct = Math.round(data.probability * 100);
    document.getElementById("gaugeScore").innerText = `${probPct}%`;
    document.getElementById("probStatusLabel").innerText = data.prediction.toUpperCase();
    
    // SVG Needle rotation: -90 deg (0%) to +90 deg (100%)
    const needleAngle = -90 + (probPct / 100) * 180;
    const needleElem = document.getElementById("gaugeNeedle");
    if (needleElem) {
        needleElem.style.transform = `rotate(${needleAngle}deg)`;
    }

    // Operating metrics chips
    document.getElementById("operatingSens").innerText = `${data.operating_metrics.sensitivity}%`;
    document.getElementById("operatingSpec").innerText = `${data.operating_metrics.specificity}%`;
    document.getElementById("operatingF1").innerText = `${data.operating_metrics.f1_score}%`;
    document.getElementById("clinicalNoteText").innerText = data.operating_metrics.clinical_interpretation;

    // Focal Target Box (Peak Opacity)
    if (data.prediction === "Pneumonia" && data.peak_location) {
        focalTargetBox.style.display = "block";
        focalTargetBox.style.left = `${data.peak_location[0]}px`;
        focalTargetBox.style.top = `${data.peak_location[1]}px`;
    } else {
        focalTargetBox.style.display = "none";
    }

    // Refresh Confusion Matrix if tab active
    if (window.updateConfusionMatrixUI) {
        window.updateConfusionMatrixUI(data.operating_metrics);
    }
}

// Sample Card Clicks
document.querySelectorAll(".sample-card").forEach(card => {
    card.addEventListener("click", () => {
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
        card.classList.add("active");
        state.currentSampleId = card.dataset.id;
        state.currentImagePath = null;
        state.uploadedFile = null;
        document.getElementById("fileNameDisplay").innerText = card.querySelector("h4").innerText;
        runInference();
    });
});

// Drag & Drop File Upload
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");

if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--accent-cyan)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-subtle)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-subtle)";
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    state.uploadedFile = file;
    state.currentSampleId = null;
    state.currentImagePath = null;
    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    document.getElementById("fileNameDisplay").innerText = file.name;
    runInference();
}

// Multi-Dataset Download All Action
const btnDownloadAllDatasets = document.getElementById("btnDownloadAllDatasets");
if (btnDownloadAllDatasets) {
    btnDownloadAllDatasets.addEventListener("click", async () => {
        btnDownloadAllDatasets.innerText = "⏳ Downloading Samples from Mirrors...";
        btnDownloadAllDatasets.disabled = true;
        try {
            const resp = await fetch("/api/datasets/download", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "dataset_key=all"
            });
            const res = await resp.json();
            btnDownloadAllDatasets.innerText = "✓ Sync Complete! All Datasets Updated";
            setTimeout(() => {
                btnDownloadAllDatasets.innerText = "📥 Download & Sync All Dataset Samples";
                btnDownloadAllDatasets.disabled = false;
            }, 3000);
        } catch (e) {
            console.error(e);
            btnDownloadAllDatasets.innerText = "Sync Failed (Check Connection)";
            btnDownloadAllDatasets.disabled = false;
        }
    });
}

// Multi-Dataset Case Browser Action
const datasetCasesBrowser = document.getElementById("datasetCasesBrowser");
const datasetCasesGrid = document.getElementById("datasetCasesGrid");
const datasetCasesTitle = document.getElementById("datasetCasesTitle");
const btnCloseCasesBrowser = document.getElementById("btnCloseCasesBrowser");

document.querySelectorAll(".btn-load-dataset-cases").forEach(btn => {
    btn.addEventListener("click", async () => {
        const dsKey = btn.dataset.key;
        datasetCasesBrowser.style.display = "block";
        datasetCasesGrid.innerHTML = "<p style='font-size: 11px; color: var(--text-dim);'>Loading radiographs...</p>";
        datasetCasesTitle.innerText = `Clinical Radiographs: ${dsKey.toUpperCase()}`;

        try {
            const resp = await fetch(`/api/datasets/${dsKey}/cases`);
            const cases = await resp.json();
            
            if (cases.length === 0) {
                datasetCasesGrid.innerHTML = "<p style='font-size: 11px; color: var(--text-dim);'>No local images found. Click 'Download & Sync All Dataset Samples' above.</p>";
                return;
            }

            datasetCasesGrid.innerHTML = "";
            cases.forEach(c => {
                const card = document.createElement("div");
                card.style.cssText = "background: rgba(255,255,255,0.05); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 6px; cursor: pointer; text-align: center;";
                card.innerHTML = `
                    <img src="/${c.rel_path}" style="width: 100%; height: 90px; object-fit: cover; border-radius: 4px; background: #000;">
                    <div style="font-size: 9px; font-weight: 600; margin-top: 4px; color: ${c.ground_truth === 'Normal' ? 'var(--accent-emerald)' : 'var(--accent-crimson)'};">
                        ${c.ground_truth.toUpperCase()}
                    </div>
                    <div style="font-size: 8px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        ${c.filename}
                    </div>
                `;
                card.addEventListener("click", () => {
                    state.currentImagePath = c.rel_path;
                    state.currentSampleId = null;
                    state.uploadedFile = null;
                    document.querySelectorAll(".sample-card").forEach(sc => sc.classList.remove("active"));
                    document.getElementById("fileNameDisplay").innerText = `[${c.dataset}] ${c.filename}`;
                    
                    // Switch to Diagnostic Cockpit Tab
                    document.querySelector('[data-tab="tabWorkstation"]').click();
                    runInference();
                });
                datasetCasesGrid.appendChild(card);
            });
        } catch (e) {
            datasetCasesGrid.innerHTML = "<p style='font-size: 11px; color: #ef4444;'>Failed to load cases.</p>";
        }
    });
});

if (btnCloseCasesBrowser) {
    btnCloseCasesBrowser.addEventListener("click", () => {
        datasetCasesBrowser.style.display = "none";
    });
}

// Hyperparameter Testing Playground Action
const btnApplyParamTest = document.getElementById("btnApplyParamTest");
if (btnApplyParamTest) {
    btnApplyParamTest.addEventListener("click", () => {
        const optSelect = document.getElementById("paramSelectOptimizer");
        const dropSelect = document.getElementById("paramSelectDropout");
        const thSelect = document.getElementById("paramSelectThreshold");

        state.selectedOptimizer = optSelect.value;
        state.selectedDropout = dropSelect.value;
        state.threshold = parseFloat(thSelect.value);

        if (thresholdSlider) thresholdSlider.value = state.threshold;
        if (thresholdValueDisplay) thresholdValueDisplay.innerText = state.threshold.toFixed(2);

        // Switch to Viewport Tab
        document.querySelector('[data-tab="tabWorkstation"]').click();
        runInference();
    });
}

// Clinical Report Modal & PDF Print
const reportModal = document.getElementById("reportModal");
const reportIframe = document.getElementById("reportIframe");
const btnGenerateReport = document.getElementById("btnGenerateReport");
const btnCloseModal = document.getElementById("btnCloseModal");

if (btnGenerateReport) {
    btnGenerateReport.addEventListener("click", async () => {
        if (!state.lastInference) return;
        
        const inf = state.lastInference;
        const formData = new FormData();
        formData.append("patient_id", `PED-${Math.floor(1000 + Math.random() * 9000)}`);
        formData.append("accession_no", `ACC-2026-${Math.floor(100 + Math.random() * 900)}`);
        formData.append("prediction_label", inf.prediction);
        formData.append("probability", inf.probability);
        formData.append("threshold", inf.threshold);
        formData.append("triage_tier", inf.triage_tier);
        formData.append("sensitivity", inf.operating_metrics.sensitivity);
        formData.append("specificity", inf.operating_metrics.specificity);
        formData.append("original_b64", inf.images.original);
        formData.append("gradcam_b64", inf.images.blended);
        formData.append("notes", inf.recommendation);

        const resp = await fetch("/api/report/html", {
            method: "POST",
            body: formData
        });

        const html = await resp.text();
        reportModal.classList.add("active");
        reportIframe.srcdoc = html;
    });
}

if (btnCloseModal) {
    btnCloseModal.addEventListener("click", () => {
        reportModal.classList.remove("active");
    });
}

// Initial Boot Inference
window.addEventListener("DOMContentLoaded", () => {
    applyViewportTransform();
    runInference();
});
