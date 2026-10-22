/**
 * Clinical Evaluation & Academic Performance Charts
 * Renders interactive ROC-AUC curves, 12-epoch convergence history,
 * multi-optimizer radar benchmarks, and dynamic confusion matrix audits.
 */

let rocChartInstance = null;
let convergenceLossChartInstance = null;
let convergenceAccChartInstance = null;
let optimizerRadarChartInstance = null;

// Initialize charts when tab activated
document.querySelector('[data-tab="tabAudit"]').addEventListener("click", () => {
    if (!rocChartInstance) initROCChart();
});

document.querySelector('[data-tab="tabConvergence"]').addEventListener("click", () => {
    if (!convergenceLossChartInstance) initConvergenceCharts();
});

const experimentsTabBtn = document.querySelector('[data-tab="tabExperiments"]');
if (experimentsTabBtn) {
    experimentsTabBtn.addEventListener("click", () => {
        if (!optimizerRadarChartInstance) initOptimizerRadarChart();
        loadExperimentTables();
        loadCrossDatasetTable();
    });
}

async function initROCChart() {
    try {
        const resp = await fetch("/api/roc");
        const data = await resp.json();
        
        document.getElementById("rocAucDisplay").innerText = data.auc.toFixed(4);

        const ctx = document.getElementById("rocChartCanvas").getContext("2d");
        
        const chartPoints = data.fpr.map((fpr, i) => ({
            x: fpr,
            y: data.tpr[i]
        }));

        rocChartInstance = new Chart(ctx, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: `Proposed 4-Tier CNN (AUC = ${data.auc})`,
                        data: chartPoints,
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.15)",
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.2,
                        pointRadius: 2,
                        pointHoverRadius: 6
                    },
                    {
                        label: "Random Chance (AUC = 0.5000)",
                        data: [{x: 0, y: 0}, {x: 1, y: 1}],
                        borderColor: "#64748b",
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: "linear",
                        min: 0,
                        max: 1,
                        title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#94a3b8" }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: { display: true, text: "True Positive Rate (Sensitivity / Recall)", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#94a3b8" }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#f1f5f9" } }
                }
            }
        });
    } catch (e) {
        console.error("Failed to load ROC curve:", e);
    }
}

async function initConvergenceCharts() {
    try {
        const resp = await fetch("/api/convergence");
        const data = await resp.json();

        // 1. Loss Chart
        const ctxLoss = document.getElementById("lossChartCanvas").getContext("2d");
        convergenceLossChartInstance = new Chart(ctxLoss, {
            type: "line",
            data: {
                labels: data.epochs.map(e => `Epoch ${e}`),
                datasets: [
                    {
                        label: "Training BCE Loss",
                        data: data.train_loss,
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.1)",
                        borderWidth: 2,
                        tension: 0.3
                    },
                    {
                        label: "Validation BCE Loss",
                        data: data.val_loss,
                        borderColor: "#f59e0b",
                        backgroundColor: "rgba(245, 158, 11, 0.1)",
                        borderWidth: 2,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: "Binary Cross-Entropy Loss", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#94a3b8" }
                    },
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#94a3b8" }
                    }
                },
                plugins: { legend: { labels: { color: "#f1f5f9" } } }
            }
        });

        // 2. Accuracy Chart
        const ctxAcc = document.getElementById("accChartCanvas").getContext("2d");
        convergenceAccChartInstance = new Chart(ctxAcc, {
            type: "line",
            data: {
                labels: data.epochs.map(e => `Epoch ${e}`),
                datasets: [
                    {
                        label: "Training Accuracy (%)",
                        data: data.train_acc,
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.1)",
                        borderWidth: 2,
                        tension: 0.3
                    },
                    {
                        label: "Validation Accuracy (%)",
                        data: data.val_acc,
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.1)",
                        borderWidth: 2,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: "Accuracy (%)", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#94a3b8" }
                    },
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#94a3b8" }
                    }
                },
                plugins: { legend: { labels: { color: "#f1f5f9" } } }
            }
        });
    } catch (e) {
        console.error("Failed to load convergence history:", e);
    }
}

async function initOptimizerRadarChart() {
    const canvas = document.getElementById("optimizerRadarCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    optimizerRadarChartInstance = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "Loss Efficiency"],
            datasets: [
                {
                    label: "RMSprop (Proposed)",
                    data: [94.8, 97.4, 90.6, 94.6, 96.0, 92.5],
                    borderColor: "#06b6d4",
                    backgroundColor: "rgba(6, 182, 212, 0.25)",
                    borderWidth: 2,
                    pointBackgroundColor: "#06b6d4"
                },
                {
                    label: "Adam",
                    data: [93.9, 95.8, 90.8, 94.7, 95.2, 88.0],
                    borderColor: "#f59e0b",
                    backgroundColor: "rgba(245, 158, 11, 0.2)",
                    borderWidth: 2,
                    pointBackgroundColor: "#f59e0b"
                },
                {
                    label: "SGD + Momentum",
                    data: [89.4, 91.2, 86.4, 91.9, 91.5, 75.0],
                    borderColor: "#a855f7",
                    backgroundColor: "rgba(168, 85, 247, 0.2)",
                    borderWidth: 2,
                    pointBackgroundColor: "#a855f7"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: "rgba(255, 255, 255, 0.1)" },
                    grid: { color: "rgba(255, 255, 255, 0.08)" },
                    pointLabels: { color: "#cbd5e1", font: { size: 10 } },
                    ticks: { display: false, min: 70, max: 100 }
                }
            },
            plugins: {
                legend: { labels: { color: "#f1f5f9" } }
            }
        }
    });
}

async function loadExperimentTables() {
    try {
        const resp = await fetch("/api/experiments");
        const expData = await resp.json();
        
        const tbody = document.getElementById("optimizerTableBody");
        if (!tbody || tbody.children.length > 0) return;

        expData.optimizers.forEach(opt => {
            const tr = document.createElement("tr");
            const isProposed = opt.id === "opt_rmsprop";
            tr.innerHTML = `
                <td><strong style="color: ${isProposed ? 'var(--accent-cyan)' : 'var(--text-main)'};">${opt.name}</strong></td>
                <td><code>${opt.learning_rate}</code></td>
                <td><strong style="color: var(--accent-emerald);">${opt.val_accuracy}%</strong></td>
                <td><strong style="color: ${isProposed ? 'var(--accent-cyan)' : 'var(--text-main)'};">${opt.sensitivity}%</strong></td>
                <td>${opt.specificity}%</td>
                <td>${opt.f1_score}%</td>
                <td>${opt.roc_auc}</td>
                <td>${opt.final_val_loss}</td>
                <td style="font-size: 10px; color: var(--text-dim);">${opt.clinical_verdict}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load experiment tables:", e);
    }
}

async function loadCrossDatasetTable() {
    try {
        const resp = await fetch("/api/cross-dataset-benchmark");
        const data = await resp.json();
        const tbody = document.getElementById("crossDatasetTableBody");
        if (!tbody || tbody.children.length > 0) return;

        data.parameter_runs.forEach(run => {
            const tr = document.createElement("tr");
            const isTop = run.overall_rank === 1;
            tr.innerHTML = `
                <td><strong style="color: ${isTop ? 'var(--accent-cyan)' : 'var(--text-main)'};">${run.name}</strong></td>
                <td><strong>${run.kermany_acc}%</strong> / ${run.kermany_sens}%</td>
                <td><strong>${run.nih_acc}%</strong> / ${run.nih_sens}%</td>
                <td><strong>${run.covid_acc}%</strong> / ${run.covid_sens}%</td>
                <td><strong>${run.rsna_acc}%</strong> / ${run.rsna_sens}%</td>
                <td><span style="font-size: 13px; font-weight: 800; color: ${isTop ? 'var(--accent-emerald)' : 'var(--text-muted)'}; font-family: var(--font-mono);">${run.mean_accuracy}%</span></td>
                <td style="font-size: 10px; color: var(--text-dim);">${run.verdict}</td>
            `;
            if (isTop) {
                tr.style.background = "rgba(6, 182, 212, 0.08)";
            }
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load cross dataset table:", e);
    }
}

// Global hook to update Confusion Matrix dynamically
window.updateConfusionMatrixUI = function(metrics) {
    const cm = metrics.confusion_matrix;
    if (!cm) return;

    const tpElem = document.getElementById("cmTP");
    const tnElem = document.getElementById("cmTN");
    const fpElem = document.getElementById("cmFP");
    const fnElem = document.getElementById("cmFN");

    if (tpElem) tpElem.innerText = cm.tp;
    if (tnElem) tnElem.innerText = cm.tn;
    if (fpElem) fpElem.innerText = cm.fp;
    if (fnElem) fnElem.innerText = cm.fn;

    const sensElem = document.getElementById("auditSens");
    const specElem = document.getElementById("auditSpec");
    const precElem = document.getElementById("auditPrec");
    const f1Elem = document.getElementById("auditF1");
    const accElem = document.getElementById("auditAcc");

    if (sensElem) sensElem.innerText = `${metrics.sensitivity}%`;
    if (specElem) specElem.innerText = `${metrics.specificity}%`;
    if (precElem) precElem.innerText = `${metrics.precision}%`;
    if (f1Elem) f1Elem.innerText = `${metrics.f1_score}%`;
    if (accElem) accElem.innerText = `${metrics.accuracy}%`;
};
