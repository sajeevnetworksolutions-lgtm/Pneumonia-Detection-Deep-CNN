/**
 * Clinical Evaluation & Academic Performance Charts
 * Renders interactive ROC-AUC curves, 12-epoch convergence history,
 * and dynamic confusion matrix audits.
 */

let rocChartInstance = null;
let convergenceLossChartInstance = null;
let convergenceAccChartInstance = null;

// Initialize charts when tab activated
document.querySelector('[data-tab="tabAudit"]').addEventListener("click", () => {
    if (!rocChartInstance) initROCChart();
});

document.querySelector('[data-tab="tabConvergence"]').addEventListener("click", () => {
    if (!convergenceLossChartInstance) initConvergenceCharts();
});

async function initROCChart() {
    try {
        const resp = await fetch("/api/roc");
        const data = await resp.json();
        
        document.getElementById("rocAucDisplay").innerText = data.auc.toFixed(4);

        const ctx = document.getElementById("rocChartCanvas").getContext("2d");
        
        // Build data points
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
