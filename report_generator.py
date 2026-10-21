"""
Clinical Diagnostic Report Generator
Produces structured, printable clinical CAD triage reports
complete with patient demographics, AI confidence intervals,
threshold parameters, and Grad-CAM localization evidence.
"""

import base64
import datetime
from io import BytesIO


def build_clinical_report_html(
    patient_id: str,
    accession_no: str,
    prediction_label: str,
    probability: float,
    threshold: float,
    triage_tier: str,
    sensitivity: float,
    specificity: float,
    original_b64: str,
    gradcam_b64: str,
    notes: str = ""
) -> str:
    """
    Renders an elegant, standardized medical audit report printable directly from the browser.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_positive = (prediction_label.upper() == "PNEUMONIA")
    
    status_color = "#dc2626" if is_positive else "#16a34a"
    status_bg = "#fef2f2" if is_positive else "#f0fdf4"
    status_border = "#f87171" if is_positive else "#86efac"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Clinical CAD Triage Report - {accession_no}</title>
    <style>
        @page {{ size: A4 portrait; margin: 15mm; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background: #ffffff;
            margin: 0;
            padding: 24px;
            font-size: 13px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #0284c7;
            padding-bottom: 14px;
            margin-bottom: 18px;
        }}
        .hospital-brand h1 {{
            margin: 0;
            font-size: 20px;
            color: #0f172a;
            letter-spacing: -0.5px;
        }}
        .hospital-brand p {{
            margin: 3px 0 0;
            color: #64748b;
            font-size: 11px;
        }}
        .doc-meta {{
            text-align: right;
            font-size: 11px;
            color: #64748b;
        }}
        .patient-card {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 18px;
        }}
        .meta-item label {{
            display: block;
            font-size: 10px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 600;
        }}
        .meta-item span {{
            font-weight: 600;
            color: #0f172a;
            font-size: 13px;
        }}
        .findings-box {{
            background: {status_bg};
            border: 1px solid {status_border};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .findings-left h2 {{
            margin: 0 0 4px;
            color: {status_color};
            font-size: 18px;
        }}
        .findings-left p {{
            margin: 0;
            color: #475569;
            font-size: 12px;
        }}
        .findings-badge {{
            background: {status_color};
            color: #ffffff;
            font-weight: 700;
            font-size: 13px;
            padding: 6px 14px;
            border-radius: 6px;
            letter-spacing: 0.5px;
        }}
        .images-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }}
        .img-panel {{
            text-align: center;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            background: #f8fafc;
        }}
        .img-panel img {{
            max-width: 100%;
            height: 250px;
            object-fit: contain;
            border-radius: 4px;
            background: #000000;
        }}
        .img-panel caption {{
            display: block;
            margin-top: 8px;
            font-size: 11px;
            font-weight: 600;
            color: #475569;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 12px;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            text-align: left;
        }}
        .metrics-table th {{
            background: #f1f5f9;
            color: #475569;
            font-weight: 600;
        }}
        .signature-section {{
            margin-top: 25px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            border-top: 1px solid #e2e8f0;
            padding-top: 16px;
        }}
        .sig-box {{
            border-bottom: 1px dashed #cbd5e1;
            padding-bottom: 40px;
        }}
        .sig-box p {{
            margin: 0;
            font-size: 11px;
            color: #64748b;
        }}
        .disclaimer {{
            margin-top: 20px;
            font-size: 10px;
            color: #94a3b8;
            text-align: justify;
        }}
        @media print {{
            .no-print {{ display: none; }}
            body {{ padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="no-print" style="margin-bottom: 15px; text-align: right;">
        <button onclick="window.print()" style="padding: 8px 18px; background: #0284c7; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">Print / Save as PDF</button>
    </div>

    <div class="header">
        <div class="hospital-brand">
            <h1>AURA-PNEUMO CAD DIAGNOSTIC WORKSTATION</h1>
            <p>Department of Radiodiagnosis & Pediatric Thoracic Imaging</p>
        </div>
        <div class="doc-meta">
            <div>Report Timestamp: <strong>{now_str}</strong></div>
            <div>Accession ID: <strong>{accession_no}</strong></div>
            <div>CAD Engine: <strong>4-Tier Regularized CNN (v2.4)</strong></div>
        </div>
    </div>

    <div class="patient-card">
        <div class="meta-item">
            <label>Patient ID</label>
            <span>{patient_id}</span>
        </div>
        <div class="meta-item">
            <label>Exam Type</label>
            <span>Pediatric CXR (AP View)</span>
        </div>
        <div class="meta-item">
            <label>Clinical Triage Tier</label>
            <span style="color: {status_color};">{triage_tier}</span>
        </div>
        <div class="meta-item">
            <label>Decision Threshold</label>
            <span>tau = {threshold:.2f}</span>
        </div>
    </div>

    <div class="findings-box">
        <div class="findings-left">
            <h2>AI CAD INTERPRETATION: {prediction_label.upper()}</h2>
            <p>Consolidated posterior probability: <strong>{probability * 100:.2f}%</strong> (Operating point threshold: {threshold:.2f})</p>
        </div>
        <div class="findings-badge">
            {triage_tier}
        </div>
    </div>

    <div class="images-grid">
        <div class="img-panel">
            <img src="{original_b64}" alt="Original CXR">
            <caption>Fig 1: Standardized Pediatric AP Chest Radiograph</caption>
        </div>
        <div class="img-panel">
            <img src="{gradcam_b64}" alt="Grad-CAM Consolidation Map">
            <caption>Fig 2: Conv Block 4 Grad-CAM Pathological Activation Heatmap</caption>
        </div>
    </div>

    <table class="metrics-table">
        <thead>
            <tr>
                <th>Operational Metric</th>
                <th>Calibrated Benchmark Value</th>
                <th>Diagnostic Significance</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Operational Sensitivity (Recall)</td>
                <td><strong>{sensitivity:.1f}%</strong></td>
                <td>Measures power to detect confirmed positive pneumonia cases and eliminate False Negatives.</td>
            </tr>
            <tr>
                <td>Operational Specificity</td>
                <td><strong>{specificity:.1f}%</strong></td>
                <td>Identifies healthy uninfected lungs to prevent unnecessary antimicrobial treatment.</td>
            </tr>
            <tr>
                <td>Predicted Pathological Confidence</td>
                <td><strong>{probability * 100:.2f}%</strong></td>
                <td>Sigmoidal activation output from penultimate dense classifier head.</td>
            </tr>
        </tbody>
    </table>

    <div class="signature-section">
        <div class="sig-box">
            <p>Attending Pediatric Radiologist Sign-Off:</p>
        </div>
        <div class="sig-box">
            <p>Clinical Correlation & Follow-up Recommendation:</p>
            <div style="font-size: 11px; color: #1e293b; margin-top: 5px;">
                {notes if notes else ("Immediate clinical evaluation recommended." if is_positive else "No acute alveolar consolidation identified.")}
            </div>
        </div>
    </div>

    <div class="disclaimer">
        NOTICE: This Computer-Aided Diagnosis (CAD) report is generated using deep convolutional spatial feature extraction for clinical triage decision support. Final medical diagnosis and treatment prescription must be verified by a licensed radiologist.
    </div>
</body>
</html>
"""
    return html
