"""
AURA-PNEUMO CAD Server Launcher
Launches the clinical CAD workstation on http://127.0.0.1:8000
"""

import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  AURA-PNEUMO CAD v2.4 - Pediatric Thoracic CAD Workstation")
    print("  M.Tech AI/ML Dissertation: 4-Tier Batch-Normalized Regularized CNN")
    print("=" * 70)
    print("Starting high-precision clinical CAD server on http://127.0.0.1:8000 ...")
    print("Open http://127.0.0.1:8000 in your browser to access the workstation.")
    print("=" * 70)

    # Launch Uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
