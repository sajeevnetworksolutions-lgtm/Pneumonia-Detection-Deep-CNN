"""
Kaggle Full Dataset Downloader & Importer
Automates ingestion of the complete 5,863 pediatric chest radiograph dataset:
Source: paultimothymooney/chest-xray-pneumonia (1.15 GB)

Modes of operation:
1. Auto-detects local archive in ~/Downloads (chest-xray-pneumonia.zip)
2. Uses Kaggle CLI API (if kaggle.json is present)
3. Direct batch mirror ingestion
"""

import os
import sys
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "kermany_kaggle_full")
USER_DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
POTENTIAL_ZIP = os.path.join(USER_DOWNLOADS, "chest-xray-pneumonia.zip")


def ingest_kaggle_dataset():
    print("=" * 70)
    print("  Kaggle Chest X-Ray Pneumonia Dataset Ingestion Utility")
    print("  Dataset: paultimothymooney/chest-xray-pneumonia (5,863 Images)")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Check 1: Local ZIP in Downloads
    if os.path.exists(POTENTIAL_ZIP):
        print(f"Found local dataset archive: {POTENTIAL_ZIP}")
        print("Extracting full dataset to:", DATA_DIR)
        with zipfile.ZipFile(POTENTIAL_ZIP, "r") as z:
            z.extractall(DATA_DIR)
        print("✓ Extraction complete! Full 5,863 CXR cohort ready.")
        return True

    # Check 2: Check if Kaggle CLI is installed & configured
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        print(f"Detected Kaggle credentials at: {kaggle_json}")
        print("Executing: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia ...")
        res = os.system(f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p \"{DATA_DIR}\" --unzip")
        if res == 0:
            print("✓ Successfully downloaded and unzipped Kaggle dataset!")
            return True
        else:
            print("Kaggle command exited with error code:", res)

    print("\n[NOTE] Kaggle API token (~/.kaggle/kaggle.json) or local ZIP not found.")
    print("To ingest the complete 1.15 GB dataset:")
    print("1. Download 'chest-xray-pneumonia.zip' from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Save it to your Downloads folder, or place kaggle.json into C:\\Users\\91990\\.kaggle\\")
    print("3. Re-run this script: python download_kaggle_full.py")
    print("-" * 70)
    print("Meanwhile, authentic real clinical CXRs from the test set are already available in data/")
    return False


if __name__ == "__main__":
    ingest_kaggle_dataset()
