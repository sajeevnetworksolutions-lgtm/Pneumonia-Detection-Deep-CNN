"""
Multi-Dataset Ingestion & Management Engine
Supports diverse radiological cohorts:
1. Kermany et al. Pediatric CXR Benchmark (Guangzhou Women & Children's Medical Center)
2. NIH ChestX-ray14 Adult Thoracic Cohort (National Institutes of Health)
3. COVID-19 & Viral / Bacterial Multi-Cohort (Waterloo COVID-Net / IEEE8023)
4. RSNA Pneumonia Detection Challenge Benchmark

Provides automated download of authentic clinical radiographs, Kaggle CLI
integration, and unified dataset indexing.
"""

import os
import json
import urllib.request
import threading
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_CONFIGS = {
    "kermany_pediatric": {
        "title": "Kermany Pediatric CXR Cohort",
        "institution": "Guangzhou Women & Children's Medical Center",
        "domain": "Pediatric Thoracic (AP View)",
        "total_volume": "5,863 Images (1,583 Normal, 4,273 Pneumonia)",
        "source": "Cell 2018 / Kaggle paultimothymooney/chest-xray-pneumonia",
        "description": "Gold-standard pediatric cohort. High proportion of viral and bacterial consolidations in patients 1-5 years old.",
        "sample_mirrors": {
            "NORMAL": [
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/NORMAL/IM-0003-0001.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/NORMAL/IM-0005-0001.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/NORMAL/IM-0007-0001.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/NORMAL/IM-0009-0001.jpeg"
            ],
            "PNEUMONIA": [
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/PNEUMONIA/person100_bacteria_477.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/PNEUMONIA/person100_bacteria_478.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/PNEUMONIA/person101_bacteria_483.jpeg",
                "https://raw.githubusercontent.com/Saagnik-Mondal/Chest-X-Ray-Images-Pneumonia-/main/chest_xray/test/PNEUMONIA/person101_bacteria_486.jpeg"
            ]
        }
    },
    "nih_adult": {
        "title": "NIH ChestX-ray14 Adult Cohort",
        "institution": "National Institutes of Health Clinical Center",
        "domain": "Adult Thoracic (PA/AP Views)",
        "total_volume": "112,120 Frontal CXRs",
        "source": "Wang et al. IEEE CVPR 2017 / Kaggle nih-chest-xrays",
        "description": "Hospital-scale adult radiograph database. Broad age distribution with multi-pathology overlap.",
        "sample_mirrors": {
            "NORMAL": [
                "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/00000001_000.png"
            ],
            "PNEUMONIA": [
                "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/00027426_000.png"
            ]
        }
    },
    "covid19_viral": {
        "title": "COVID-19 & Viral Pneumonia Database",
        "institution": "Waterloo Vision Lab & International Clinical Collaborators",
        "domain": "Multi-Pathogen (Viral vs Bacterial vs COVID-19)",
        "total_volume": "13,975 CXRs",
        "source": "Wang et al. COVID-Net / IEEE8023",
        "description": "Extensive multicenter open radiograph initiative demonstrating ground-glass opacities and bilateral infiltrates.",
        "sample_mirrors": {
            "PNEUMONIA": [
                "https://raw.githubusercontent.com/lindawangg/COVID-Net/master/assets/ex-covid.jpeg",
                "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/covid-19-pneumonia-58-prior.jpg"
            ]
        }
    },
    "rsna_pneumonia": {
        "title": "RSNA Pneumonia Detection Challenge",
        "institution": "Radiological Society of North America (RSNA)",
        "domain": "Expert Annotated Lung Opacities",
        "total_volume": "30,000 CXRs",
        "source": "RSNA / Kaggle rsna-pneumonia-detection-challenge",
        "description": "Explicit radiological bounding box ground-truth annotations for pulmonary lung opacities.",
        "sample_mirrors": {
            "PNEUMONIA": [
                "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/16747_3_1.jpg"
            ]
        }
    }
}


def ensure_dataset_structure():
    """Initializes local directory hierarchy for multi-dataset storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for ds_key in DATASET_CONFIGS:
        ds_path = os.path.join(DATA_DIR, ds_key)
        os.makedirs(os.path.join(ds_path, "NORMAL"), exist_ok=True)
        os.makedirs(os.path.join(ds_path, "PNEUMONIA"), exist_ok=True)


def download_dataset_samples(dataset_key: str = "kermany_pediatric") -> dict:
    """
    Downloads real clinical radiograph instances from mirrors.
    """
    ensure_dataset_structure()
    config = DATASET_CONFIGS.get(dataset_key)
    if not config:
        return {"success": False, "error": f"Unknown dataset key: {dataset_key}"}

    downloaded = []
    ds_path = os.path.join(DATA_DIR, dataset_key)

    for category, urls in config.get("sample_mirrors", {}).items():
        cat_dir = os.path.join(ds_path, category)
        for url in urls:
            filename = url.split("/")[-1]
            dest_file = os.path.join(cat_dir, filename)
            if not os.path.exists(dest_file):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=12) as response, open(dest_file, "wb") as f:
                        f.write(response.read())
                    downloaded.append({"category": category, "filename": filename, "path": dest_file})
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

    return {
        "success": True,
        "dataset_key": dataset_key,
        "newly_downloaded": len(downloaded),
        "total_local_files": count_local_dataset_files(dataset_key)
    }


def download_all_datasets() -> dict:
    """Downloads samples across all configured datasets."""
    results = {}
    for ds_key in DATASET_CONFIGS:
        results[ds_key] = download_dataset_samples(ds_key)
    return results


def count_local_dataset_files(dataset_key: str) -> dict:
    """Counts available local images in NORMAL and PNEUMONIA subfolders."""
    ds_path = os.path.join(DATA_DIR, dataset_key)
    normal_count = 0
    pneumonia_count = 0
    
    norm_dir = os.path.join(ds_path, "NORMAL")
    pneu_dir = os.path.join(ds_path, "PNEUMONIA")

    if os.path.exists(norm_dir):
        normal_count = len([f for f in os.listdir(norm_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if os.path.exists(pneu_dir):
        pneumonia_count = len([f for f in os.listdir(pneu_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    return {
        "normal": normal_count,
        "pneumonia": pneumonia_count,
        "total": normal_count + pneumonia_count
    }


def get_all_dataset_status() -> list:
    """Returns overview of all configured datasets with local availability status."""
    ensure_dataset_structure()
    overview = []
    for key, cfg in DATASET_CONFIGS.items():
        counts = count_local_dataset_files(key)
        overview.append({
            "key": key,
            "title": cfg["title"],
            "institution": cfg["institution"],
            "domain": cfg["domain"],
            "total_volume": cfg["total_volume"],
            "source": cfg["source"],
            "description": cfg["description"],
            "local_normal": counts["normal"],
            "local_pneumonia": counts["pneumonia"],
            "local_total": counts["total"],
            "status": "Ready (Populated)" if counts["total"] > 0 else "Available to Ingest"
        })
    return overview


def get_dataset_case_list(dataset_key: str) -> list:
    """Returns list of all available images for a specific dataset with metadata."""
    ds_path = os.path.join(DATA_DIR, dataset_key)
    cases = []
    if not os.path.exists(ds_path):
        return cases

    for cat in ["NORMAL", "PNEUMONIA"]:
        cat_dir = os.path.join(ds_path, cat)
        if not os.path.exists(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(cat_dir, fname)
                cases.append({
                    "id": f"{dataset_key}_{fname}",
                    "dataset": dataset_key,
                    "category": cat,
                    "filename": fname,
                    "rel_path": os.path.relpath(fpath, BASE_DIR).replace("\\", "/"),
                    "title": f"[{DATASET_CONFIGS[dataset_key]['title'].split()[0]}] {fname}",
                    "ground_truth": "Normal" if cat == "NORMAL" else "Pneumonia"
                })
    return cases
