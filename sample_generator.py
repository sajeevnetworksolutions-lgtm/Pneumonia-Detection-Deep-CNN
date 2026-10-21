"""
Synthetic Clinical Radiograph Generator & Preloader
Synthesizes anatomically realistic pediatric chest radiographs (AP projection)
featuring thoracic anatomy (rib cage, spine, cardiac silhouette, diaphragms)
with distinct clinical pathologies: Normal, Lobar Pneumonia, and Viral Infiltrates.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def generate_thoracic_radiograph(
    case_type: str = "normal",
    width: int = 512,
    height: int = 512,
    seed: int = 42
) -> np.ndarray:
    """
    Renders an AP pediatric chest radiograph with realistic anatomical structures:
    - Thoracic cage boundary
    - Bilateral radiolucent lung fields
    - Cardiac & mediastinal shadow
    - Posterior/Anterior rib shadows
    - Thoracic vertebral column
    - Diaphragmatic contours and costophrenic angles
    - Specific focal or diffuse opacifications (if pneumonia)
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width), dtype=np.float32)

    # 1. Background soft-tissue thoracic density
    y_coords, x_coords = np.ogrid[:height, :width]
    center_x, center_y = width / 2.0, height / 2.0
    
    # Body contour ellipse
    body_mask = (((x_coords - center_x) / (width * 0.44))**2 + ((y_coords - center_y) / (height * 0.48))**2) <= 1.0
    img[body_mask] = 140.0 + rng.normal(0, 4, size=np.sum(body_mask))

    # 2. Bilateral Radiolucent Lung Fields (Air is dark on X-rays)
    # Left Lung
    left_lung_mask = (
        ((x_coords - (center_x - width * 0.22)) / (width * 0.16))**2 +
        ((y_coords - (center_y - height * 0.05)) / (height * 0.32))**2
    ) <= 1.0
    
    # Right Lung
    right_lung_mask = (
        ((x_coords - (center_x + width * 0.22)) / (width * 0.16))**2 +
        ((y_coords - (center_y - height * 0.05)) / (height * 0.32))**2
    ) <= 1.0

    lung_base_intensity = 35.0  # Radiolucent (darker)
    img[left_lung_mask] = lung_base_intensity + rng.normal(0, 5, size=np.sum(left_lung_mask))
    img[right_lung_mask] = lung_base_intensity + rng.normal(0, 5, size=np.sum(right_lung_mask))

    # 3. Mediastinum and Cardiac Silhouette (dense tissue is radiopaque / whiter)
    # Cardiac apex extends towards patient's left (viewer's right)
    cardiac_mask = (
        ((x_coords - (center_x + width * 0.06)) / (width * 0.15))**2 +
        ((y_coords - (center_y + height * 0.12)) / (height * 0.18))**2
    ) <= 1.0
    img[cardiac_mask] = 185.0 + rng.normal(0, 4, size=np.sum(cardiac_mask))

    # Central spine column
    spine_mask = (np.abs(x_coords - center_x) < (width * 0.04)) & (y_coords < height * 0.85)
    img[spine_mask] = np.maximum(img[spine_mask], 160.0 + rng.normal(0, 3, size=np.sum(spine_mask)))

    # 4. Diaphragmatic Domes (sharp curved borders at bottom of lungs)
    for side, cx in [(-1, center_x - width * 0.22), (1, center_x + width * 0.22)]:
        dome = (
            ((x_coords - cx) / (width * 0.20))**2 +
            ((y_coords - (center_y + height * 0.30)) / (height * 0.12))**2
        ) <= 1.0
        img[dome] = np.maximum(img[dome], 175.0)

    # 5. Rib Cage Shadows (Horizontal curved bone opacities)
    for rib_i in range(7):
        rib_y = int(center_y - height * 0.25 + rib_i * (height * 0.08))
        rib_curve = (
            np.abs(y_coords - (rib_y + 0.0003 * (x_coords - center_x)**2)) < 6
        ) & (body_mask)
        img[rib_curve] = np.clip(img[rib_curve] + 45.0, 0, 255)

    # 6. Pulmonary Vascular Markings (fine dendritic branching)
    for _ in range(35):
        start_x = int(center_x + rng.uniform(-width * 0.05, width * 0.05))
        start_y = int(center_y + rng.uniform(-height * 0.1, height * 0.1))
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(width * 0.08, width * 0.22)
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))
        cv2.line(img, (start_x, start_y), (end_x, end_y), color=float(rng.uniform(70, 110)), thickness=1)

    # 7. Pathological Infiltration Rendering (per case)
    if case_type == "lobar_pneumonia":
        # Right lower/middle lobe dense alveolar consolidation
        # High opacity, fluffy cotton-wool margins, visible air-bronchograms
        cv2.circle(img, (int(center_x - width * 0.20), int(center_y + height * 0.14)), int(width * 0.13), color=195.0, thickness=-1)
        cv2.circle(img, (int(center_x - width * 0.16), int(center_y + height * 0.06)), int(width * 0.09), color=175.0, thickness=-1)
        
    elif case_type == "viral_pneumonia":
        # Diffuse bilateral peribronchial thickening and interstitial infiltrates
        for _ in range(80):
            ix = int(center_x + rng.choice([-1, 1]) * rng.uniform(width * 0.10, width * 0.26))
            iy = int(center_y + rng.uniform(-height * 0.18, height * 0.22))
            rad = int(rng.uniform(10, 26))
            cv2.circle(img, (ix, iy), rad, color=float(rng.uniform(115, 160)), thickness=-1)

    elif case_type == "early_infiltrate":
        # Subtle, small focal opacity in left mid-lung
        cv2.circle(img, (int(center_x + width * 0.19), int(center_y - height * 0.02)), int(width * 0.07), color=145.0, thickness=-1)

    # 8. Post-processing Gaussian smoothing and radiographic noise
    img = cv2.GaussianBlur(img, (7, 7), 2.0)
    noise = rng.normal(0, 3.5, size=(height, width))
    final_img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return final_img


def ensure_sample_radiographs(sample_dir: str):
    """Generates standard curated sample radiographs for instant UI selection."""
    os.makedirs(sample_dir, exist_ok=True)
    
    samples_manifest = [
        {
            "id": "CASE-1049-NORMAL",
            "filename": "pediatric_normal_case_01.png",
            "type": "normal",
            "title": "Case #1049: Clear Pediatric CXR",
            "description": "Standard AP radiograph with sharp costophrenic margins and clear bilateral aeration.",
            "ground_truth": "Normal",
            "seed": 101
        },
        {
            "id": "CASE-2831-NORMAL",
            "filename": "pediatric_normal_case_02.png",
            "type": "normal",
            "title": "Case #2831: Normal CXR (Minor Posture Skew)",
            "description": "Pediatric emergency positioning with slight lateral tilt; lungs fully clear.",
            "ground_truth": "Normal",
            "seed": 202
        },
        {
            "id": "CASE-4412-LOBAR",
            "filename": "bacterial_lobar_pneumonia.png",
            "type": "lobar_pneumonia",
            "title": "Case #4412: Acute Bacterial Lobar Pneumonia",
            "description": "Dense confluent consolidation in the right middle/lower zone with alveolar opacification.",
            "ground_truth": "Pneumonia",
            "seed": 303
        },
        {
            "id": "CASE-5109-VIRAL",
            "filename": "viral_interstitial_pneumonia.png",
            "type": "viral_pneumonia",
            "title": "Case #5109: Viral Interstitial Pneumonia",
            "description": "Bilateral diffuse bronchovascular cuffing and reticulonodular haziness.",
            "ground_truth": "Pneumonia",
            "seed": 404
        },
        {
            "id": "CASE-6390-EARLY",
            "filename": "early_focal_infiltrate.png",
            "type": "early_infiltrate",
            "title": "Case #6390: Early Focal Sub-segmental Infiltrate",
            "description": "Subtle left mid-zone opacity requiring high-sensitivity algorithmic detection.",
            "ground_truth": "Pneumonia",
            "seed": 505
        }
    ]

    for item in samples_manifest:
        file_path = os.path.join(sample_dir, item["filename"])
        if not os.path.exists(file_path):
            img_arr = generate_thoracic_radiograph(case_type=item["type"], seed=item["seed"])
            Image.fromarray(img_arr).save(file_path)

    return samples_manifest
