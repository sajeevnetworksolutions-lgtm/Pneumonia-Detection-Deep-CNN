"""
Data Preprocessing & Augmentation Engine
Implements Section 7 of the M.Tech Dissertation Synopsis:
- Luminance-preserving Grayscale Conversion: I_gray = 0.299*R + 0.587*G + 0.114*B
- Bicubic Spatial Rescaling to 150 x 150 x 1
- Floating-point Min-Max Normalization: X_hat = I / 255.0 in [0.0, 1.0]
- Affine Geometric Data Augmentation:
    * Random Rotation: theta in [-20 deg, +20 deg]
    * Random Zoom & Shear: scale in [0.8, 1.2]
    * Width & Height Shift: dx, dy in [-10%, +10%]
    * Horizontal Flip: 50% probability
"""

import io
import cv2
import numpy as np
from PIL import Image
import torch


def rgb_to_clinical_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """
    Computes ITU-R BT.601 standard single-channel luminance array:
    I_gray(x, y) = 0.299 * R + 0.587 * G + 0.114 * B
    """
    if len(image_rgb.shape) == 2:
        return image_rgb.astype(np.float32)
    
    # Formula explicit implementation
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def preprocess_radiograph(image_bytes_or_arr, target_size=(150, 150)) -> dict:
    """
    Standardizes raw heterogeneous radiographs:
    1. Ingests bytes, PIL Image, or numpy array.
    2. Converts to standard grayscale luminance array.
    3. Performs bicubic spatial resizing to 150 x 150.
    4. Normalizes pixel intensities to floating-point range [0.0, 1.0].
    
    Returns:
        dict: {
            'tensor': torch.Tensor of shape (1, 1, 150, 150),
            'normalized_np': np.ndarray (150, 150) in [0, 1],
            'display_np': np.ndarray (H, W) uint8 for visualization,
            'original_shape': tuple
        }
    """
    if isinstance(image_bytes_or_arr, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_bytes_or_arr)).convert("RGB")
        raw_np = np.array(pil_img)
    elif isinstance(image_bytes_or_arr, Image.Image):
        raw_np = np.array(image_bytes_or_arr.convert("RGB"))
    elif isinstance(image_bytes_or_arr, np.ndarray):
        if len(image_bytes_or_arr.shape) == 2:
            raw_np = np.stack([image_bytes_or_arr]*3, axis=-1)
        else:
            raw_np = image_bytes_or_arr
    else:
        raise ValueError(f"Unsupported image input type: {type(image_bytes_or_arr)}")

    orig_h, orig_w = raw_np.shape[:2]

    # Step 1: Grayscale Luminance conversion
    gray_full = rgb_to_clinical_grayscale(raw_np)

    # Step 2: Bicubic Rescaling to target_size (150, 150)
    resized_gray = cv2.resize(gray_full, target_size, interpolation=cv2.INTER_CUBIC)

    # Step 3: Intensity Normalization to [0.0, 1.0]
    normalized_img = np.clip(resized_gray / 255.0, 0.0, 1.0).astype(np.float32)

    # Create PyTorch tensor: (Batch=1, Channels=1, Height=150, Width=150)
    tensor = torch.from_numpy(normalized_img).unsqueeze(0).unsqueeze(0)

    # 8-bit display image for UI canvas
    display_img = np.clip(gray_full, 0, 255).astype(np.uint8)

    return {
        "tensor": tensor,
        "normalized_np": normalized_img,
        "display_np": display_img,
        "original_shape": (orig_h, orig_w)
    }


def apply_affine_augmentation(img_150: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Applies composite affine geometric transformations per Section 7.B:
    - Random Rotation: [-20 deg, +20 deg]
    - Random Zoom & Shear: [0.8, 1.2]
    - Coordinate Translation: [-10%, +10%]
    - Horizontal Flip: 50% probability
    """
    rng = np.random.default_rng(seed)
    h, w = img_150.shape[:2]
    
    # 1. Horizontal Reflection (Bilateral invariance)
    if rng.random() > 0.5:
        img_150 = cv2.flip(img_150, 1)

    # 2. Random Rotation [-20, +20 deg]
    angle = rng.uniform(-20.0, 20.0)
    
    # 3. Random Zoom [0.8, 1.2]
    scale = rng.uniform(0.8, 1.2)
    
    # 4. Translations [-10%, +10%]
    dx = rng.uniform(-0.10 * w, 0.10 * w)
    dy = rng.uniform(-0.10 * h, 0.10 * h)

    # Construct Affine Matrix
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0, 2] += dx
    rot_mat[1, 2] += dy

    # Apply warp with reflection/edge padding
    augmented = cv2.warpAffine(
        img_150,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return np.clip(augmented, 0.0, 1.0)
