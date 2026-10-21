"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Explainability Engine
Generates interpretable visual heatmaps highlighting pathological opacities,
bronchovascular cuffing, and consolidations from the penultimate Conv Block 4.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Attaches backward and forward hooks to Conv Block 4."""
        target_layer = self.model.block4.conv

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Executes Grad-CAM algorithm:
        1. Forward pass to obtain logit.
        2. Backward pass to obtain gradients at Conv Block 4.
        3. Global average pooling across gradients to compute channel importance weights.
        4. Linear combination of activations and ReLU activation.
        5. Normalization to [0.0, 1.0].
        """
        self.model.zero_grad()
        
        # Forward pass
        input_tensor.requires_grad = True
        
        # Forward features
        feat = self.model.forward_features(input_tensor)
        flat = feat.view(feat.size(0), -1)
        h = self.model.fc1(flat)
        h = self.model.bn_dense(h)
        h = F.relu(h)
        h = self.model.dropout_dense(h)
        logits = self.model.out(h)
        
        # Target score
        score = logits[0, 0]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't capture
            return np.zeros((150, 150), dtype=np.float32)

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Global Average Pooling of gradients: alpha_k = (1/Z) * sum(grad_k)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Only retain positive evidence
        
        cam_np = cam.squeeze().cpu().numpy()
        
        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-7:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_heatmap(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    colormap: str = "turbo",
    alpha: float = 0.55
) -> dict:
    """
    Renders diagnostic heatmaps blended over original radiograph with customizable
    colormaps: 'jet', 'turbo', 'viridis', 'inferno'.
    
    Returns:
        dict: {
            'blended_rgb': np.ndarray (H, W, 3) uint8,
            'heatmap_colored': np.ndarray (H, W, 3) uint8,
            'peak_location': (x, y) coordinates of maximal opacity activation
        }
    """
    h, w = base_image.shape[:2]
    
    # Upsample Grad-CAM heatmap to original image resolution with bicubic filtering
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Convert to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colormap selection
    cmap_dict = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO
    }
    cv2_cmap = cmap_dict.get(colormap.lower(), cv2.COLORMAP_TURBO)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2_cmap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Base image standard 3-channel uint8
    if len(base_image.shape) == 2:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_image.copy()

    # Alpha blending: (1 - alpha) * Base + alpha * Heatmap
    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)

    # Find peak activation point (focal center of suspected consolidation)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)

    return {
        "blended_rgb": blended,
        "heatmap_colored": colored_heatmap_rgb,
        "peak_location": max_loc,
        "peak_intensity": float(max_val)
    }
