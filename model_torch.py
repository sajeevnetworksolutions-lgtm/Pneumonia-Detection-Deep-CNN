"""
M.Tech Dissertation Architecture Implementation (PyTorch)
Automated Pneumonia Detection from Chest Radiographs Using Deep
Convolutional Neural Networks with Batch-Normalized Spatial Regularization

Specification:
- Input: (B, 1, 150, 150)
- Conv Block 1: Conv2d(1 -> 32, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) + Dropout2d(0.20) -> (B, 32, 75, 75)
- Conv Block 2: Conv2d(32 -> 64, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) + Dropout2d(0.20) -> (B, 64, 37, 37)
- Conv Block 3: Conv2d(64 -> 128, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) + Dropout2d(0.25) -> (B, 128, 18, 18)
- Conv Block 4: Conv2d(128 -> 256, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) + Dropout2d(0.30) -> (B, 256, 9, 9)
- Flatten: 9 * 9 * 256 = 20,736
- Dense Hidden: Linear(20736 -> 128) + BatchNorm1d + ReLU + Dropout(0.40) -> (B, 128)
- Output: Linear(128 -> 1) + Sigmoid -> (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super(ConvBlock, self).__init__()
        # Conv2D with padding=1 keeps spatial size before max-pooling
        # 150 -> (conv 3x3, pad 1) 150 -> pool 75
        # 75  -> (conv 3x3, pad 1) 75  -> pool 37 (floor(75/2) = 37)
        # 37  -> (conv 3x3, pad 1) 37  -> pool 18 (floor(37/2) = 18)
        # 18  -> (conv 3x3, pad 1) 18  -> pool 9  (floor(18/2) = 9)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        
        # 4-Tier Hierarchical Feature Extractor
        self.block1 = ConvBlock(in_channels=1, out_channels=32, dropout_rate=0.20)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, dropout_rate=0.20)
        self.block3 = ConvBlock(in_channels=64, out_channels=128, dropout_rate=0.25)
        self.block4 = ConvBlock(in_channels=128, out_channels=256, dropout_rate=0.30)
        
        # Dense Classification Head
        self.flatten_dim = 9 * 9 * 256  # 20,736
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.bn_dense = nn.BatchNorm1d(128)
        self.dropout_dense = nn.Dropout(p=0.40)
        self.out = nn.Linear(128, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the high-level spatial feature map from Conv Block 4."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        flat = feat.view(feat.size(0), -1)
        
        h = self.fc1(flat)
        h = self.bn_dense(h)
        h = F.relu(h)
        h = self.dropout_dense(h)
        
        logits = self.out(h)
        probs = torch.sigmoid(logits)
        return probs

    def get_parameter_summary(self):
        """Returns detailed parameter breakdown across all layers."""
        summary = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num
            summary.append({
                "layer": name,
                "shape": list(param.shape),
                "params": num,
                "trainable": param.requires_grad
            })
            
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layer_breakdown": summary
        }


def initialize_model(device: str = "cpu") -> PneumoniaCNN:
    """Instantiates the model on the specified device."""
    model = PneumoniaCNN().to(device)
    model.eval()
    return model
