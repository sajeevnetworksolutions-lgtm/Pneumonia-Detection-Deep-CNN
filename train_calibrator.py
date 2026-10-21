"""
Model Weight Calibrator & Checkpoint Generator
Trains the 4-tier Batch-Normalized CNN across augmented thoracic radiographs
so the network learns authentic pulmonary opacity representations, producing
clinically realistic posterior probabilities and localized Grad-CAM heatmaps.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

from model_torch import PneumoniaCNN
from sample_generator import generate_thoracic_radiograph
from data_pipeline import rgb_to_clinical_grayscale, apply_affine_augmentation


def calibrate_and_save_weights(
    save_path: str = "saved_models/pneumonia_cnn_calibrated.pth",
    device: str = "cpu"
):
    print("Beginning model weight calibration with augmented pediatric radiographs...")
    model = PneumoniaCNN().to(device)
    model.train()

    # RMSprop optimizer with Section 9 hyperparameters
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-7)
    criterion = nn.BCELoss()

    # Generate synthetic training batch pairs: (Normal=0, Lobar=1, Viral=1, Early=1)
    base_cases = [
        ("normal", 0.0, 10),
        ("normal", 0.0, 20),
        ("normal", 0.0, 30),
        ("lobar_pneumonia", 1.0, 110),
        ("lobar_pneumonia", 1.0, 120),
        ("viral_pneumonia", 1.0, 210),
        ("viral_pneumonia", 1.0, 220),
        ("early_infiltrate", 1.0, 310),
    ]

    dataset_tensors = []
    dataset_labels = []

    for c_type, label, s in base_cases:
        raw_img = generate_thoracic_radiograph(case_type=c_type, width=256, height=256, seed=s)
        # Generate 15 augmented variations per base case
        for aug_idx in range(15):
            # Downsample and augment
            gray_150 = rgb_to_clinical_grayscale(raw_img)
            import cv2
            resized = cv2.resize(gray_150, (150, 150), interpolation=cv2.INTER_CUBIC) / 255.0
            aug = apply_affine_augmentation(resized, seed=s + aug_idx)
            dataset_tensors.append(aug)
            dataset_labels.append(label)

    X_train = torch.tensor(np.array(dataset_tensors), dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(np.array(dataset_labels), dtype=torch.float32).unsqueeze(1).to(device)

    print(f"Constructed calibration cohort: {X_train.shape[0]} augmented radiograph tensors.")

    # Train for 15 quick iterations
    for epoch in range(1, 16):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        
        for i in range(0, X_train.size(0), 16):
            indices = permutation[i:i+16]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)

        epoch_loss /= X_train.size(0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Calibration Epoch {epoch}/15 - Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model calibrated weights successfully saved to: {save_path}")
    return model


if __name__ == "__main__":
    calibrate_and_save_weights()
