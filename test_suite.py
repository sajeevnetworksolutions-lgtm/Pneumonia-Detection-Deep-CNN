"""
Comprehensive Test Suite for Pneumonia CAD System
Verifies model architecture, data pipeline, Grad-CAM, metrics, report generation,
multi-dataset ingestion, and hyperparameter benchmarking.
"""

import os
import unittest
import numpy as np
import torch

from model_torch import PneumoniaCNN, initialize_model
from data_pipeline import (
    rgb_to_clinical_grayscale,
    preprocess_radiograph,
    apply_affine_augmentation
)
from explainability import GradCAM, overlay_heatmap
from metrics_engine import ClinicalMetricsEngine
from report_generator import build_clinical_report_html
from sample_generator import generate_thoracic_radiograph
from dataset_manager import get_all_dataset_status, count_local_dataset_files, get_dataset_case_list
from parameter_experimenter import get_benchmark_experiments
from cross_dataset_evaluator import get_cross_dataset_benchmark


class TestPneumoniaCAD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = initialize_model(device=cls.device)
        weights_path = os.path.join(os.path.dirname(__file__), "saved_models", "pneumonia_cnn_calibrated.pth")
        if os.path.exists(weights_path):
            cls.model.load_state_dict(torch.load(weights_path, map_location=cls.device))
        cls.model.eval()

    def test_01_model_architecture_and_output_shape(self):
        """Verifies 4-tier ConvNet produces expected output tensor shape and parameters."""
        dummy_input = torch.randn(2, 1, 150, 150).to(self.device)
        out = self.model(dummy_input)
        self.assertEqual(out.shape, (2, 1))
        # Ensure sigmoid outputs are bounded [0, 1]
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

        summary = self.model.get_parameter_summary()
        self.assertGreater(summary["total_parameters"], 2_500_000)
        self.assertLess(summary["total_parameters"], 3_500_000)

    def test_02_preprocessing_pipeline(self):
        """Verifies luminance grayscale conversion, resizing to 150x150, and [0, 1] normalization."""
        dummy_rgb = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        res = preprocess_radiograph(dummy_rgb, target_size=(150, 150))
        self.assertEqual(res["tensor"].shape, (1, 1, 150, 150))
        self.assertAlmostEqual(float(res["normalized_np"].min()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(res["normalized_np"].max()), 1.0, delta=0.05)

    def test_03_affine_augmentation(self):
        """Verifies affine transformations preserve dimensions and intensity bounds."""
        base_img = np.random.rand(150, 150).astype(np.float32)
        aug_img = apply_affine_augmentation(base_img, seed=42)
        self.assertEqual(aug_img.shape, (150, 150))
        self.assertTrue((aug_img >= 0.0).all() and (aug_img <= 1.0).all())

    def test_04_gradcam_explainability(self):
        """Verifies Grad-CAM extracts valid normalized heatmaps from Conv Block 4."""
        cam_engine = GradCAM(self.model)
        dummy_input = torch.randn(1, 1, 150, 150, requires_grad=True).to(self.device)
        heatmap = cam_engine.generate_heatmap(dummy_input)
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertTrue((heatmap >= 0.0).all() and (heatmap <= 1.0).all())

        base_vis = np.zeros((256, 256), dtype=np.uint8)
        overlay = overlay_heatmap(base_vis, heatmap, colormap="turbo", alpha=0.5)
        self.assertEqual(overlay["blended_rgb"].shape, (256, 256, 3))
        self.assertIn("peak_location", overlay)

    def test_05_clinical_metrics_and_roc(self):
        """Verifies sensitivity, specificity, and ROC-AUC computation."""
        engine = ClinicalMetricsEngine()
        m50 = engine.compute_metrics_at_threshold(0.50)
        self.assertGreater(m50["accuracy"], 90.0)
        self.assertGreater(m50["sensitivity"], 90.0)
        self.assertGreater(m50["specificity"], 90.0)

        # Sliding threshold down should maintain or improve sensitivity
        m30 = engine.compute_metrics_at_threshold(0.30)
        self.assertGreaterEqual(m30["sensitivity"], m50["sensitivity"])

        roc_data = engine.get_roc_curve_data()
        self.assertGreater(roc_data["auc"], 0.90)
        self.assertEqual(len(roc_data["fpr"]), 50)

    def test_06_synthetic_radiograph_generator(self):
        """Verifies procedural thoracic radiograph generation."""
        img_norm = generate_thoracic_radiograph("normal", width=256, height=256, seed=1)
        self.assertEqual(img_norm.shape, (256, 256))
        self.assertEqual(img_norm.dtype, np.uint8)

        img_pneu = generate_thoracic_radiograph("lobar_pneumonia", width=256, height=256, seed=2)
        self.assertEqual(img_pneu.shape, (256, 256))

    def test_07_clinical_report_generation(self):
        """Verifies clinical diagnostic report HTML generation."""
        html = build_clinical_report_html(
            patient_id="TEST-001",
            accession_no="ACC-TEST",
            prediction_label="Pneumonia",
            probability=0.95,
            threshold=0.50,
            triage_tier="CRITICAL",
            sensitivity=99.5,
            specificity=98.8,
            original_b64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            gradcam_b64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        self.assertIn("AURA-PNEUMO CAD DIAGNOSTIC WORKSTATION", html)
        self.assertIn("TEST-001", html)
        self.assertIn("ACC-TEST", html)

    def test_08_dataset_manager(self):
        """Verifies multi-dataset management and indexing."""
        datasets = get_all_dataset_status()
        self.assertGreaterEqual(len(datasets), 4)
        ds_keys = [d["key"] for d in datasets]
        self.assertIn("kermany_pediatric", ds_keys)
        self.assertIn("nih_adult", ds_keys)
        self.assertIn("covid19_viral", ds_keys)
        self.assertIn("rsna_pneumonia", ds_keys)

        kermany_cases = get_dataset_case_list("kermany_pediatric")
        self.assertGreater(len(kermany_cases), 0)

    def test_09_parameter_experimenter(self):
        """Verifies hyperparameter benchmarking matrix and ablation results."""
        exp = get_benchmark_experiments()
        self.assertIn("optimizers", exp)
        self.assertIn("learning_rates", exp)
        self.assertIn("ablations", exp)
        self.assertIn("batch_sizes", exp)
        self.assertIn("threshold_grid", exp)
        
        # Verify RMSprop is highest performing or proposed
        rmsprop = next(o for o in exp["optimizers"] if o["id"] == "opt_rmsprop")
        self.assertGreater(rmsprop["val_accuracy"], 94.0)

    def test_10_cross_dataset_evaluator(self):
        """Verifies multi-dataset benchmark across Kaggle, NIH, COVID-19, and RSNA cohorts."""
        cross_res = get_cross_dataset_benchmark()
        self.assertIn("top_configuration", cross_res)
        self.assertIn("parameter_runs", cross_res)
        self.assertGreaterEqual(len(cross_res["parameter_runs"]), 4)

        top_cfg = cross_res["top_configuration"]
        self.assertGreater(top_cfg["mean_accuracy"], 93.0)
        self.assertGreater(top_cfg["mean_sensitivity"], 96.0)
        self.assertEqual(top_cfg["overall_rank"], 1)


if __name__ == "__main__":
    unittest.main()
