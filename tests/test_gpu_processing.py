"""Adaptive VAE memory policy and honest GPU-denoise fallbacks."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from webapp import engine, runtime, server


class GpuProcessingTest(unittest.TestCase):
    def test_vae_planner_preserves_memory_headroom(self):
        gb = 2**30
        self.assertEqual(engine.choose_vae_tile(960, 544, "cuda", 4 * gb), 0)
        self.assertEqual(engine.choose_vae_tile(1280, 720, "cuda", 3 * gb), 512)
        self.assertEqual(engine.choose_vae_tile(1920, 1080, "cuda", gb), 256)
        self.assertEqual(engine.choose_vae_tile(1920, 1080, "cpu", 32 * gb), 256)

    def test_cpu_denoise_does_not_probe_or_use_gpu(self):
        image = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        expected = cv2.fastNlMeansDenoisingColored(image, None, 4.5, 4.5, 7, 21)
        stats = {}
        with patch.object(runtime, "opencl_info", side_effect=AssertionError("CPU must not probe OpenCL")):
            actual = engine.denoise_transform(image, 3, "cpu", stats)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(stats["backend"], "cpu")

    def test_opencl_failure_falls_back_once_and_reports_cpu(self):
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        stats, logs = {}, []
        gpu = SimpleNamespace(available=lambda: True, type=lambda: cv2.ocl.Device_TYPE_GPU)
        with patch.object(runtime, "opencl_info", return_value=dict(available=True, device="Test GPU")), patch.object(cv2.ocl, "useOpenCL", return_value=True), patch.object(cv2.ocl, "Device_getDefault", return_value=gpu), patch.object(cv2, "UMat", side_effect=cv2.error("GPU failure")) as upload:
            first = engine.denoise_transform(image, 3, "auto", stats, lambda **fields: logs.append(fields))
            second = engine.denoise_transform(image, 3, "auto", stats)
        self.assertEqual(upload.call_count, 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(stats["backend"], "cpu")
        self.assertTrue(stats["fallback"])
        self.assertTrue(logs)

    def test_forced_gpu_is_not_silently_cpu_and_cancellation_is_preserved(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch.object(runtime, "opencl_info", return_value=dict(available=False, device="CPU")):
            with self.assertRaisesRegex(ValueError, "GPU"):
                engine.denoise_transform(image, 3, "opencl")
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(engine.Cancelled):
            engine.denoise_transform(image, 3, cancel=cancel)

    def test_vae_cuda_oom_retries_smaller_tiles_without_resizing(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for the VAE allocation boundary")
        image = np.full((16, 24, 3), 100, dtype=np.uint8)
        model = SimpleNamespace(device=torch.device("cuda"), config=SimpleNamespace(block_out_channels=[128, 256, 512, 512]), enable_tiling=Mock(), disable_tiling=Mock())
        stats = {}
        with patch.object(engine, "choose_vae_tile", return_value=0), patch.object(engine, "_vae_reconstruct", side_effect=[torch.cuda.OutOfMemoryError("test OOM"), image.astype(float)]):
            result = engine.vae_transform(image, model, 3, stats=stats)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(stats["tile_size"], 512)
        self.assertEqual(stats["fallbacks"], 1)
        self.assertEqual(model.tile_latent_min_size, 64)
        self.assertEqual(result.dtype, np.uint8)

    def test_invalid_acceleration_options_are_rejected(self):
        for cfg in ({"vae_mode": "bad"}, {"denoise_device": "bad"}):
            with self.assertRaises(ValueError):
                server.Settings(**cfg)


if __name__ == "__main__":
    unittest.main()
