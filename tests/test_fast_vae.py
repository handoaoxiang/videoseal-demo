"""The optional lightweight VAE must keep geometry and report its actual model."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from webapp import engine, server


class FastVaeTest(unittest.TestCase):
    def test_missing_selected_model_is_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(server, "RUNS", Path(temp)), patch.dict(server.JOBS, {}, clear=True), patch.dict(server.CANCEL, {}, clear=True), patch.object(engine, "vae_ready", side_effect=lambda mode="auto": mode != "fast"), patch.object(server, "POOL", ThreadPoolExecutor(max_workers=1)):
            with TestClient(server.app) as client:
                self.assertFalse(client.get("/api/capabilities").json()["fast_vae"])
                response = client.post("/api/jobs", data={"config": '{"methods":["vae"],"vae_mode":"fast"}', "sample": "true"})
                self.assertEqual(response.status_code, 400)
                self.assertIn("prepare_vae.py --tiny", response.json()["detail"])

    def test_cancellation_between_encoder_and_decoder(self):
        import torch
        cancel = threading.Event()
        model = SimpleNamespace(encode=Mock(side_effect=lambda x: (cancel.set(), SimpleNamespace(latents=x))[1]), decode=Mock())
        with self.assertRaises(engine.Cancelled):
            engine._vae_reconstruct(torch.zeros(1, 3, 8, 8), model, cancel)
        model.decode.assert_not_called()

    def test_cpu_fast_model_keeps_odd_dimensions_and_finite_pixels(self):
        if not engine.vae_ready("fast"):
            self.skipTest("optional TAESD model not installed")
        model = engine.load_vae("cpu", "fast")
        image = np.random.default_rng(12).integers(0, 256, (25, 33, 3), dtype=np.uint8)
        stats = {}
        result = engine.vae_transform(image, model, 3, "fast", stats)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(result.std()), 1)
        self.assertEqual(stats["model"], "madebyollin/taesd")
        self.assertEqual(stats["precision"], "float32")
        self.assertEqual(stats["mode"], "fast")

    def test_fast_full_hd_all_frames_seven_methods_and_audio(self):
        import torch
        if not torch.cuda.is_available() or not engine.vae_ready("fast"):
            self.skipTest("CUDA and optional TAESD model required")
        with tempfile.TemporaryDirectory(prefix="videoseal-fast-vae-") as temp:
            folder = Path(temp)
            engine.ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=0.3", "-f", "lavfi", "-i",
                           "sine=frequency=440:duration=0.3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", folder / "input.mp4"], threading.Event())
            cfg = server.Settings(seconds=.3, fps=30, max_side=1920, vae_mode="fast", compute_device="cuda", steps=1, export_mode="full").model_dump()
            report = engine.run_pipeline(folder, cfg, threading.Event(), lambda **fields: None)
            self.assertEqual(report["processing_resolution"], [1080, 1920])
            self.assertEqual(report["sampling"]["selected_percent"], 100)
            self.assertEqual(report["sampling"]["total_frames"], 9)
            self.assertEqual(report["sampling"]["selected_frames"], 9)
            self.assertEqual(set(report["timings"]["methods"]), set(server.DEFAULT_METHODS))
            self.assertEqual(report["acceleration"]["vae"]["model"], "madebyollin/taesd")
            self.assertEqual(report["metrics"]["watermark_verdict"], "unverified")
            self.assertTrue(report["export"]["audio"])
            self.assertAlmostEqual(report["export"]["output_seconds"], .3, delta=.06)
            print("Fast VAE: 1080x1920 / 30 FPS, all 9 frames, seven methods, audio and actual backend verified.", flush=True)


if __name__ == "__main__":
    unittest.main()
