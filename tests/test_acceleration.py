"""Acceleration must report the real backend and preserve cancellation."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from webapp import engine, runtime


class AccelerationTest(unittest.TestCase):
    def test_explicit_cuda_does_not_silently_run_on_cpu(self):
        with patch.object(runtime, "compute_info", return_value=dict(cuda=False, reason="CUDA unavailable")):
            self.assertEqual(runtime.resolve_device("auto"), "cpu")
            self.assertEqual(runtime.resolve_device("cpu"), "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                runtime.resolve_device("cuda")

    def test_nvenc_failure_uses_reported_cpu_fallback(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-encoder-test-") as temp:
            folder = Path(temp)
            Image.new("RGB", (256, 256)).save(folder / "000001.png")
            cfg = dict(fps=4, crf=18, encoder="auto")
            logs = []
            with patch.object(runtime, "nvenc_available", return_value=True), patch.object(engine, "ffmpeg", side_effect=[RuntimeError("encoder unavailable"), b""]) as command:
                encoder = engine.encode(folder, folder / "input.mp4", folder / "result.mp4", cfg, 1, threading.Event(), log=lambda **kwargs: logs.append(kwargs))
            self.assertEqual(encoder, "libx264")
            self.assertEqual(command.call_count, 2)
            self.assertIn("h264_nvenc", command.call_args_list[0].args[0])
            self.assertIn("libx264", command.call_args_list[1].args[0])
            self.assertTrue(logs)
            with patch.object(engine, "ffmpeg", side_effect=engine.Cancelled("cancelled")) as command:
                with self.assertRaises(engine.Cancelled):
                    engine.encode(folder, folder / "input.mp4", folder / "result.mp4", cfg, 1, threading.Event(), encoder="h264_nvenc")
                self.assertEqual(command.call_count, 1)


if __name__ == "__main__":
    unittest.main()
