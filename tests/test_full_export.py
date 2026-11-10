"""Full exports append the original tail at the correct cut and keep audio."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from webapp import engine, server


class FullExportTest(unittest.TestCase):
    def test_full_export_keeps_tail_timeline_and_continuous_audio(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-full-export-") as temp:
            folder = Path(temp)
            cancel = threading.Event()
            engine.ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=s=128x256:r=12:d=4", "-f", "lavfi", "-i",
                           "sine=frequency=440:duration=4", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", "-shortest", folder / "input.mp4"], cancel)
            cfg = server.Settings(methods=["dct"], seconds=1, fps=8, max_side=128, crf=12,
                                  export_mode="full", encoder="cpu").model_dump()
            with patch.object(engine, "dct_transform", side_effect=lambda image, settings: np.full_like(image, (255, 0, 0))) as transform:
                report = engine.run_pipeline(folder, cfg, cancel, lambda **kwargs: None)
            self.assertEqual(transform.call_count, 8)
            self.assertEqual(report["processed_frames"], 8)
            self.assertTrue(report["export"]["tail_reencoded"])
            self.assertAlmostEqual(report["export"]["tail_seconds"], 3, delta=.05)
            self.assertAlmostEqual(report["export"]["output_seconds"], 4, delta=.15)
            self.assertTrue(report["export"]["audio"])
            self.assertAlmostEqual(engine.probe(folder / "before.mp4")["duration"], 1, delta=.15)
            full_cfg = dict(cfg, seconds=4)
            actual = engine.extract(folder / "result.mp4", folder / "check-result", full_cfg, cancel)
            original = engine.extract(folder / "input.mp4", folder / "check-original", full_cfg, cancel)
            self.assertEqual(len(actual), 32)
            last_head = np.asarray(Image.open(actual[7])).astype(float)
            self.assertGreater(last_head[:, :, 0].mean(), 245)
            self.assertLess(last_head[:, :, 1:].mean(), 5)
            for index in (8, 16, 31):
                a = np.asarray(Image.open(actual[index])).astype(float)
                b = np.asarray(Image.open(original[index])).astype(float)
                self.assertLess(np.abs(a - b).mean(), 8, f"Tail shifted at frame {index}")
            pcm = engine.ffmpeg(["-i", folder / "result.mp4", "-map", "0:a:0", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"], cancel)
            audio = np.frombuffer(pcm, dtype=np.float32)
            self.assertAlmostEqual(len(audio) / 8000, 4, delta=.15)
            for second in (.1, .9, 1.1, 3.5):
                chunk = audio[int(second * 8000):int((second + .1) * 8000)]
                self.assertGreater(float(np.sqrt(np.mean(chunk ** 2))), .03)

    def test_full_export_without_tail_or_audio(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-full-no-tail-") as temp:
            folder = Path(temp)
            cancel = threading.Event()
            engine.ffmpeg(["-y", "-f", "lavfi", "-i", "color=c=blue:s=128x128:r=4:d=1", "-c:v", "libx264", folder / "input.mp4"], cancel)
            cfg = server.Settings(methods=["dct"], seconds=2, fps=4, max_side=128, export_mode="full").model_dump()
            report = engine.run_pipeline(folder, cfg, cancel, lambda **kwargs: None)
            self.assertFalse(report["export"]["audio"])
            self.assertFalse(report["export"]["tail_reencoded"])
            self.assertEqual(report["export"]["tail_seconds"], 0)
            self.assertAlmostEqual(report["export"]["output_seconds"], 1, delta=.15)


if __name__ == "__main__":
    unittest.main()
