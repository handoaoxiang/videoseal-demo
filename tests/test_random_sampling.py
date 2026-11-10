"""Sparse processing keeps the timeline, audio and unselected decoded pixels."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from webapp import engine, server


class RandomSamplingTest(unittest.TestCase):
    def test_selection_is_repeatable_exact_and_bounded(self):
        cfg = dict(frame_mode="random", random_frames=10, random_seed=42)
        selected = engine.select_frame_indices(1800, cfg)
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected, engine.select_frame_indices(1800, cfg))
        self.assertNotEqual(selected, engine.select_frame_indices(1800, dict(cfg, random_seed=43)))
        self.assertEqual(engine.select_frame_indices(3, cfg), {0, 1, 2})
        self.assertEqual(engine.select_frame_indices(3, {}), {0, 1, 2})
        self.assertEqual(engine.select_frame_indices(2755, dict(cfg, random_frames=9000)), set(range(2755)))

    def test_random_pipeline_preserves_skipped_pixels_and_temporal_neighbors(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-random-test-") as temp:
            folder = Path(temp)
            cancel = threading.Event()
            engine.ffmpeg(["-y", "-i", server.SAMPLE / "watermarked_sample.mp4", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", folder / "input.mp4"], cancel)
            cfg = server.Settings(methods=["dct", "temporal"], seconds=2, fps=4, max_side=256,
                                  frame_mode="random", random_frames=3, random_seed=5).model_dump()
            observed = []
            original_temporal = engine.temporal

            def record_temporal(image, previous, settings):
                observed.append(None if previous is None else previous.copy())
                return original_temporal(image, previous, settings)

            with patch.object(engine, "dct_transform", wraps=engine.dct_transform) as dct, patch.object(engine, "temporal", side_effect=record_temporal):
                report = engine.run_pipeline(folder, cfg, cancel, lambda **kwargs: None)
            self.assertEqual(dct.call_count, 3)
            selected = [number - 1 for number in report["sampling"]["frame_numbers"]]
            self.assertGreater(selected[0], 0)
            self.assertEqual(report["processed_frames"], 8)
            self.assertEqual(report["sampling"]["selected_frames"], 3)
            self.assertEqual(report["preview_frame_number"], selected[0] + 1)
            self.assertEqual(report["metrics"]["watermark_verdict"], "unverified")
            paths = sorted((folder / "frames").glob("*.png"))
            np.testing.assert_array_equal(np.asarray(Image.open(folder / "before.png")), np.asarray(Image.open(paths[selected[0]])))
            for i, path in enumerate(paths):
                if i not in selected:
                    self.assertEqual(path.read_bytes(), (folder / "processed" / path.name).read_bytes())
            for i, previous in zip(selected, observed):
                if i == 0:
                    self.assertIsNone(previous)
                else:
                    expected = np.asarray(Image.open(paths[i - 1]).convert("RGB"))
                    if i - 1 in selected:
                        expected = engine.dct_transform(expected, cfg)
                    np.testing.assert_array_equal(previous, expected)
            metadata = engine.probe(folder / "result.mp4")
            self.assertTrue(metadata["audio"])
            self.assertAlmostEqual(metadata["duration"], 2, delta=.15)
            decoded = engine.extract(folder / "result.mp4", folder / "check-output", cfg, cancel)
            self.assertEqual(len(decoded), 8)

    def test_all_and_random_modes_keep_more_than_2400_output_frames(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-random-long-test-") as temp:
            folder = Path(temp)
            cancel = threading.Event()
            engine.ffmpeg(["-y", "-f", "lavfi", "-i", "color=c=gray:s=128x128:r=30:d=81", "-c:v", "libx264", "-preset", "ultrafast", folder / "input.mp4"], cancel)
            cfg = server.Settings(methods=["dct"], seconds=81, fps=30, max_side=128,
                                  frame_mode="random", random_frames=3).model_dump()
            with patch.object(engine, "dct_transform", wraps=engine.dct_transform) as dct:
                report = engine.run_pipeline(folder, dict(cfg, frame_mode="all"), cancel, lambda **kwargs: None)
            self.assertEqual(dct.call_count, 2430)
            self.assertEqual(report["sampling"]["selected_frames"], 2430)
            self.assertEqual(report["sampling"]["selected_percent"], 100)
            self.assertEqual(len(list((folder / "processed").glob("*.png"))), 2430)
            # Use another directory so the second run cannot reuse processed frames.
            random_folder = folder / "random"
            random_folder.mkdir()
            (random_folder / "input.mp4").write_bytes((folder / "input.mp4").read_bytes())
            with patch.object(engine, "dct_transform", wraps=engine.dct_transform) as dct:
                report = engine.run_pipeline(random_folder, cfg, cancel, lambda **kwargs: None)
            self.assertEqual(dct.call_count, 3)
            self.assertEqual(report["processed_frames"], 2430)
            self.assertEqual(len(list((random_folder / "processed").glob("*.png"))), 2430)
            self.assertAlmostEqual(engine.probe(folder / "result.mp4")["duration"], 81, delta=.15)


if __name__ == "__main__":
    unittest.main()
