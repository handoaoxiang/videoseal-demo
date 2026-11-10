"""Pixel equality against reference outputs captured before CPU optimizations."""
from pathlib import Path
import unittest

import numpy as np

from webapp import engine


class CpuProcessingTest(unittest.TestCase):
    def test_dct_geometry_and_temporal_preserve_reference_pixels(self):
        fixture = Path(__file__).parent / "fixtures" / "cpu_processing_reference.npz"
        with np.load(fixture, allow_pickle=False) as data:
            for index in range(3):
                image = data[f"image_{index}"]
                original = image.copy()
                for strength in (.1, 3, 10):
                    for band in ("mid", "mid_high"):
                        cfg = dict(strength=strength, band=band, frequency_mode="dct")
                        actual = engine.dct_transform(image, cfg)
                        np.testing.assert_array_equal(actual, data[f"dct_{index}_{strength}_{band}"])
                    for _ in range(2):
                        actual = engine.geometry(image, dict(strength=strength))
                        np.testing.assert_array_equal(actual, data[f"geometry_{index}_{strength}"])
                np.testing.assert_array_equal(image, original)
                current, previous = data[f"temporal_image_{index}"], data[f"previous_{index}"]
                np.testing.assert_array_equal(engine.temporal(current, previous, dict(strength=3)), data[f"temporal_{index}"])

    def test_scene_cut_threshold_and_first_frame_still_preserve_pixels(self):
        for shape in ((24, 32, 3), (25, 33, 3), (63, 47, 3)):
            image = np.full(shape, 100, dtype=np.uint8)
            np.testing.assert_array_equal(engine.temporal(image, None, dict(strength=3)), image)
            np.testing.assert_array_equal(engine.temporal(image, image + 36, dict(strength=3)), image)
            # Exactly 35 is below the scene-cut branch; uniform inputs have zero flow.
            expected = np.full_like(image, 105)
            np.testing.assert_array_equal(engine.temporal(image, image + 35, dict(strength=3)), expected)



if __name__ == "__main__":
    unittest.main()
