"""Long full-frame uploads must pass admission without changing their settings."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from webapp import engine, server


class SubmissionTest(unittest.TestCase):
    def test_long_all_mode_and_legacy_full_selection_admission(self):
        metadata = dict(width=1080, height=1920, duration=91.833333, audio=True)
        cfg = dict(methods=["dct", "spectral", "geometry"], seconds=300, fps=30, max_side=1920, frame_mode="all", export_mode="full")
        pool = Mock()
        with tempfile.TemporaryDirectory() as temp, patch.object(server, "RUNS", Path(temp) / "missing" / "runs"), patch.dict(server.JOBS, {}, clear=True), patch.dict(server.CANCEL, {}, clear=True), patch.object(server, "POOL", pool), patch.object(engine, "probe", return_value=metadata):
            with TestClient(server.app) as client:
                self.assertEqual(client.get("/api/capabilities").json()["max_processing_frames"], 9000)
                response = client.post("/api/jobs", data={"config": json.dumps(cfg)}, files={"video": ("long.mp4", b"probe-is-mocked")})
                self.assertEqual(response.status_code, 202, response.text)
                job = response.json()
                self.assertEqual(job["config"]["frame_mode"], "all")
                self.assertEqual(job["config"]["fps"], 30)
                self.assertEqual(job["config"]["max_side"], 1920)
                self.assertTrue((server.RUNS / job["id"] / "job.json").is_file())
                pool.submit.assert_called_once()
                legacy_limit = lambda settings: 9000 if settings.get("frame_mode") == "random" else 2400
                with patch.object(engine, "frame_limit", side_effect=legacy_limit):
                    rejected = client.post("/api/jobs", data={"config": json.dumps(cfg)}, files={"video": ("too-long-for-old-all.mp4", b"probe-is-mocked")})
                    self.assertEqual(rejected.status_code, 400)
                    self.assertIn("2755", rejected.json()["detail"])
                    compatible = dict(cfg, frame_mode="random", random_frames=9000, random_seed=42)
                    accepted = client.post("/api/jobs", data={"config": json.dumps(compatible)}, files={"video": ("compatible.mp4", b"probe-is-mocked")})
                    self.assertEqual(accepted.status_code, 202, accepted.text)
                    self.assertEqual(engine.select_frame_indices(2755, compatible), set(range(2755)))


if __name__ == "__main__":
    unittest.main()
