"""Full-HD portrait input keeps its geometry through all seven default methods."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from webapp import engine, runtime, server


class NativeResolutionTest(unittest.TestCase):
    def test_resolution_validation(self):
        self.assertEqual(server.Settings(max_side=1920).max_side, 1920)
        with self.assertRaises(ValidationError):
            server.Settings(max_side=1921)

    def test_full_hd_portrait_all_default_methods_and_full_export(self):
        if not runtime.compute_info()["cuda"]:
            self.skipTest("Full-HD seven-method hardware check requires CUDA")
        with tempfile.TemporaryDirectory(prefix="videoseal-full-hd-") as temp:
            folder = Path(temp)
            source = folder / "portrait-1080x1920.mp4"
            engine.ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=1", "-f", "lavfi", "-i",
                           "sine=frequency=440:duration=1", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", source], threading.Event())
            runs = folder / "runs"
            runs.mkdir()
            with patch.object(server, "RUNS", runs), patch.dict(server.JOBS, {}, clear=True), patch.dict(server.CANCEL, {}, clear=True), patch.object(server, "POOL", ThreadPoolExecutor(max_workers=1)):
                with TestClient(server.app) as client:
                    self.assertEqual(client.get("/api/capabilities").json()["max_processing_side"], 1920)
                    cfg = dict(seconds=.5, fps=30, max_side=1920, frame_mode="random", random_frames=1,
                               strength=.5, steps=1, compute_device="cuda", export_mode="full")
                    with source.open("rb") as video:
                        response = client.post("/api/jobs", data={"config": json.dumps(cfg), "batch_id": "d" * 32}, files={"video": (source.name, video, "video/mp4")})
                    self.assertEqual(response.status_code, 202, response.text)
                    job_id = response.json()["id"]
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        job = client.get(f"/api/jobs/{job_id}").json()
                        if job["status"] not in ("queued", "running"):
                            break
                        time.sleep(.2)
                    self.assertEqual(job["status"], "completed", job)
                    report = job["report"]
                    self.assertEqual(report["processing_resolution"], [1080, 1920])
                    self.assertEqual(report["sampling"]["selected_frames"], 1)
                    self.assertEqual(report["sampling"]["total_frames"], 15)
                    self.assertEqual(set(report["timings"]["methods"]), set(server.DEFAULT_METHODS))
                    acceleration = report["acceleration"]
                    self.assertEqual(acceleration["vae"]["mode"], "auto")
                    self.assertIn(acceleration["vae"]["tile_size"], (0, 256, 512))
                    self.assertIn(acceleration["denoise"]["backend"], ("cpu", "opencl"))
                    self.assertEqual("denoise" in acceleration["cpu_methods"], acceleration["denoise"]["backend"] == "cpu")
                    self.assertTrue(any("VAE 实际方式" in entry["text"] for entry in job["logs"]))
                    self.assertTrue(any("去噪实际设备" in entry["text"] for entry in job["logs"]))
                    self.assertTrue(report["export"]["tail_reencoded"])
                    self.assertTrue(report["export"]["audio"])
                    result = runs / job_id / "result.mp4"
                    metadata = engine.probe(result)
                    self.assertEqual((metadata["width"], metadata["height"]), (1080, 1920))
                    self.assertAlmostEqual(metadata["duration"], 1, delta=.05)
                    streams = json.loads(engine.command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(result)], threading.Event()))["streams"]
                    video = next(s for s in streams if s["codec_type"] == "video")
                    self.assertEqual(video["r_frame_rate"], "30/1")
                    self.assertEqual(int(video["nb_frames"]), 30)
                    self.assertEqual(client.get(f"/api/jobs/{job_id}/files/result.mp4?download=true").status_code, 200)
                    print("1080x1920 / 30 FPS, seven methods, full tail, audio and download verified.", flush=True)


if __name__ == "__main__":
    unittest.main()
