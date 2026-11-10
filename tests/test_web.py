"""End-to-end local tests: real MP4, audio, all transforms and durable jobs."""
from pathlib import Path
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from webapp import engine, server


class WorkbenchTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-web-test-") as temp:
            folder = Path(temp)
            with patch.object(server, "RUNS", folder / "runs"):
                server.RUNS.mkdir()
                server.JOBS.clear()
                server.CANCEL.clear()
                source = folder / "audio.mp4"
                engine.ffmpeg(["-y", "-i", server.SAMPLE / "watermarked_sample.mp4", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", source], threading.Event())
                message = json.loads((server.SAMPLE / "message.json").read_text())["reference_bits"]
                with TestClient(server.app) as client:
                    self.assertEqual(client.get("/").status_code, 200)
                    self.assertEqual(client.get("/app.js").status_code, 200)
                    self.assertEqual(len(client.get("/api/capabilities").json()["methods"]), 8)
                    self.assertEqual(client.post("/api/jobs", headers={"Origin": "https://example.com"}).status_code, 403)
                    self.assertEqual(client.post("/api/jobs", data={"config": json.dumps({"methods": ["bogus"]}), "sample": "true"}).status_code, 422)
                    self.assertEqual(client.post("/api/jobs", data={"config": json.dumps({"message_bits": "01"}), "sample": "true"}).status_code, 422)
                    self.assertEqual(client.post("/api/jobs", data={"config": json.dumps({"compute_device": "invalid"}), "sample": "true"}).status_code, 422)
                    for invalid in ({"frame_mode": "invalid"}, {"random_frames": 0}, {"random_seed": -1}, {"export_mode": "invalid"}):
                        self.assertEqual(client.post("/api/jobs", data={"config": json.dumps(invalid), "sample": "true"}).status_code, 422)
                    self.assertEqual(client.post("/api/jobs", data={"config": json.dumps({"methods": ["carrier"]}), "sample": "true"}).status_code, 400)
                    self.assertEqual(client.post("/api/jobs", data={"config": "{}"}, files={"video": ("fake.mp4", b"not a video", "video/mp4")}).status_code, 400)
                    response = client.post("/api/jobs", data={"config": json.dumps({"methods": ["carrier"]}), "sample": "true"}, files={"reference_clean": ("a.mp4", b"same-content"), "reference_marked": ("b.mp4", b"same-content")})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("完全相同", response.text)
                    cfg = dict(methods=[m["id"] for m in engine.METHODS], seconds=2, fps=4, max_side=256, steps=2, strength=2, message_bits=message)
                    with source.open("rb") as video, (server.SAMPLE / "clean_sample.mp4").open("rb") as clean, (server.SAMPLE / "watermarked_sample.mp4").open("rb") as marked:
                        response = client.post("/api/jobs", data={"config": json.dumps(cfg)}, files={"video": ("测试片段.mp4", video, "video/mp4"), "reference_clean": ("clean.mp4", clean, "video/mp4"), "reference_marked": ("marked.mp4", marked, "video/mp4")})
                    self.assertEqual(response.status_code, 202, response.text)
                    job_id = response.json()["id"]
                    print("All 8 methods submitted:", job_id, flush=True)
                    queued = client.post("/api/jobs", data={"config": json.dumps({"methods": ["guided"], "steps": 30}), "sample": "true"})
                    cancel_id = queued.json()["id"]
                    client.post(f"/api/jobs/{cancel_id}/cancel")
                    deadline = time.monotonic() + 240
                    while time.monotonic() < deadline:
                        job = client.get(f"/api/jobs/{job_id}").json()
                        if job["status"] not in ("queued", "running"):
                            break
                        time.sleep(.2)
                    self.assertEqual(job["status"], "completed", job)
                    for name in ("before.mp4", "result.mp4", "before.png", "after.png", "difference.png", "report.json"):
                        self.assertEqual(client.get(f"/api/jobs/{job_id}/files/{name}").status_code, 200, name)
                    self.assertEqual(client.get(f"/api/jobs/{job_id}/files/input.mp4").status_code, 404)
                    report = job["report"]
                    self.assertEqual(report["processed_frames"], 8)
                    self.assertEqual(report["metrics"]["watermark_verdict"], "known_message_compared")
                    self.assertTrue(engine.probe(server.RUNS / job_id / "result.mp4")["audio"])
                    self.assertIn(report["acceleration"]["compute_device"], ("cpu", "cuda"))
                    self.assertIn(report["acceleration"]["encoder"], ("libx264", "h264_nvenc"))
                    self.assertEqual(set(report["timings"]["methods"]), set(cfg["methods"]))
                    self.assertAlmostEqual(engine.probe(server.RUNS / job_id / "result.mp4")["duration"], 2, delta=.15)
                    deadline = time.monotonic() + 10
                    while client.get(f"/api/jobs/{cancel_id}").json()["status"] != "cancelled" and time.monotonic() < deadline:
                        time.sleep(.1)
                    self.assertEqual(client.get(f"/api/jobs/{cancel_id}").json()["status"], "cancelled")
                    self.assertEqual(client.get(f"/api/jobs/{cancel_id}").json()["phase"], "已取消")
                    self.assertTrue((server.RUNS / cancel_id / "input.mp4").exists())
                    print("Audio, report, cancellation and file access verified.", flush=True)
                    # Unknown uploads receive no watermark verdict.
                    unknown_cfg = dict(methods=["dct"], seconds=1, fps=2, max_side=128, frequency_mode="wavelet", frame_mode="random", random_frames=1, export_mode="full")
                    with source.open("rb") as video:
                        unknown = client.post("/api/jobs", data={"config": json.dumps(unknown_cfg)}, files={"video": ("clip.mp4", video, "video/mp4")}).json()
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        unknown = client.get(f"/api/jobs/{unknown['id']}").json()
                        if unknown["status"] not in ("queued", "running"):
                            break
                        time.sleep(.1)
                    self.assertEqual(unknown["status"], "completed", unknown)
                    self.assertEqual(unknown["report"]["metrics"]["watermark_verdict"], "unverified")
                    self.assertEqual(unknown["report"]["sampling"]["selected_frames"], 1)
                    self.assertEqual(unknown["report"]["sampling"]["total_frames"], 2)
                    self.assertEqual(unknown["report"]["export"]["mode"], "full")
                    self.assertAlmostEqual(unknown["report"]["export"]["output_seconds"], 2, delta=.15)
                    persisted = json.loads((server.RUNS / job_id / "job.json").read_text(encoding="utf-8"))
                    self.assertEqual(persisted["status"], "completed")
                    interrupted = dict(persisted, id="restart-case", status="running")
                    (server.RUNS / "restart-case").mkdir()
                    server.save(interrupted)
                    server.recover()
                    self.assertEqual(server.JOBS["restart-case"]["status"], "interrupted")
                    self.assertTrue((server.RUNS / job_id / "result.mp4").exists())
                    print("Unknown-video boundary, wavelet processing and restart recovery verified.", flush=True)


if __name__ == "__main__":
    unittest.main()
