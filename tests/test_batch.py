"""Batch queue admission, independent outputs, and retry deduplication."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from webapp import engine, server


class BatchTest(unittest.TestCase):
    def test_batch_exceeds_old_queue_limit_and_preserves_each_result(self):
        with tempfile.TemporaryDirectory(prefix="videoseal-batch-") as temp:
            runs = Path(temp)
            gate = threading.Event()
            pipeline = engine.run_pipeline
            failed_name = "batch-2.mp4"

            def controlled_pipeline(folder, cfg, cancel, update):
                gate.wait(30)
                if server.JOBS[folder.name]["name"] == failed_name:
                    raise ValueError("本文件处理失败，后续任务仍应继续")
                return pipeline(folder, cfg, cancel, update)

            with patch.object(server, "RUNS", runs), patch.dict(server.JOBS, {}, clear=True), patch.dict(server.CANCEL, {}, clear=True), patch.object(server, "POOL", ThreadPoolExecutor(max_workers=1)), patch.object(engine, "run_pipeline", side_effect=controlled_pipeline):
                with TestClient(server.app) as client:
                    try:
                        caps = client.get("/api/capabilities").json()
                        self.assertEqual(caps["default_methods"], server.DEFAULT_METHODS)
                        self.assertNotIn("carrier", server.Settings().methods)
                        self.assertEqual(len(server.Settings().methods), 7)
                        self.assertEqual(caps["max_pending_jobs"], 50)
                        cfg = dict(seconds=.5, fps=2, max_side=128, steps=1, strength=.5, export_mode="full")
                        batch_id = "a" * 32
                        payload = (server.SAMPLE / "watermarked_sample.mp4").read_bytes()
                        ids = []
                        for index in range(6):
                            if index == 3:
                                invalid = client.post("/api/jobs", data={"config": json.dumps(cfg), "batch_id": batch_id, "request_id": "f" * 32}, files={"video": ("bad.mp4", b"invalid")})
                                self.assertEqual(invalid.status_code, 400)
                            request_id = f"{index + 1:032x}"
                            response = client.post("/api/jobs", data={"config": json.dumps(cfg), "batch_id": batch_id, "request_id": request_id}, files={"video": (f"batch-{index}.mp4", payload)})
                            self.assertEqual(response.status_code, 202, response.text)
                            job = response.json()
                            self.assertEqual(job["config"]["methods"], server.DEFAULT_METHODS)
                            self.assertEqual(job["batch_id"], batch_id)
                            ids.append(job["id"])
                        self.assertEqual(len(set(ids)), 6)
                        duplicate = client.post("/api/jobs", data={"config": json.dumps(cfg), "batch_id": batch_id, "request_id": "1".zfill(32)}, files={"video": ("batch-0.mp4", payload)})
                        self.assertEqual(duplicate.status_code, 202)
                        self.assertEqual(duplicate.json()["id"], ids[0])
                        self.assertEqual(len(client.get("/api/jobs").json()), 6)
                        with patch.object(server, "MAX_PENDING", 6):
                            full = client.post("/api/jobs", data={"config": json.dumps(cfg)}, files={"video": ("extra.mp4", payload)})
                            self.assertEqual(full.status_code, 429)
                    finally:
                        gate.set()
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        jobs = [client.get(f"/api/jobs/{job_id}").json() for job_id in ids]
                        if all(j["status"] in ("completed", "failed") for j in jobs):
                            break
                        time.sleep(.1)
                    self.assertEqual([j["status"] for j in jobs], ["completed", "completed", "failed", "completed", "completed", "completed"])
                    for index, job in enumerate(jobs):
                        self.assertTrue((runs / job["id"] / "input.mp4").is_file())
                        if index == 2:
                            continue
                        self.assertEqual(set(job["report"]["timings"]["methods"]), set(server.DEFAULT_METHODS))
                        self.assertEqual(job["report"]["metrics"]["watermark_verdict"], "unverified")
                        download = client.get(f"/api/jobs/{job['id']}/files/result.mp4?download=true")
                        self.assertEqual(download.status_code, 200)
                        self.assertIn(f"batch-{index}_", download.headers["content-disposition"])
                    server.JOBS.clear()
                    server.recover()
                    self.assertEqual(len(server.JOBS), 6)
                    self.assertTrue(all(j["batch_id"] == batch_id for j in server.JOBS.values()))


if __name__ == "__main__":
    unittest.main()
