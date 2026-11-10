"""Compare the original and optimized transforms on identical local video frames."""
import argparse
import gc
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
import torch

from webapp import engine, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[960, 1280, 1920])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = engine.load_vae("cuda")
    report = dict(source=str(args.input.resolve()), strength=3, precision="float32", device=runtime.compute_info(), opencl=runtime.opencl_info(), results=[])
    for size in args.sizes:
        folder = args.output / str(size)
        cfg = dict(seconds=1, fps=1, max_side=size)
        frame_path = engine.extract(args.input, folder / "source", cfg, threading.Event(), max_frames=1)[0]
        image = np.asarray(Image.open(frame_path).convert("RGB"))
        row = dict(max_side=size, width=image.shape[1], height=image.shape[0], transforms={})
        outputs = {}
        for label in ("vae_legacy", "vae_auto", "denoise_cpu", "denoise_gpu"):
            stats = {}
            def run():
                if label.startswith("vae"):
                    return engine.vae_transform(image, model, 3, "legacy" if label == "vae_legacy" else "auto", stats)
                return engine.denoise_transform(image, 3, "cpu" if label == "denoise_cpu" else "opencl", stats)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run()
            torch.cuda.synchronize()
            warm = time.perf_counter() - start
            times = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = run()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            outputs[label] = result
            row["transforms"][label] = dict(warm_seconds=round(warm, 4), mean_seconds=round(float(np.mean(times)), 4), seconds=[round(t, 4) for t in times], stats=stats, peak_torch_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20), source_psnr_db=engine.psnr(image, result), source_ssim=engine.ssim(image, result))
            Image.fromarray(result).save(folder / f"{label}.png")
            print(json.dumps(dict(size=size, transform=label, **row["transforms"][label]), ensure_ascii=False), flush=True)
        for before, after in (("vae_legacy", "vae_auto"), ("denoise_cpu", "denoise_gpu")):
            a, b = outputs[before], outputs[after]
            row[f"{after}_vs_original_backend"] = dict(psnr_db=engine.psnr(a, b), ssim=engine.ssim(a, b), mean_abs_difference=float(np.abs(a.astype(float) - b).mean()))
        report["results"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
