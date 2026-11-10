"""Measure fixed per-method workloads and retain comparable JSON results."""
from pathlib import Path
import argparse
import inspect
import json
import sys
import threading
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp import engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    user_frame = ROOT / "outputs/web/42f2f2ef58d044f39bf98071030a2da6/frames/000001.png"
    source = user_frame if user_frame.is_file() else ROOT / "outputs/20261027-105027/watermarked_frame.png"
    original = np.asarray(Image.open(source).convert("RGB"))
    cfg = dict(strength=5, steps=6, band="mid_high", frequency_mode="dct", compute_device=args.device)
    cancel = threading.Event()
    def sync():
        import torch
        if args.device == "cuda":
            torch.cuda.synchronize()
    def timed(label, operation):
        sync()
        start = time.perf_counter()
        out = operation()
        sync()
        seconds = time.perf_counter() - start
        if isinstance(out, np.ndarray):
            assert out.shape == original.shape and np.isfinite(out).all()
            Image.fromarray(out).save(ROOT / "outputs" / f"benchmark-{args.label}-{label}.png")
        result["timings"][label] = round(seconds, 4)
        print(label, f"{seconds:.3f}s", flush=True)
        return out
    result = dict(label=args.label, device=args.device, source=str(source), shape=list(original.shape), steps=cfg["steps"], timings={})
    if args.device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    loader_args = (args.device,) if "device" in inspect.signature(engine.load_vae).parameters else ()
    vae = timed("load_vae", lambda: engine.load_vae(*loader_args))
    detector_args = (args.device,) if "device" in inspect.signature(engine.load_detector).parameters else ()
    detector = timed("load_detector", lambda: engine.load_detector(*detector_args))
    timed("vae_frame", lambda: engine.vae_transform(original, vae, cfg["strength"]))
    timed("spectral_frame", lambda: engine.optimize(original, cfg, cancel))
    timed("guided_frame", lambda: engine.optimize(original, cfg, cancel, detector))
    # A second frame separates CUDA's first-operation startup from steady work.
    timed("vae_warm_frame", lambda: engine.vae_transform(original, vae, cfg["strength"]))
    timed("spectral_warm_frame", lambda: engine.optimize(original, cfg, cancel))
    timed("guided_warm_frame", lambda: engine.optimize(original, cfg, cancel, detector))
    if args.device == "cuda":
        result["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        result["cpu_gpu_comparison"] = {}
        for name in ("vae_frame", "spectral_frame", "guided_frame"):
            cpu_file = ROOT / "outputs" / f"benchmark-optimized-cpu-{name}.png"
            if cpu_file.is_file():
                a = np.asarray(Image.open(cpu_file))
                b = np.asarray(Image.open(ROOT / "outputs" / f"benchmark-{args.label}-{name}.png"))
                result["cpu_gpu_comparison"][name] = dict(psnr=engine.psnr(a, b), ssim=engine.ssim(a, b))
    output = ROOT / "outputs" / f"benchmark-{args.label}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
