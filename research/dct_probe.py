"""Measure block-DCT noise against one existing, known VideoSeal sample.

This is a robustness probe, not a detector or remover for unknown watermarks.
Run with the demo's Python environment; it requires no extra dependencies.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import demo


def transform(frames, epsilon):
    """Orthonormal 8x8 DCT, bounded mid/high-band noise, then inverse DCT."""
    count, height, width, channels = frames.shape
    if height % 8 or width % 8:
        raise ValueError("Probe frames must have dimensions divisible by 8")
    positions = np.arange(8, dtype=np.float64)
    basis = np.cos(np.pi * positions[:, None] * (2 * positions + 1) / 16)
    basis[0] *= np.sqrt(1 / 8)
    basis[1:] *= np.sqrt(2 / 8)
    blocks = frames.astype(np.float64).reshape(count, height // 8, 8, width // 8, 8, channels)
    blocks = blocks.transpose(0, 1, 3, 5, 2, 4) - 128
    coefficients = basis @ blocks @ basis.T
    # Hold the random field fixed across strengths for a paired comparison.
    noise = np.random.default_rng(20261028).uniform(-1, 1, coefficients.shape)
    mask = np.add.outer(positions, positions) >= 3
    restored = basis.T @ (coefficients + epsilon * noise * mask) @ basis
    restored = restored.transpose(0, 1, 4, 2, 5, 3).reshape(frames.shape) + 128
    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


def main():
    sample = ROOT / "outputs" / "20261027-105027"
    frames = demo.read_frames(sample / "watermarked_sample.mp4")
    expected = json.loads((sample / "message.json").read_text(encoding="utf-8"))["reference_bits"]
    expected = [int(bit) for bit in expected]
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))
    model = torch.jit.load(str(demo.MODEL), map_location="cpu").eval()
    output = ROOT / "outputs" / datetime.now().strftime("dct-probe-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    if not np.array_equal(transform(frames, 0), frames):
        raise RuntimeError("DCT zero-perturbation round trip failed")

    def match(decoded):
        tensor = torch.from_numpy(decoded).permute(0, 3, 1, 2).float() / 255
        bits = demo.extract(model, tensor)
        if len(bits) != len(expected):
            raise RuntimeError("Unexpected decoded message length")
        return sum(a == b for a, b in zip(bits, expected)) / len(expected)

    records = []
    baseline = match(frames)
    for epsilon in (0, 2, 8, 32, 64):
        changed = transform(frames, epsilon)
        path = output / f"dct-epsilon-{epsilon}.mp4"
        demo.write_frames(changed, path)
        saved = demo.read_frames(path)
        accuracy = match(saved)
        mse = float(np.square(saved.astype(np.float64) - frames).mean())
        record = {
            "epsilon": epsilon,
            "saved_bit_match": accuracy,
            "complement_aware_match": max(accuracy, 1 - accuracy),
            "saved_psnr_vs_input_db": None if mse == 0 else round(float(10 * np.log10(255 ** 2 / mse)), 3),
            "output": str(path),
        }
        records.append(record)
        Image.fromarray(saved[0]).save(output / f"dct-epsilon-{epsilon}.png")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report = {
        "scope": "One known VideoSeal sample only; no claim about unknown platform watermarks or complete removal.",
        "source": str(sample / "watermarked_sample.mp4"),
        "baseline_bit_match": baseline,
        "frames": len(frames),
        "resolution": list(frames.shape[1:3]),
        "mp4_crf": 12,
        "method": "Independent block-DCT uniform noise probe; epsilon is in coefficient units, not a per-pixel bound.",
        "variants": records,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REPORT", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
