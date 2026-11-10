"""Independent experimental transforms; no third-party removal code is copied."""
from __future__ import annotations

import importlib.util
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import pywt
from PIL import Image
from scipy.fft import dctn, idctn

from . import runtime

ROOT = Path(__file__).resolve().parents[1]
VAE_PATH = ROOT / "checkpoints" / "sd-vae-ft-mse"
TINY_VAE_PATH = ROOT / "checkpoints" / "taesd"
METHODS = [
    dict(id="dct", name="DCT / 小波处理", subtitle="改变选定频带的系数", group="频域", condition="无需模型"),
    dict(id="carrier", name="参考载波抵消", subtitle="从已对齐的原片 / 水印片估计残差", group="频域", condition="需要两份参考视频"),
    dict(id="spectral", name="频谱约束优化", subtitle="在变化上限内优化频谱差异", group="频域", condition="支持 CUDA 加速"),
    dict(id="denoise", name="去噪与重建", subtitle="非局部均值去噪，抑制细微残差", group="重建", condition="支持 OpenCL GPU 加速"),
    dict(id="vae", name="VAE 重建", subtitle="经过潜空间编码和解码重建画面", group="重建", condition="需要本地 VAE 模型"),
    dict(id="geometry", name="几何同步扰动", subtitle="局部平滑形变与重采样", group="几何", condition="无需模型"),
    dict(id="guided", name="VideoSeal 引导优化", subtitle="降低配套提取器的消息预测幅度", group="模型", condition="仅针对已知 VideoSeal"),
    dict(id="temporal", name="时域运动补偿", subtitle="运动对齐后混合相邻帧", group="时域", condition="无需模型"),
]


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel.is_set():
        raise Cancelled("任务已取消，已生成的文件保留")


def command(args, cancel, timeout=300):
    """Run without a shell and interrupt the owned process on cancellation."""
    check_cancel(cancel)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    import time
    started = time.monotonic()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=.3)
            break
        except subprocess.TimeoutExpired:
            if cancel.is_set() or time.monotonic() - started > timeout:
                process.kill()
                process.communicate()
                check_cancel(cancel)
                raise RuntimeError("媒体处理超时")
    check_cancel(cancel)
    if process.returncode:
        raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1600:] or "媒体处理失败")
    return stdout


def ffmpeg(args, cancel):
    return command([shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *map(str, args)], cancel)


def probe(path, cancel=None):
    data = command([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], cancel or threading.Event(), 30)
    info = json.loads(data)
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not videos:
        raise ValueError("文件中没有视频流")
    stream = videos[0]
    return dict(width=stream["width"], height=stream["height"], duration=float(info.get("format", {}).get("duration") or stream.get("duration") or 0), audio=any(s["codec_type"] == "audio" for s in info["streams"]))


def vae_ready(mode="auto"):
    path = TINY_VAE_PATH if mode == "fast" else VAE_PATH
    return all((path / name).is_file() for name in ("config.json", "diffusion_pytorch_model.safetensors")) and importlib.util.find_spec("diffusers") is not None


def capabilities():
    vae_available = vae_ready() or vae_ready("fast")
    model_ready = (ROOT / "checkpoints" / "y_256b_img.jit").is_file()
    return [dict(m, ready=(vae_available if m["id"] == "vae" else model_ready if m["id"] == "guided" else True)) for m in METHODS]


def frame_limit(cfg):
    return 9000


def select_frame_indices(total, cfg):
    if cfg.get("frame_mode", "all") == "all":
        return set(range(total))
    count = min(total, cfg.get("random_frames", 10))
    return set(random.Random(cfg.get("random_seed", 42)).sample(range(total), count))


def extract(source, folder, cfg, cancel, seconds=None, max_frames=2400):
    folder.mkdir(parents=True, exist_ok=True)
    duration = cfg["seconds"] if seconds is None else seconds
    size = cfg["max_side"]
    scale = f"scale=w='min({size},iw)':h='min({size},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
    ffmpeg(["-i", source, "-t", duration, "-vf", f"fps={cfg['fps']},{scale}", "-frames:v", max_frames,
            "-compression_level", "1", "-threads", "2", folder / "%06d.png"], cancel)
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise ValueError("所选时间段没有可处理的视频帧")
    return paths


def encode(folder, source, dest, cfg, duration, cancel, encoder=None, log=None, full_duration=None):
    # Audio is aligned with the selected source segment and bounded to video length.
    if encoder is None:
        preference = cfg.get("encoder", "auto")
        encoder = "h264_nvenc" if preference != "cpu" and runtime.nvenc_available() else "libx264"
        # NVENC has minimum dimensions; small experiments are cheap on CPU.
        with Image.open(folder / "000001.png") as first:
            if min(first.size) < 256:
                encoder = "libx264"
    def run(codec):
        quality = ["-c:v", codec, "-preset", "p4", "-rc", "vbr", "-cq", cfg["crf"], "-b:v", "0"] if codec == "h264_nvenc" else ["-c:v", "libx264", "-preset", "fast", "-crf", cfg["crf"]]
        video_map = ["-map", "0:v:0"]
        if full_duration is not None and full_duration > duration + .001:
            with Image.open(folder / "000001.png") as first:
                width, height = first.size
            # Keep audio on the original full timeline; concatenate only video.
            filters = (f"[0:v]setpts=PTS-STARTPTS,setsar=1,format=yuv420p[head];"
                       f"[1:v:0]trim=start={duration:.9f},setpts=PTS-STARTPTS,fps={cfg['fps']},"
                       f"scale={width}:{height},setsar=1,format=yuv420p[tail];"
                       "[head][tail]concat=n=2:v=1:a=0[video]")
            video_map = ["-filter_complex", filters, "-map", "[video]"]
        ffmpeg(["-y", "-framerate", cfg["fps"], "-i", folder / "%06d.png", "-i", source,
                *video_map, "-map", "1:a:0?", "-t", full_duration if full_duration is not None else duration,
                *quality, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-map_metadata", "-1", "-movflags", "+faststart", dest], cancel)
    try:
        run(encoder)
    except RuntimeError:
        if encoder != "h264_nvenc":
            raise
        check_cancel(cancel)
        if log:
            log(log="NVENC 本次编码失败，回退 CPU 编码；输出文件保留同设置对照")
        encoder = "libx264"
        run(encoder)
    return encoder


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse == 0 else float(10 * math.log10(255 ** 2 / mse))


def ssim(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    ma, mb = cv2.GaussianBlur(a, (7, 7), 1.5), cv2.GaussianBlur(b, (7, 7), 1.5)
    va = cv2.GaussianBlur(a * a, (7, 7), 1.5) - ma * ma
    vb = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mb * mb
    cov = cv2.GaussianBlur(a * b, (7, 7), 1.5) - ma * mb
    return float(np.mean(((2 * ma * mb + 6.5025) * (2 * cov + 58.5225)) / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225))))


def dct_transform(image, cfg):
    strength = cfg["strength"]
    if cfg["frequency_mode"] == "wavelet":
        out = np.empty_like(image)
        for ch in range(3):
            low, details = pywt.dwt2(image[:, :, ch].astype(float), "haar")
            detail = tuple(pywt.threshold(d, strength * 2, mode="soft") for d in details)
            restored = pywt.idwt2((low, detail), "haar")
            out[:, :, ch] = np.clip(restored[:image.shape[0], :image.shape[1]], 0, 255)
        return out
    h, w = image.shape[:2]
    padded = image.astype(float)
    if h % 8 or w % 8:
        padded = np.pad(padded, ((0, -h % 8), (0, -w % 8), (0, 0)), mode="edge")
    ph, pw = padded.shape[:2]
    blocks = padded.reshape(ph // 8, 8, pw // 8, 8, 3).transpose(0, 2, 4, 1, 3)
    coeff = dctn(blocks, axes=(-2, -1), norm="ortho", overwrite_x=True)
    freq = np.add.outer(np.arange(8), np.arange(8))
    mask = (freq >= 3) & (freq <= 8) if cfg["band"] == "mid" else freq >= 3
    step = max(.1, strength * 4)
    selected = coeff[..., mask]
    selected /= step
    np.rint(selected, out=selected)
    selected *= step
    coeff[..., mask] = selected
    out = idctn(coeff, axes=(-2, -1), norm="ortho", overwrite_x=True).transpose(0, 3, 1, 4, 2).reshape(ph, pw, 3)
    out = out[:h, :w]
    np.rint(out, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


@lru_cache(maxsize=2)
def load_detector(device="cpu"):
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = torch.jit.load(str(ROOT / "checkpoints" / "y_256b_img.jit"), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def optimize(image, cfg, cancel, detector=None):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    h, w = image.shape[:2]
    device = runtime.resolve_device(cfg.get("compute_device", "auto"))
    target = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    ratio = min(1, 256 / max(h, w))
    size = (max(8, round(h * ratio)), max(8, round(w * ratio)))
    base = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
    rng = torch.Generator(device=device).manual_seed(42)
    delta = torch.nn.Parameter(torch.randn(base.shape, generator=rng, device=device) * .0001)
    optimizer = torch.optim.Adam([delta], lr=.002)
    yy = torch.fft.fftfreq(size[0], device=device).view(-1, 1)
    xx = torch.fft.fftfreq(size[1], device=device).view(1, -1)
    radius = (xx.square() + yy.square()).sqrt()
    band = ((radius >= .08) & (radius <= .4)).float() if cfg["band"] == "mid" else (radius >= .08).float()
    spectrum = torch.log1p(torch.fft.fft2(base, norm="ortho").abs()).detach()
    limit = cfg["strength"] * 2 / 255
    for _ in range(cfg["steps"]):
        check_cancel(cancel)
        optimizer.zero_grad()
        changed = (base + delta).clamp(0, 1)
        if detector is not None:
            square = F.interpolate(changed, (256, 256), mode="bilinear", align_corners=False)
            logits = detector.detect(square, is_video=False)[:, -256:]
            objective = logits.square().mean()
        else:
            difference = (torch.log1p(torch.fft.fft2(changed, norm="ortho").abs()) - spectrum).abs()
            objective = -(difference * band).mean()
        # Independent lightweight objective; not a reproduction of UnMarker.
        fidelity = F.mse_loss(changed, base) * 8
        objective = objective + fidelity
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-limit, limit)
    with torch.no_grad():
        full_delta = F.interpolate(delta, (h, w), mode="bilinear", align_corners=False)
        out = (target + full_delta).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.rint(out * 255).astype(np.uint8)


@lru_cache(maxsize=2)
def _pixel_grid(h, w):
    grid = np.mgrid[:h, :w].astype(np.float32)
    grid.setflags(write=False)
    return grid


@lru_cache(maxsize=2)
def _geometry_maps(h, w, strength):
    rng = np.random.default_rng(42)  # Fixed spatial field avoids per-frame random flicker.
    field = rng.normal(size=(8, 8, 2)).astype(np.float32)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    y, x = _pixel_grid(h, w)
    displacement = strength * .25
    map_x, map_y = x + field[:, :, 0] * displacement, y + field[:, :, 1] * displacement
    map_x.setflags(write=False)
    map_y.setflags(write=False)
    return map_x, map_y


def geometry(image, cfg):
    map_x, map_y = _geometry_maps(*image.shape[:2], cfg["strength"])
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def temporal(image, previous, cfg):
    if previous is None:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    if np.mean(cv2.absdiff(gray, prev_gray), dtype=np.float64) > 35:
        return image  # Avoid blending across a scene cut.
    flow = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, .5, 3, 15, 2, 5, 1.2, 0)
    y, x = _pixel_grid(*image.shape[:2])
    aligned = cv2.remap(previous, x + flow[:, :, 0], y + flow[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    weight = min(.45, cfg["strength"] * .05)
    return cv2.addWeighted(image, 1 - weight, aligned, weight, 0)


@lru_cache(maxsize=4)
def load_vae(device="cpu", mode="auto"):
    from diffusers import AutoencoderKL, AutoencoderTiny
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # GTX 16-series uses FP32 here to avoid half-precision VAE NaNs.
    backend, path = (AutoencoderTiny, TINY_VAE_PATH) if mode == "fast" else (AutoencoderKL, VAE_PATH)
    model = backend.from_pretrained(str(path), local_files_only=True, use_safetensors=True).eval().to(device)
    model.enable_tiling()
    return model


def choose_vae_tile(height, width, device, available_bytes=0):
    """Keep headroom for the detector, display, and full-size frame tensors."""
    if device != "cuda":
        return 256
    budget = available_bytes * .75
    full_estimate = height * width * 4096 + 256 * 2**20
    if full_estimate <= budget:
        return 0
    return 512 if budget >= 1200 * 2**20 else 256


def _vae_reconstruct(tensor, model, cancel=None):
    import torch
    with torch.inference_mode():
        encoded = model.encode(tensor)
        latent = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded.latents
        if cancel is not None:
            check_cancel(cancel)
        out = model.decode(latent).sample
    return (out.squeeze().permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1) * 127.5


def vae_transform(image, model, strength, mode="auto", stats=None, log=None, cancel=None):
    import torch
    h, w = image.shape[:2]
    padded = np.pad(image, ((0, -h % 8), (0, -w % 8), (0, 0)), mode="edge")
    tensor = torch.from_numpy(padded.copy()).permute(2, 0, 1).unsqueeze(0).to(device=model.device, dtype=torch.float32) / 127.5 - 1
    state = stats if stats is not None else {}
    device = model.device.type
    if "tile_size" not in state:
        available = 0
        if device == "cuda":
            free, _ = torch.cuda.mem_get_info(model.device)
            available = free + torch.cuda.memory_reserved(model.device) - torch.cuda.memory_allocated(model.device)
        state.update(mode=mode, model="madebyollin/taesd" if mode == "fast" else "stabilityai/sd-vae-ft-mse", precision="float32",
                     tile_size=256 if mode == "legacy" else choose_vae_tile(*padded.shape[:2], device, available), fallbacks=0)
    tile = state["tile_size"]
    candidates = [tile] + ([512, 256] if tile == 0 else [256] if tile == 512 else [])
    for index, tile in enumerate(candidates):
        if cancel is not None:
            check_cancel(cancel)
        if tile:
            model.tile_sample_min_size = tile
            channels = getattr(model.config, "block_out_channels", None) or model.config.decoder_block_out_channels
            model.tile_latent_min_size = tile // (2 ** (len(channels) - 1))
            model.enable_tiling()
        else:
            model.disable_tiling()
        oom = False
        try:
            out = _vae_reconstruct(tensor, model, cancel)
        except torch.cuda.OutOfMemoryError:
            if device != "cuda" or index == len(candidates) - 1:
                raise
            oom = True
        if oom:
            # The exception's inference intermediates are released before clearing the cache.
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            state["fallbacks"] += 1
            if log:
                log(log="VAE 可用显存不足，自动缩小分块后重试当前帧；画面尺寸和 FP32 精度保持不变")
            continue
        if not state.get("reported") or tile != state["tile_size"]:
            if log:
                label = f"{tile} px 分块" if tile else "整帧计算"
                backend_name = "轻量 TAESD" if mode == "fast" else "原模型"
                log(log=f"VAE 实际方式：{backend_name} · {label} · {device.upper()} · FP32")
            state["reported"] = True
        state["tile_size"] = tile
        state["strategy"] = f"tiled_{tile}" if tile else "full_frame"
        break
    if not np.isfinite(out).all():
        raise RuntimeError("VAE 产生非有限数值，已停止以避免保存损坏画面")
    weight = min(1, strength / 5)
    return np.clip(np.rint(image.astype(float) * (1 - weight) + out[:h, :w] * weight), 0, 255).astype(np.uint8)


def denoise_transform(image, strength, preference="auto", stats=None, log=None, cancel=None):
    state = stats if stats is not None else {}
    if cancel is not None:
        check_cancel(cancel)
    if "backend" not in state:
        info = runtime.opencl_info() if preference != "cpu" else dict(available=False, device="CPU")
        if preference == "opencl" and not info["available"]:
            raise ValueError("GPU 去噪不可用，请选择自动或 CPU 去噪")
        state.update(backend="opencl" if info["available"] else "cpu", device=info["device"] if info["available"] else "CPU")
    if state["backend"] == "opencl":
        try:
            # OpenCV's OpenCL execution context is thread-local: enable in the worker too.
            cv2.ocl.setUseOpenCL(True)
            gpu = cv2.ocl.Device_getDefault()
            if not cv2.ocl.useOpenCL() or not gpu.available() or not gpu.type() & cv2.ocl.Device_TYPE_GPU:
                raise RuntimeError("OpenCL GPU 不可用")
            out = cv2.fastNlMeansDenoisingColored(cv2.UMat(image), None, strength * 1.5, strength * 1.5, 7, 21).get()
        except (cv2.error, RuntimeError):
            if preference == "opencl":
                raise RuntimeError("GPU 去噪失败，请改选自动或 CPU 去噪后重试")
            state.update(backend="cpu", device="CPU", fallback=True)
            if log:
                log(log="GPU 去噪本次不可用，回退 CPU；后续帧继续使用 CPU 去噪")
    if state["backend"] == "cpu":
        out = cv2.fastNlMeansDenoisingColored(image, None, strength * 1.5, strength * 1.5, 7, 21)
    if not state.get("reported"):
        if log:
            log(log=f"去噪实际设备：{state['device']} · {'OpenCL GPU' if state['backend'] == 'opencl' else 'CPU'}")
        state["reported"] = True
    if cancel is not None:
        check_cancel(cancel)
    return out


def evaluate(baseline, result, output, message, cancel, device="cpu"):
    # Compare aligned decoded output samples, not only pre-encoding arrays.
    cfg = dict(seconds=10, fps=2, max_side=256)
    a = extract(baseline, output / "eval-before", cfg, cancel, max_frames=16)
    b = extract(result, output / "eval-after", cfg, cancel, max_frames=16)
    count = min(len(a), len(b))
    pairs = [(np.asarray(Image.open(x).convert("RGB")), np.asarray(Image.open(y).convert("RGB"))) for x, y in zip(a[:count], b[:count])]
    values = [psnr(x, y) for x, y in pairs]
    report = dict(sample_frames=count, sample_psnr_db=None if all(v is None for v in values) else round(float(np.mean([v for v in values if v is not None])), 3), sample_ssim=round(float(np.mean([ssim(x, y) for x, y in pairs])), 5), watermark_verdict="unverified", watermark_note="未知平台水印未验证；画质变化不是水印存在或消失的证据。")
    if message:
        import torch
        model = load_detector(device)
        expected = np.array([int(b) for b in message])
        matches = []
        for index in (0, 1):
            check_cancel(cancel)
            array = np.stack([cv2.resize(pair[index], (256, 256)) for pair in pairs])
            # Small batches cap validation VRAM regardless of video duration.
            predictions = []
            for start in range(0, len(array), 2):
                check_cancel(cancel)
                tensor = torch.from_numpy(array[start:start + 2]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255
                with torch.inference_mode():
                    logits = model.detect(tensor, is_video=True)[:, -256:]
                    if logits.ndim > 2:
                        logits = logits.flatten(2).mean(-1)
                    predictions.append(logits.cpu())
            bits = (torch.cat(predictions).mean(0) > 0).int().numpy()
            matches.append(float(np.mean(bits == expected)))
        report.update(videoseal_before=matches[0], videoseal_after=matches[1], complement_aware_after=max(matches[1], 1 - matches[1]), watermark_verdict="known_message_compared", watermark_note="仅对给定 VideoSeal 消息比较恢复率；恢复率下降不等于所有水印已消除。")
    return report


def run_pipeline(folder, cfg, cancel, update):
    started = time.perf_counter()
    device = runtime.resolve_device(cfg.get("compute_device", "auto"))
    timings = {}
    method_times = {m: 0.0 for m in cfg["methods"]}
    source = next(folder.glob("input.*"))
    metadata = probe(source, cancel)
    duration = min(metadata["duration"], cfg["seconds"])
    if duration <= 0:
        raise ValueError("无法读取视频时长")
    limit = frame_limit(cfg)
    if math.ceil(duration * cfg["fps"]) > limit:
        raise ValueError(f"当前设置超过 {limit} 帧，请缩短时长或降低帧率")
    update(progress=3, phase="抽取处理副本", log="开始解码；输入原文件保留")
    paths = extract(source, folder / "frames", cfg, cancel, max_frames=limit)
    timings["decode_seconds"] = round(time.perf_counter() - started, 3)
    actual_duration = len(paths) / cfg["fps"]
    selected = select_frame_indices(len(paths), cfg)
    preview_index = min(selected)
    sampling = dict(mode=cfg.get("frame_mode", "all"), total_frames=len(paths), selected_frames=len(selected),
                    selected_percent=round(len(selected) / len(paths) * 100, 3),
                    seed=cfg.get("random_seed", 42) if cfg.get("frame_mode") == "random" else None,
                    frame_numbers=[i + 1 for i in sorted(selected)] if cfg.get("frame_mode") == "random" else [])
    if sampling["mode"] == "random":
        update(log=f"随机抽帧：{len(paths)} 帧中处理 {len(selected)} 帧（{sampling['selected_percent']}%），种子 {sampling['seed']}；其余帧不执行所选方法，仍按尺寸、帧率和编码设置输出。")
    update(progress=8, phase="准备对照视频", log=f"处理 {len(paths)} 帧，{actual_duration:.2f} 秒")
    clock = time.perf_counter()
    encoder = encode(folder / "frames", source, folder / "before.mp4", cfg, actual_duration, cancel, log=update)
    timings["baseline_encode_seconds"] = round(time.perf_counter() - clock, 3)
    update(log=f"计算设备：{runtime.compute_info()['device'] if device == 'cuda' else 'CPU'}；编码器：{encoder}")
    clock = time.perf_counter()
    update(phase="加载处理模型", log="模型按设备缓存，后续任务复用；保留原尺寸和 FP32 精度")
    detector = load_detector(device) if "guided" in cfg["methods"] else None
    vae = (load_vae(device, "fast") if cfg.get("vae_mode") == "fast" else load_vae(device)) if "vae" in cfg["methods"] else None
    timings["model_load_seconds"] = round(time.perf_counter() - clock, 3)
    template = None
    if "carrier" in cfg["methods"]:
        clean = next(folder.glob("reference-clean.*"))
        marked = next(folder.glob("reference-marked.*"))
        ca, ma = probe(clean, cancel), probe(marked, cancel)
        if (ca["width"], ca["height"]) != (ma["width"], ma["height"]) or abs(ca["duration"] - ma["duration"]) > .15:
            raise ValueError("两份参考视频必须尺寸、时长一致，并按相同内容和时间对齐")
        refs = [extract(p, folder / name, cfg, cancel, seconds=min(duration, 4), max_frames=32) for p, name in ((clean, "ref-clean"), (marked, "ref-marked"))]
        if len(refs[0]) != len(refs[1]):
            raise ValueError("参考视频解码帧数不一致")
        residuals = [np.asarray(Image.open(m), dtype=float) - np.asarray(Image.open(c), dtype=float) for c, m in zip(*refs)]
        template = np.mean(residuals, axis=0)
        if template.shape != np.asarray(Image.open(paths[0])).shape:
            raise ValueError("参考视频处理后的尺寸必须与输入视频一致")
        th, tw = template.shape[:2]
        radius = np.hypot(np.fft.fftfreq(th)[:, None], np.fft.fftfreq(tw)[None, :])
        mask = (radius >= .08) & (radius <= (.25 if cfg["band"] == "mid" else .5))
        spectrum = np.fft.fft2(template, axes=(0, 1))
        template = np.fft.ifft2(spectrum * mask[:, :, None], axes=(0, 1)).real
        update(log="参考残差可能包含压缩差异，不作为水印存在的证明")
    processed = folder / "processed"
    processed.mkdir(exist_ok=True)
    previous = None
    vae_stats, denoise_stats = {}, {}
    selected_done = 0
    for i, path in enumerate(paths):
        check_cancel(cancel)
        if i not in selected:
            # Preserve decoded pixels without spending time on models or PNG compression.
            try:
                os.link(path, processed / path.name)
            except OSError:
                shutil.copyfile(path, processed / path.name)
            if i % 30 == 0 or i == len(paths) - 1:
                update(progress=12 + int(70 * (i + 1) / len(paths)), phase=f"保留画面 {i + 1}/{len(paths)} · 已抽帧处理 {selected_done}/{len(selected)}")
            continue
        frame = np.asarray(Image.open(path).convert("RGB"))
        raw = frame.copy()
        if "temporal" in cfg["methods"] and i > 0 and i - 1 not in selected:
            # Never bridge a long skipped gap with a distant temporal reference.
            previous = np.asarray(Image.open(paths[i - 1]).convert("RGB"))
        for method_index, method in enumerate(cfg["methods"]):
            check_cancel(cancel)
            title = next(m["name"] for m in METHODS if m["id"] == method)
            update(progress=12 + int(70 * (i + method_index / len(cfg["methods"])) / len(paths)), phase=f"画面 {i + 1}/{len(paths)} · {title}")
            clock = time.perf_counter()
            if method == "dct":
                frame = dct_transform(frame, cfg)
            elif method == "carrier":
                delta = np.clip(template * cfg["strength"] / 3, -cfg["strength"] * 2, cfg["strength"] * 2)
                frame = np.clip(frame.astype(float) - delta, 0, 255).astype(np.uint8)
            elif method == "spectral":
                frame = optimize(frame, cfg, cancel)
            elif method == "denoise":
                frame = denoise_transform(frame, cfg["strength"], cfg.get("denoise_device", "auto"), denoise_stats, update, cancel)
            elif method == "vae":
                frame = vae_transform(frame, vae, cfg["strength"], cfg.get("vae_mode", "auto"), vae_stats, update, cancel)
            elif method == "geometry":
                frame = geometry(frame, cfg)
            elif method == "guided":
                frame = optimize(frame, cfg, cancel, detector)
            elif method == "temporal":
                temporal_input = frame.copy()
                frame = temporal(frame, previous, cfg)
                previous = temporal_input
            method_times[method] += time.perf_counter() - clock
        Image.fromarray(frame).save(processed / path.name, compress_level=1)
        if i == preview_index:
            Image.fromarray(raw).save(folder / "before.png")
            Image.fromarray(frame).save(folder / "after.png")
            Image.fromarray(np.clip(np.abs(raw.astype(float) - frame) * 8, 0, 255).astype(np.uint8)).save(folder / "difference.png")
        selected_done += 1
        update(progress=12 + int(70 * (i + 1) / len(paths)), phase=f"处理画面 {i + 1}/{len(paths)} · 已处理 {selected_done}/{len(selected)}")
    full_duration = metadata["duration"] if cfg.get("export_mode", "clip") == "full" else None
    tail_seconds = max(0, (full_duration or actual_duration) - actual_duration)
    update(progress=85, phase="编码完整视频" if full_duration is not None else "编码结果",
           log=(f"处理片段 {actual_duration:.3f} 秒 + 剩余原视频 {tail_seconds:.3f} 秒；后段不执行算法，按相同尺寸、帧率重新编码，整段音频连续保留。" if full_duration is not None else "重新编码视频并保留对应片段的音频"))
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()  # Release unused allocations before NVENC starts.
    clock = time.perf_counter()
    output_encoder = encode(processed, source, folder / "result.mp4", cfg, actual_duration, cancel, encoder=encoder, log=update, full_duration=full_duration)
    if output_encoder != encoder:
        encode(folder / "frames", source, folder / "before.mp4", cfg, actual_duration, cancel, encoder=output_encoder, log=update)
    timings["result_encode_seconds"] = round(time.perf_counter() - clock, 3)
    exported = probe(folder / "result.mp4", cancel)
    update(progress=92, phase="计算画质与验证结果", log="从实际保存的视频重新解码比较")
    clock = time.perf_counter()
    metrics = evaluate(folder / "before.mp4", folder / "result.mp4", folder, cfg.get("message_bits"), cancel, device)
    timings["validation_seconds"] = round(time.perf_counter() - clock, 3)
    timings["methods"] = {key: round(value, 3) for key, value in method_times.items()}
    update(log="方法耗时：" + "；".join(f"{next(m['name'] for m in METHODS if m['id'] == key)} {value:.1f}s" for key, value in method_times.items()))
    report = dict(config=cfg, source=metadata, processed_frames=len(paths), processed_seconds=actual_duration, processing_resolution=list(raw.shape[1::-1]), metrics=metrics, timings=timings,
                  sampling=sampling, preview_frame_number=preview_index + 1,
                  export=dict(mode=cfg.get("export_mode", "clip"), output_seconds=exported["duration"],
                              processed_seconds=min(actual_duration, metadata["duration"]), tail_seconds=round(tail_seconds, 6),
                              tail_reencoded=tail_seconds > .001, width=exported["width"], height=exported["height"], audio=exported["audio"]),
                  acceleration=dict(compute_device=device, device_name=runtime.compute_info()["device"] if device == "cuda" else "CPU", encoder=output_encoder, model_precision="float32", vae=vae_stats or None, denoise=denoise_stats or None, cpu_methods=[m for m in cfg["methods"] if (m == "denoise" and denoise_stats.get("backend") != "opencl") or (m != "denoise" and (device == "cpu" or m not in ("vae", "guided", "spectral")))]),
                  scope="实验性水印鲁棒性处理，不提供未知平台水印的有无结论。", implementation="独立轻量实现，不是 UnMarker 或 reverse-SynthID 的完整复现。")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
