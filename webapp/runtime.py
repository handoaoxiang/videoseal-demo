"""Probe actual acceleration support once, with explicit CPU fallbacks."""
from functools import lru_cache
import os
import shutil
import subprocess


@lru_cache(maxsize=1)
def compute_info():
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = dict(cuda=False, device="CPU", torch_version=torch.__version__, reason="当前 PyTorch 未启用 CUDA")
    if not torch.cuda.is_available():
        return result
    try:
        check = torch.ones(1, device="cuda")
        assert check.sum().item() == 1
        prop = torch.cuda.get_device_properties(0)
        result.update(cuda=True, device=prop.name, memory_mb=round(prop.total_memory / 2**20), reason="")
    except Exception as exc:
        result["reason"] = f"CUDA 初始化失败：{exc}"
    return result


def resolve_device(preference="auto"):
    if preference == "cpu":
        return "cpu"
    available = compute_info()["cuda"]
    if preference == "cuda" and not available:
        raise ValueError(compute_info()["reason"])
    return "cuda" if available else "cpu"


@lru_cache(maxsize=1)
def nvenc_available():
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                 "color=c=black:s=640x360:r=30", "-frames:v", "1", "-c:v", "h264_nvenc",
                                 "-pix_fmt", "yuv420p", "-f", "null", "-"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15, creationflags=flags)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def info():
    return dict(compute_info(), nvenc=nvenc_available(), denoise_opencl=opencl_info())


@lru_cache(maxsize=1)
def opencl_info():
    """Require a real GPU and a successful UMat denoise probe, not just an OpenCL build."""
    import cv2
    import numpy as np
    result = dict(available=False, device="CPU", reason="没有可用的 OpenCL GPU")
    try:
        cv2.ocl.setUseOpenCL(True)
        device = cv2.ocl.Device_getDefault()
        if not cv2.ocl.useOpenCL() or not device.available() or not device.type() & cv2.ocl.Device_TYPE_GPU:
            return result
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        output = cv2.fastNlMeansDenoisingColored(cv2.UMat(image), None, 3, 3, 7, 21).get()
        if output.shape != image.shape or not cv2.ocl.useOpenCL():
            return result
        result.update(available=True, device=device.name(), reason="")
    except cv2.error:
        result["reason"] = "OpenCL GPU 去噪检查失败，可使用 CPU 去噪"
    return result
