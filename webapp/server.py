from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import engine, runtime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "web"
RUNS.mkdir(parents=True, exist_ok=True)
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-task")
LOCK = threading.RLock()
JOBS = {}
CANCEL = {}
MAX_UPLOAD = 512 * 1024 * 1024
MAX_PENDING = 50
MAX_PROCESSING_SIDE = 1920
DEFAULT_METHODS = ["dct", "spectral", "denoise", "vae", "geometry", "guided", "temporal"]
SAMPLE = ROOT / "outputs" / "20261027-105027"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    methods: list[str] = Field(default_factory=lambda: list(DEFAULT_METHODS), min_length=1, max_length=8)
    strength: float = Field(default=3, ge=.1, le=10)
    seconds: float = Field(default=10, ge=.25, le=300)
    fps: int = Field(default=8, ge=1, le=30)
    max_side: int = Field(default=640, ge=128, le=MAX_PROCESSING_SIDE)
    crf: int = Field(default=18, ge=10, le=30)
    steps: int = Field(default=6, ge=1, le=30)
    frequency_mode: str = "dct"
    band: str = "mid_high"
    message_bits: str = ""
    compute_device: str = "auto"
    encoder: str = "auto"
    vae_mode: str = "auto"
    denoise_device: str = "auto"
    frame_mode: str = "all"
    random_frames: int = Field(default=10, ge=1, le=9000)
    random_seed: int = Field(default=42, ge=0, le=4294967295)
    export_mode: str = "clip"

    @field_validator("export_mode")
    @classmethod
    def valid_export_mode(cls, value):
        if value not in ("clip", "full"):
            raise ValueError("导出范围无效")
        return value

    @field_validator("frame_mode")
    @classmethod
    def valid_frame_mode(cls, value):
        if value not in ("all", "random"):
            raise ValueError("抽帧方式无效")
        return value

    @field_validator("compute_device")
    @classmethod
    def valid_device(cls, value):
        if value not in ("auto", "cpu", "cuda"):
            raise ValueError("计算设备无效")
        return value

    @field_validator("encoder")
    @classmethod
    def valid_encoder(cls, value):
        if value not in ("auto", "cpu"):
            raise ValueError("编码设备无效")
        return value

    @field_validator("vae_mode")
    @classmethod
    def valid_vae_mode(cls, value):
        if value not in ("auto", "legacy", "fast"):
            raise ValueError("VAE 运算方式无效")
        return value

    @field_validator("denoise_device")
    @classmethod
    def valid_denoise_device(cls, value):
        if value not in ("auto", "cpu", "opencl"):
            raise ValueError("去噪设备无效")
        return value

    @field_validator("methods")
    @classmethod
    def valid_methods(cls, values):
        if len(set(values)) != len(values) or any(v not in {m["id"] for m in engine.METHODS} for v in values):
            raise ValueError("处理方法无效或重复")
        return values

    @field_validator("frequency_mode")
    @classmethod
    def valid_mode(cls, value):
        if value not in ("dct", "wavelet"):
            raise ValueError("频域模式无效")
        return value

    @field_validator("band")
    @classmethod
    def valid_band(cls, value):
        if value not in ("mid", "mid_high"):
            raise ValueError("频带无效")
        return value

    @field_validator("message_bits")
    @classmethod
    def valid_message(cls, value):
        value = value.strip()
        if value and not re.fullmatch("[01]{256}", value):
            raise ValueError("参考消息必须是 256 位 0 / 1 字符")
        return value


def save(job):
    folder = RUNS / job["id"]
    pending = folder / "job.tmp"
    pending.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(folder / "job.json")


def recover():
    for path in RUNS.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in ("queued", "running", "cancelling"):
                job.update(status="interrupted", phase="服务停止，任务已中断", finished_at=datetime.now().isoformat())
                save(job)
            elif job["status"] == "cancelled" and job.get("phase", "").startswith("正在取消"):
                job["phase"] = "已取消"
                save(job)
            JOBS[job["id"]] = job
        except (OSError, ValueError, KeyError):
            continue


@asynccontextmanager
async def lifespan(app):
    recover()
    yield
    for event in list(CANCEL.values()):
        event.set()
    POOL.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="频谱工坊 · VideoSeal Lab", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_writes(request: Request, call_next):
    # Keep cross-site requests from submitting local CPU jobs.
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin not in ("http://127.0.0.1:8765", "http://localhost:8765"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "仅允许本地网页提交任务"}, status_code=403)
    return await call_next(request)


@app.get("/api/capabilities")
def get_capabilities():
    acceleration = runtime.info()
    return dict(methods=engine.capabilities(), ffmpeg=bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
                sample_ready=(SAMPLE / "watermarked_sample.mp4").is_file(), max_upload_mb=512,
                max_pending_jobs=MAX_PENDING, default_methods=DEFAULT_METHODS, max_processing_side=MAX_PROCESSING_SIDE, adaptive_vae=True,
                max_processing_frames=engine.frame_limit({}),
                fast_vae=engine.vae_ready("fast"),
                scope="未知平台暗水印未验证。处理完成不代表水印已清除。", device=acceleration["device"], acceleration=acceleration, vae_path=str(engine.VAE_PATH))


@app.get("/api/jobs")
def list_jobs():
    with LOCK:
        return sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)


def find_job(job_id):
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "任务不存在")
        return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return find_job(job_id)


def worker(job_id, cfg):
    started = time.monotonic()
    event = CANCEL[job_id]

    def update(log=None, **fields):
        with LOCK:
            job = JOBS[job_id]
            if job["status"] == "cancelling" and fields.get("status") not in ("cancelled", "failed", "completed"):
                fields.pop("phase", None)
            job.update(fields)
            job["elapsed_seconds"] = round(time.monotonic() - started, 1)
            if log:
                job["logs"].append(dict(time=datetime.now().strftime("%H:%M:%S"), text=log))
            save(job)

    try:
        engine.check_cancel(event)
        update(status="running", started_at=datetime.now().isoformat(), phase="准备处理", log="任务开始，仅在本机执行")
        report = engine.run_pipeline(RUNS / job_id, cfg, event, update)
        engine.check_cancel(event)
        update(status="completed", phase="处理完成 · 查看验证范围", progress=100, report=report,
               finished_at=datetime.now().isoformat(), log="结果视频与报告已保存；处理完成不代表未知水印已清除")
    except engine.Cancelled as exc:
        update(status="cancelled", phase="已取消", error=str(exc), finished_at=datetime.now().isoformat(), log=str(exc))
    except Exception as exc:
        update(status="failed", phase="处理失败", error=str(exc), finished_at=datetime.now().isoformat(), log=f"失败：{exc}")


async def store_upload(upload, folder, prefix):
    suffix = Path(upload.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"):
        raise HTTPException(400, "请选择 MP4 / MOV / MKV / WEBM / AVI / M4V 视频")
    dest = folder / f"{prefix}{suffix}"
    total = 0
    with dest.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise HTTPException(413, "单个文件不能超过 512 MB")
            output.write(chunk)
    if not total:
        raise HTTPException(400, "视频文件为空")
    return dest


@app.post("/api/jobs", status_code=202)
async def create_job(config: str = Form(...), sample: bool = Form(False), video: UploadFile | None = File(None),
                     reference_clean: UploadFile | None = File(None), reference_marked: UploadFile | None = File(None),
                     batch_id: str = Form("", max_length=32, pattern=r"^$|^[0-9a-f]{32}$"),
                     request_id: str = Form("", max_length=32, pattern=r"^$|^[0-9a-f]{32}$")):
    try:
        settings = Settings.model_validate_json(config)
    except ValidationError as exc:
        raise HTTPException(422, "; ".join(e["msg"] for e in exc.errors()))
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise HTTPException(400, "请先安装 FFmpeg 与 ffprobe")
    ready = {m["id"]: m["ready"] for m in engine.capabilities()}
    if "vae" in settings.methods and not engine.vae_ready(settings.vae_mode):
        command = "python prepare_vae.py --tiny" if settings.vae_mode == "fast" else "setup_web.ps1 -WithVae"
        raise HTTPException(400, "所选 VAE 模型未就绪，请先运行 " + command)
    missing = [m for m in settings.methods if not ready[m]]
    if missing:
        raise HTTPException(400, "所选功能尚未就绪：" + ", ".join(missing) + "。VAE 请先运行 setup_web.ps1 -WithVae")
    if settings.compute_device == "cuda" and not runtime.compute_info()["cuda"]:
        raise HTTPException(400, runtime.compute_info()["reason"])
    if "denoise" in settings.methods and settings.denoise_device == "opencl" and not runtime.opencl_info()["available"]:
        raise HTTPException(400, "GPU 去噪不可用，请选择自动或 CPU 去噪")
    if "carrier" in settings.methods and (not reference_clean or not reference_marked):
        raise HTTPException(400, "载波抵消需要上传已对齐的原片与水印片")
    if not sample and video is None:
        raise HTTPException(400, "请上传视频或使用验证样本")
    with LOCK:
        if request_id:
            existing = next((j for j in JOBS.values() if j.get("request_id") == request_id), None)
            if existing:
                return existing
        if sum(j["status"] in ("queued", "running", "cancelling") for j in JOBS.values()) >= MAX_PENDING:
            raise HTTPException(429, f"已有 {MAX_PENDING} 个待处理任务，请等待部分任务完成后重试未提交的视频")
    job_id = uuid.uuid4().hex
    folder = RUNS / job_id
    folder.mkdir(parents=True)
    try:
        if sample:
            source = SAMPLE / "watermarked_sample.mp4"
            if not source.is_file():
                raise HTTPException(400, "本地验证样本不存在")
            shutil.copyfile(source, folder / "input.mp4")
            settings.message_bits = json.loads((SAMPLE / "message.json").read_text(encoding="utf-8"))["reference_bits"]
            name = "VideoSeal 已知水印样本.mp4"
        else:
            await store_upload(video, folder, "input")
            name = Path(video.filename).name
        if "carrier" in settings.methods:
            clean_path = await store_upload(reference_clean, folder, "reference-clean")
            marked_path = await store_upload(reference_marked, folder, "reference-marked")
            if clean_path.stat().st_size == marked_path.stat().st_size:
                with clean_path.open("rb") as clean_stream, marked_path.open("rb") as marked_stream:
                    identical = hashlib.file_digest(clean_stream, "sha256").digest() == hashlib.file_digest(marked_stream, "sha256").digest()
                if identical:
                    raise HTTPException(400, "两份参考视频完全相同，无法估计水印残差。请取消参考载波抵消，或提供真正不同的原片与水印片。")
        cfg = settings.model_dump()
        # Reject invalid media before occupying the serial processing queue.
        try:
            metadata = await asyncio.to_thread(engine.probe, next(folder.glob("input.*")))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, "无法读取这个视频，请确认文件完整，并能在本机播放器正常播放。") from exc
        limit = engine.frame_limit(cfg)
        frames = math.ceil(min(metadata["duration"], cfg["seconds"]) * cfg["fps"])
        if frames > limit:
            raise HTTPException(400, f"所选片段约 {frames} 帧，超过 {limit} 帧上限，请缩短处理时长或降低帧率后重试")
        job = dict(id=job_id, name=name, status="queued", phase="等待处理", progress=0, logs=[],
                   created_at=datetime.now().isoformat(), config=cfg, elapsed_seconds=0,
                   batch_id=batch_id, request_id=request_id)
        with LOCK:
            # Uploads yield to other requests; recheck before reserving the queue slot.
            if request_id:
                existing = next((j for j in JOBS.values() if j.get("request_id") == request_id), None)
                if existing:
                    return existing
            if sum(j["status"] in ("queued", "running", "cancelling") for j in JOBS.values()) >= MAX_PENDING:
                raise HTTPException(429, f"已有 {MAX_PENDING} 个待处理任务，请等待部分任务完成后重试未提交的视频")
            JOBS[job_id] = job
            CANCEL[job_id] = threading.Event()
            save(job)
        POOL.submit(worker, job_id, cfg)
        return job
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc))
    finally:
        for upload in (video, reference_clean, reference_marked):
            if upload:
                await upload.close()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = find_job(job_id)
    with LOCK:
        if job["status"] in ("queued", "running"):
            CANCEL[job_id].set()
            job.update(status="cancelling", phase="正在取消，等待当前步骤结束")
            save(job)
    return job


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str, download: bool = False):
    job = find_job(job_id)
    if name not in ("before.mp4", "result.mp4", "before.png", "after.png", "difference.png", "report.json"):
        raise HTTPException(404)
    path = RUNS / job_id / name
    if not path.is_file():
        raise HTTPException(404, "文件尚未生成")
    filename = name
    if name == "result.mp4":
        filename = f"{Path(job['name']).stem}_处理后.mp4"
    elif name == "report.json":
        filename = f"{Path(job['name']).stem}_报告.json"
    return FileResponse(path, filename=filename if download else None)


app.mount("/", StaticFiles(directory=ROOT / "webapp" / "static", html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.server:app", host="127.0.0.1", port=8765)
