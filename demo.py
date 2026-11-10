"""Local VideoSeal round-trip demo using Meta's standalone TorchScript model."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "checkpoints" / "y_256b_img.jit"
MODEL_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit"
SAMPLE = ROOT / "assets" / "official_sample.mp4"
SAMPLE_URL = "https://raw.githubusercontent.com/facebookresearch/videoseal/main/assets/videos/1.mp4"
SIZE, FPS, FRAMES, BITS = 256, 4, 8, 256
SCOPE = "仅验证 VideoSeal 自己的水印；结果不能用于判断抖音等平台的私有暗水印。"


def download_model(log=print):
    MODEL.parent.mkdir(exist_ok=True)
    if MODEL.is_file():
        return
    partial = MODEL.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as output:
            expected = int(response.headers.get("Content-Length", "0"))
            written = last = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                written += len(block)
                if written - last >= 32 * 1024 * 1024:
                    log(f"模型下载：{written / 2**20:.0f} / {expected / 2**20:.0f} MiB")
                    last = written
        if expected and partial.stat().st_size != expected:
            raise RuntimeError("模型下载不完整，请重新运行")
        partial.replace(MODEL)
    finally:
        partial.unlink(missing_ok=True)


def download_sample(log=print):
    if SAMPLE.is_file():
        return
    log("下载官方示例视频（约 3.3 MiB，仅首次）")
    SAMPLE.parent.mkdir(exist_ok=True)
    partial = SAMPLE.with_suffix(".part")
    try:
        with urllib.request.urlopen(SAMPLE_URL, timeout=60) as response, partial.open("wb") as output:
            expected = int(response.headers.get("Content-Length", "0"))
            shutil.copyfileobj(response, output)
        if expected and partial.stat().st_size != expected:
            raise RuntimeError("示例视频下载不完整")
        partial.replace(SAMPLE)
    finally:
        partial.unlink(missing_ok=True)


def ffmpeg(args, data=None):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *args]
    result = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=flags, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def read_frames(path):
    import numpy as np
    data = ffmpeg(["-i", str(path), "-t", "2", "-vf",
                   f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,crop={SIZE}:{SIZE}",
                   "-frames:v", str(FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    if not data:
        raise RuntimeError("没有读到视频帧")
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, SIZE, SIZE, 3).copy()


def write_frames(frames, path):
    ffmpeg(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
            "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "12",
            "-preset", "fast", "-pix_fmt", "yuv420p", str(path)], frames.tobytes())


def extract(model, video):
    import torch
    with torch.inference_mode():
        predictions = model.detect(video, is_video=True)
    # Check the exported interface instead of interpreting arbitrary output as a detection.
    if predictions.shape[1] == BITS + 1:
        logits = predictions[:, 1:]
    elif predictions.shape[1] == BITS:
        logits = predictions
    else:
        raise RuntimeError(f"模型输出结构不符：{tuple(predictions.shape)}")
    if logits.ndim > 2:
        logits = logits.flatten(2).mean(dim=-1)
    return (logits.mean(dim=0) > 0).to(torch.int32).cpu().tolist()


def run_demo(source=None, log=print):
    import numpy as np
    import torch
    from PIL import Image

    if not shutil.which("ffmpeg"):
        raise RuntimeError("未找到 FFmpeg，请将它加入 PATH")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.perf_counter()
    output = ROOT / "outputs" / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    log(SCOPE)
    log("第 1/5 步：准备 2 秒、256×256 的演示副本，不修改输入视频")
    if source:
        source = Path(source).resolve(strict=True)
    else:
        download_sample(log)
        source = SAMPLE
    frames = read_frames(source)
    clean_path = output / "clean_sample.mp4"
    watermarked_path = output / "watermarked_sample.mp4"
    write_frames(frames, clean_path)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255

    log("第 2/5 步：加载官方 VideoSeal v1.0 模型（CPU）")
    download_model(log)
    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    # Embed each of the eight demo frames; keep the model's default watermark strength.
    model.step_size = 1
    generator = torch.Generator().manual_seed(20261027)
    message = torch.randint(0, 2, (1, BITS), generator=generator).float()
    expected = message.to(torch.int32).flatten().tolist()

    log("第 3/5 步：嵌入已知的 256 位测试消息")
    with torch.inference_mode():
        embedded = model.embed(video, message, is_video=True)
    watermarked = embedded.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    write_frames(watermarked, watermarked_path)
    Image.fromarray(frames[0]).save(output / "clean_frame.png")
    Image.fromarray(watermarked[0]).save(output / "watermarked_frame.png")
    difference = np.clip(np.abs(watermarked[0].astype(float) - frames[0].astype(float)) * 10, 0, 255).astype(np.uint8)
    Image.fromarray(difference).save(output / "difference_x10.png")

    log("第 4/5 步：从实际保存的 MP4 重新解码，再用 extractor 提取")
    def tensor(path):
        return torch.from_numpy(read_frames(path)).permute(0, 3, 1, 2).float() / 255
    clean_bits = extract(model, tensor(clean_path))
    memory_bits = extract(model, embedded)
    saved_bits = extract(model, tensor(watermarked_path))
    match = lambda actual: sum(a == b for a, b in zip(actual, expected)) / BITS
    report = {
        "scope": SCOPE, "model": "VideoSeal v1.0 (official TorchScript)", "model_url": MODEL_URL,
        "device": "cpu", "torch_version": torch.__version__, "input_source": str(source),
        "frames": len(frames), "demo_resolution": [SIZE, SIZE], "demo_fps": FPS,
        "watermark_step_size": 1, "mp4_crf": 12,
        "message_bits": BITS, "reference_bits": "".join(map(str, expected)),
        "clean_reference_match": match(clean_bits), "embedded_memory_match": match(memory_bits),
        "saved_mp4_reference_match": match(saved_bits), "extracted_bits": "".join(map(str, saved_bits)),
        "roundtrip_pass": match(saved_bits) >= .9,
        "note": "匹配率是对本次已知随机消息的验证，不是通用水印概率。90% 仅为此 demo 的自检阈值。",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "message.json").write_text(json.dumps({"reference_bits": report["reference_bits"]}, indent=2), encoding="utf-8")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VideoSeal 本地演示</title>
<style>body{{font:16px/1.8 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#222}}.grid{{display:flex;gap:20px;flex-wrap:wrap}}figure{{margin:0}}img,video{{width:256px;border:1px solid #ddd}}table{{border-collapse:collapse}}td,th{{padding:8px 24px;border:1px solid #ddd}}.note{{background:#fff4d8;padding:14px;border-radius:8px}}</style>
<h1>VideoSeal 本地演示</h1><p class="note">{html.escape(SCOPE)}</p>
<p>CPU · {len(frames)} 帧 · 256×256 · 4 FPS · 耗时 {report['elapsed_seconds']} 秒</p>
<table><tr><th>样本</th><th>与已知 256 位消息的匹配率</th></tr>
<tr><td>未嵌入样本（对照组）</td><td>{match(clean_bits):.2%}</td></tr>
<tr><td>嵌入后（内存）</td><td>{match(memory_bits):.2%}</td></tr>
<tr><td>保存 MP4 后重新提取</td><td>{match(saved_bits):.2%}</td></tr></table>
<p>{html.escape(report['note'])}</p><div class="grid"><figure><img src="clean_frame.png"><figcaption>原始测试帧</figcaption></figure><figure><img src="watermarked_frame.png"><figcaption>嵌入水印后</figcaption></figure><figure><img src="difference_x10.png"><figcaption>差异放大 10 倍，仅用于观察</figcaption></figure></div>
<h2>测试视频</h2><div class="grid"><figure><video controls src="clean_sample.mp4"></video><figcaption>未嵌入样本</figcaption></figure><figure><video controls src="watermarked_sample.mp4"></video><figcaption>带 VideoSeal 水印的样本</figcaption></figure></div>
<p><a href="report.json">完整结果 JSON</a> · <a href="message.json">已知参考消息</a></p></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    log(f"第 5/5 步：完成。对照组 {match(clean_bits):.2%}；保存后消息匹配率 {match(saved_bits):.2%}")
    log(f"结果：{output}")
    return output, report


def gui():
    import tkinter as tk
    from tkinter import filedialog, ttk
    root = tk.Tk()
    root.title("VideoSeal 本地小 Demo")
    root.geometry("800x570")
    source = tk.StringVar()
    latest = [None]
    events = queue.Queue()
    box = ttk.Frame(root, padding=18)
    box.pack(fill="both", expand=True)
    ttk.Label(box, text="VideoSeal：嵌入 → 保存视频 → 提取验证", font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
    ttk.Label(box, text=SCOPE, wraplength=740, foreground="#986000").pack(anchor="w", pady=(8, 16))
    ttk.Label(box, text="输入视频可留空；留空使用官方示例视频。选视频时只处理前 2 秒的缩小副本。").pack(anchor="w")
    row = ttk.Frame(box)
    row.pack(fill="x", pady=10)
    ttk.Entry(row, textvariable=source).pack(side="left", fill="x", expand=True)
    ttk.Button(row, text="选择视频", command=lambda: source.set(filedialog.askopenfilename(filetypes=[("视频", "*.mp4 *.mov *.mkv"), ("所有文件", "*.*")]) or source.get())).pack(side="left", padx=(10, 0))
    bar = ttk.Progressbar(box, mode="indeterminate")
    bar.pack(fill="x", pady=8)
    logbox = tk.Text(box, height=16, wrap="word", font=("Microsoft YaHei UI", 10))
    logbox.pack(fill="both", expand=True)
    buttons = ttk.Frame(box)
    buttons.pack(fill="x", pady=(14, 0))
    def start():
        selected = source.get().strip()
        run.config(state="disabled")
        bar.start()
        def worker():
            try:
                output, report = run_demo(selected or None, lambda line: events.put(("log", line)))
                events.put(("done", output))
            except Exception as exc:
                events.put(("error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=worker, daemon=True).start()
    run = ttk.Button(buttons, text="运行一次 Demo", command=start)
    run.pack(side="left")
    report_button = ttk.Button(buttons, text="打开结果报告", state="disabled", command=lambda: webbrowser.open((latest[0] / "report.html").as_uri()))
    report_button.pack(side="left", padx=10)
    def poll():
        while not events.empty():
            kind, value = events.get_nowait()
            if kind == "done":
                latest[0] = value
                report_button.config(state="normal")
            if kind in ("done", "error"):
                bar.stop()
                run.config(state="normal")
            if kind != "done":
                logbox.insert("end", str(value) + "\n")
                logbox.see("end")
        root.after(100, poll)
    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__":
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=SCOPE)
    parser.add_argument("--self-test", action="store_true", help="命令行执行一次自检")
    parser.add_argument("--input", type=Path, help="可选视频，仅取前 2 秒的演示副本")
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    if args.download_model:
        download_model()
        download_sample()
    elif args.self_test:
        _, result = run_demo(args.input)
        raise SystemExit(0 if result["roundtrip_pass"] else 1)
    else:
        gui()
