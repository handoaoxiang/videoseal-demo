# VAE 分块与 GPU 去噪实测

设备：NVIDIA GeForce GTX 1660，6 GB 显存；PyTorch 2.7.1+cu126；VAE 为本地 `sd-vae-ft-mse`，FP32。去噪使用现有 OpenCV OpenCL GPU 路径，无新增依赖。

样本：本地历史任务 `1b1b93980a564cb2a7c4bf43d7d288f5/input.mp4`（1080×1920）。每种尺寸取同一时刻的单帧，强度 3；每个模块预热一次，再执行两次取均值。该测量不代表完整视频总耗时，也不覆盖不同内容与长时间运行的波动。

| 处理尺寸 | 原 VAE 256 分块 | 自动 VAE | 自动方式 | CPU 去噪 | OpenCL GPU 去噪 |
|---|---:|---:|---|---:|---:|
| 540×960 | 2.908 s | 1.938 s | 整帧 | 1.583 s | 0.178 s |
| 720×1280 | 5.772 s | 4.750 s | 512 分块 | 1.078 s | 0.252 s |
| 1080×1920 | 11.480 s | 10.427 s | 512 分块 | 1.467 s | 0.592 s |

1080×1920 时，VAE 耗时降低约 9.2%，去噪耗时降低约 59.6%；两模块相加降低约 14.9%。其他模块、抽帧、编码等耗时需要另计，不能把去噪的加速倍数当作整段视频的加速倍数。

PyTorch 峰值已分配显存：540×960 的原 VAE / 自动 VAE 为 534 / 1870 MiB；720×1280 为 547 / 1119 MiB；1080×1920 为 582 / 1150 MiB。这些数值不包含 OpenCL、显示驱动等分配，不能视为整张显卡总占用。

输出差异：自动 VAE 相对原分块结果，三种尺寸的平均绝对像素差约为 1.55–1.76（8 位像素）；GPU 去噪相对 CPU 约为 1.26–1.30。相对输入画面的 VAE SSIM 分别从 0.9829 / 0.9753 / 0.9858 变为 0.9864 / 0.9820 / 0.9901，GPU 去噪的 SSIM 比 CPU 低约 0.0006–0.0009。不同路径不保证逐像素一致；这些单帧画质指标无法判断未知平台暗水印是否消除。

自动 VAE 为整帧计算预留显存余量，显存不足时依次尝试 512、256 分块，保持尺寸与 FP32。自动去噪失败后使用 CPU 并在日志/报告中标记；强制 GPU 失败时明确报错。网页保留原版 256 分块与 CPU 去噪选项。

原始 JSON 与对照 PNG 保存在 `outputs/gpu-processing-20261030-optimized/`。复测时使用一个尚不存在的新输出目录：

```powershell
.\.venv\Scripts\python.exe research/benchmark_gpu_processing.py --input <视频绝对路径> --output <新的输出目录>
```

实现依据：[Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)、[OpenCV 去噪的 OpenCL 调度源码](https://github.com/opencv/opencv/blob/5.x/modules/photo/src/denoising.cpp)。实际是否使用 GPU 以本机能力检查及任务报告为准。
