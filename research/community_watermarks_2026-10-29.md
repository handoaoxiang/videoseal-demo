# GitHub / Linux.do 暗水印讨论核查与完整导出验收

核查日期：2026-10-29，Asia/Shanghai。此次只阅读公开材料，不向论坛发帖、不上传用户视频到外部服务，也未安装所查到的第三方项目。

## 结论

此次检索没有找到可复现、具有独立千川检测结果的公开去暗水印方案。值得进一步研究的是视频重建的对照实验和跨帧稳定性，但不能把其他模型的实验结论当成千川结果。现有网页的完整导出功能已完成实际下载验收。

## Linux.do 相关帖子

| 帖子 | 实际读到的内容 | 证据边界 |
| --- | --- | --- |
| [暗水印检测工具](https://linux.do/t/topic/2259717)，2026-05-28 | 讨论指向 reverse-SynthID、blind_watermark 等项目 | 推荐链接，没有平台视频对照实验 |
| [「净水器」对抗传统频域水印，小幅度干扰AI水印](https://linux.do/t/topic/2146226)，2026-05-10 | 作者介绍微小几何变化、噪声、压缩的组合，也承认对 AI 水印不能确认清除 | 图片实验性讨论，没有千川验证数据 |
| [我找到了个去除Google SynthID 的方法 支持纯本地](https://linux.do/t/topic/1217816)，2025-11-25 | 楼主分享 img2img 重绘思路和 ComfyUI 工作流 | 针对图片的用户报告，没有视频稳定性、样本规模或千川验证 |
| [有没有简单好用的视频去重方案](https://linux.do/t/topic/2925327)，2026-09-20 | 有人称自己的旧程序无法通过当时的抖音检测 | 用户经验，没有公开代码和可复现记录；不能据此确定抖音算法 |

检索到的千川课程分享帖主要是网盘目录，不包含能直接核查的算法、测试数据或检测器。标题里的“破解”不能算有效性证据。

## GitHub 项目和问题反馈

### reverse-SynthID：本地评分不能代替目标检测

[Issue #45](https://github.com/aloshdenny/reverse-SynthID/issues/45) 的发帖人报告，5 张图片本地结果良好，但上传 Gemini 后仍被识别为 Google AI 内容，后续回复也报告相同情况。该页面读取时仍为 Open。这是用户反馈，未在本机复现，但足以反对把它的代理评分当成官方判定。

### remove-ai-watermarks：有视频重建路径，但样本证据有限

[项目](https://github.com/wiltodelta/remove-ai-watermarks)明确表示没有本地 SynthID 像素检测器。它提供视频重建路径，不能据此给出任意文件的实时阴性结论。

[当前实现](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/src/remove_ai_watermarks/video_invisible.py)使用 VAE 编解码，并在各帧潜变量上复用同一份由种子生成的空间噪声；这与当前网页仅做 VAE 重建、再混合原图的实现不同。

[原始验证清单](https://raw.githubusercontent.com/wiltodelta/remove-ai-watermarks/main/data/evaluations/video-synthid-oracle.csv)只有同一个 8 秒 Veo 视频的两行：

| 噪声标准差 | 长边 / FPS | 仓库记录的判定 |
| --- | --- | --- |
| 0.10 | 512 / 12 | detected |
| 0.15 | 512 / 12 | not_detected |

两行源视频哈希相同，原片为 1920×1080、24 FPS；VAE 一列写的是 unrecorded。它记录了特定输出的结果，不能代表跨内容、跨设备的成功率，更不能外推到千川。此次没有对这些输出做独立复测。

[作者的画质研究文档](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/video-synthid-quality-research.md)也区分了画质测量和水印验证。默认配置明显降低输出尺寸和帧率；不应直接套用到用户要求保留原规格的视频。该文是研究归档，里面的历史实现问题不能直接当成当前代码缺陷。

### watermarks-remover：区分元数据和像素处理

[项目](https://github.com/guillaumemeyer/watermarks-remover)既提供元数据清理，也有可选 CtrlRegen 像素重建后端。后者是额外模型流程，不是清理容器标签就完成。

[视频处理 PR #265](https://github.com/guillaumemeyer/watermarks-remover/pull/265)讨论了逐帧处理与时间投票，但作者明确将效果保证限定于假设的投票模型，未由厂商检测器验证，且逐帧启动后端可能很慢。不能借此假定千川采用同一投票规则或阈值。

## 千川官方说明的本次复核

通过浏览器重新读取了[官方文档](https://support.oceanengine.com/support/content/207175?spaceId=195)内嵌正文，页面标记更新时间为 2026-10-28。文档区分相似分镜判断与投放账户专属暗水印识别；千川暗水印治理从 2026-06-22 起生效。具体机制及出处另见 [官方说明核查](qianchuan_rules_2026-10-29.md)。

问答没有提供“只占 10% 或 3～5 秒”的命中阈值。不能据此把随机处理 N 帧或拼接未处理后段解释为已解除搬运识别。

## 本项目验收记录

- 重新运行 `python -m unittest discover -s tests -v`：8 项通过，约 41.8 秒。包括拼接点、后段时间线、连续音频、无音频及无后段情况。
- `node --check webapp/static/app.js` 通过。
- 本地服务实际接受 `export_mode=full`，任务 `427438a258ff46d298802e28038dfb52` 正常完成。
- 输入为现有 4 秒 QA 样本，处理前 2 秒；报告和实际下载视频均为完整 4 秒、256×256，包含音频，实际编码为 NVENC。
- 在网页切换了完整导出选项、查看任务结果并点击下载。下载文件与 HTTP 文件接口返回的内容逐字节一致，原始输入仍存在。
- [任务报告](../outputs/web/427438a258ff46d298802e28038dfb52/report.json)、[结果视频](../outputs/web/427438a258ff46d298802e28038dfb52/result.mp4)、[界面截图](../outputs/web/full-export-verified-20261029.png)。

这些验证证明导出范围、拼接和文件下载正常，不证明平台暗水印已清除。
