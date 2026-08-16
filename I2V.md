# MiniMax H3 图生视频（I2V）能力说明

本文档统一说明 MiniMax H3 的 **图像→视频（I2V / Image-to-Video）** 能力：实现机制、模型依赖、下载方式与使用方法。I2V 与文生视频（T2V）、参考图生视频（R2V）共用同一套 ComfyUI 环境与扩散模型族。

---

## 1. 三种生成模式总览

MiniMax H3 提供三种以图/文驱动视频的能力，对应两个核心节点：

| 模式 | 节点类型 | 扩散模型 | 输入 | 适用场景 |
|------|----------|----------|------|----------|
| **T2V** 文生视频 | `MiniMaxH3ImageToVideo`（不接帧） | fl2va | 仅文本 prompt | 纯文字描述生成 |
| **I2V** 图生视频 | `MiniMaxH3ImageToVideo`（接 `first_frame`/`last_frame`） | fl2va | prompt + 首帧/尾帧图 | 用一张图作为起点生成视频（首帧驱动） |
| **R2V** 参考图生视频 | `MiniMaxH3ReferenceToVideo` | ref2va | prompt + 参考图/视频/音频（`<Picture i>` 标签） | 用参考图控制角色/物体外观 |

> 关键区别：
> - **I2V（fl2va）** 把图当作「关键帧」（keyframe），直接锚定视频的第 0 帧（或末帧），几何上拉伸到画布，属于「首帧/尾帧约束」。
> - **R2V（ref2va）** 把图当作「参考」（reference），通过 `<Picture i>` 标签让模型在生成过程中参考外观/风格，不锚定具体帧位置，可同时参考多张图、视频、音频。

---

## 2. 实现机制（基于 `comfy_extras/nodes_minimax_h3.py`）

### 2.1 I2V：`MiniMaxH3ImageToVideo`（t2va / fl2va）

节点输入（关键项）：

| 输入 | 类型 | 说明 |
|------|------|------|
| `clip` | CLIP | Qwen3-VL 文本/图像编码器 |
| `vae` | VAE | 视频 VAE（`minimax_h3_video_vae_fp16`） |
| `prompt` | string | 提示词（场景 + Camera: + Audio:） |
| `width` / `height` | int | 画布分辨率（32 的倍数） |
| `length` | int | 帧数（24fps，124 ≈ 5 秒，snap 到 17k+5 网格） |
| `first_frame` | IMAGE（可选） | **首帧图**：几何锚点，直接拉伸到画布 |
| `last_frame` | IMAGE（可选） | **尾帧图**：保持宽高比的 center-crop |

实现逻辑（源码摘录）：

```python
# first_frame：几何锚点，直接拉伸（disabled crop）到画布
img = _resize(first_frame[:1], width, height, "disabled")
keyframes.append({"resolved_frame_index": 0, "image": img})

# last_frame：尾帧，保持宽高比的 cover-crop（center）
img = _resize(last_frame[:1], width, height, "center")
keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

# 关键帧进入 conditioning，每一步 re-inject（永不参与去噪）
cond = node_helpers.conditioning_set_values(cond, {
    "minimax_keyframes": keyframes,
    "minimax_frame_count": frame_count,
})
```

要点：

- **首帧 = 第 0 帧硬约束**：生成视频的第一帧就是输入图（拉伸到画布），后续帧由扩散模型从该帧演化。
- **尾帧 = 末帧硬约束**：可选，用于「指定开头和结尾画面、中间自动过渡」。
- 关键帧以 latent 形式注入 DiT，每一步采样都重新注入，本身不被去噪。

### 2.2 R2V：`MiniMaxH3ReferenceToVideo`（ref2va）

节点输入（关键项）：

| 输入 | 类型 | 说明 |
|------|------|------|
| `clip` / `vae` | — | 同 I2V |
| `audio_vae` | VAE | 音频 VAE（`minimax_h3_audio_vae_fp32`），R2V 必需 |
| `ref_images` | IMAGE × 0–9 | 参考图，prompt 用 `<Picture i>` 引用 |
| `ref_videos` | IMAGE × 0–3 | 参考视频帧（2–15s @24fps），`<Video k>` 引用 |
| `ref_audios` | AUDIO × 0–3 | 独立参考音频，`<Audio j>` 引用 |
| `ref_image_size` | combo | `match`（匹配生成像素面积，快）/ `max`（2048 短边，保真但慢数倍） |

要点：

- 参考图/视频/音频的 token **贯穿每一步采样**（ride through every sampling step），因此 `max` 模式会显著变慢。
- 参考项按「图片 → 视频（配音频）→ 独立音频」的顺序进入 presentation，序号从 1 开始。

---

## 3. 模型依赖

三种模式共用同一套模型目录，按需放置：

| 模型文件 | 大小 | 目录 | T2V | I2V | R2V |
|----------|------|------|:---:|:---:|:---:|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20 GB | `diffusion_models/` | ✅ | ✅ | — |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20 GB | `diffusion_models/` | — | — | ✅ |
| `minimax_h3_video_vae_fp16.safetensors` | 4.9 GB | `vae/` | ✅ | ✅ | ✅ |
| `minimax_h3_audio_vae_fp32.safetensors` | 578 MB | `vae/` | 可选 | 可选 | ✅ |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15 GB | `text_encoders/` | ✅ | ✅ | ✅ |
| `RealESRGAN_x2plus.pth` | 64 MB | `upscale_models/` | 可选超分 | 可选超分 | 可选超分 |

> 扩散模型命名规律：
> - `fl2va` = First/Last-frame to Video → I2V（首尾帧驱动）
> - `ref2va` = Reference to Video → R2V（参考图驱动）
> - `t2va` 与 `fl2va` 共用同一节点，无帧时即纯文本 T2V

**最小 I2V 依赖**：fl2va 扩散模型 + 视频 VAE + 任一 CLIP 文本编码器（共 ~40 GB）。
**完整三种模式**：fl2va + ref2va + 视频 VAE + 音频 VAE + CLIP（共 ~60 GB，不含可选超分与 Heretic TE）。

---

## 4. 下载方式（统一）

### 4.1 推荐：目前 repo 的交互式安装器

```bash
wget https://raw.githubusercontent.com/cyyeh/dgxspark_comfyui_minimax_h3/main/install_wizard.py
python3 install_wizard.py
```

安装器只使用 Hugging Face，并逐一检查模型的精确字节数：已完成则跳过，未完成则以 `wget --continue` 续传。下载中断后直接重跑同一命令即可。

### 4.2 固定的 Hugging Face 来源

- 主模型、VAE、Heretic/nvFP4 编码器、超分模型：`drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark-weights`，revision `2e4f1dbbbc3b42a6c92ac685aa5c4a060c32b05e`
- INT8/BF16 补充编码器：`Comfy-Org/MiniMax-H3`，revision `d07f69bc8fa09c9717e1e47180034f9322e0e54d`

手动下载单一文件时也应固定 revision，并保留同名部分文件以便续传：

```bash
export http_proxy=http://PROXY:port; export https_proxy=$http_proxy
mkdir -p ComfyUI/models/diffusion_models
cd ComfyUI/models/diffusion_models
wget --continue --tries=0 --timeout=60 --read-timeout=60 \
  https://huggingface.co/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark-weights/resolve/2e4f1dbbbc3b42a6c92ac685aa5c4a060c32b05e/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

---

## 5. 使用方法

### 5.1 工作流（UI）

仓库内已提供 I2V 专用工作流：

| 工作流 | 说明 |
|--------|------|
| `h3-i2v-firstframe-enhanced.json` | ⭐ I2V 首帧驱动 + 完整加速栈（SageAttn + SolAttn + Spectrum + FBC）+ 2× 超分，640×360 → 720p |
| `h3-enhanced-fullstack.json` | I2V/T2V 增强版（无首帧时即 T2V） |
| `h3-r2v-heretic-enhanced.json` | R2V 参考图生视频 |

**I2V 操作步骤**：

1. 把首帧图放到 ComfyUI 输入目录：`ComfyUI/input/first_frame.png`
2. 打开 `http://<IP>:8188`，加载 `h3-i2v-firstframe-enhanced.json`
3. 找到 **Node #137（LoadImage）**，把 `image` 改成你的图片名
4. 找到 **Node #104（MiniMaxH3ImageToVideo）**，改 `prompt`（描述从首帧演化的画面）
5. 点 **Queue Prompt**（Ctrl+Enter）

**可选尾帧**：再拖入一个 `LoadImage` 节点，把输出连到 Node #104 的 `last_frame` 输入即可「首尾帧中间过渡」。

### 5.2 API（编程提交）

通过 ComfyUI 的 `/prompt` 接口提交，核心是把图片节点输出连到 `first_frame`：

```python
import json, requests

wf = json.load(open("h3-i2v-firstframe-enhanced.json"))
wf["104"]["inputs"]["prompt"] = "你的 I2V 提示词"
wf["104"]["inputs"]["first_frame"] = ["137", 0]   # LoadImage 输出 0
wf["137"]["inputs"]["image"] = "first_frame.png"   # 输入图名

resp = requests.post("http://127.0.0.1:8188/prompt", json={"prompt": wf})
print(resp.json())
```

### 5.3 分辨率与加速

- 采用「半分辨率生成 + 2× 超分」可提速约 6 倍：`width=640, height=360` → RealESRGAN 2× → 1280×720。
- 分辨率须为 32 的倍数；帧数 snap 到 17k+5 网格（124 ≈ 5 秒）。
- I2V（fl2va）不需要音频 VAE；只有 R2V 必须加载音频 VAE。

---

## 6. 常见问题

**Q: I2V 和 R2V 该用哪个？**
A: 想让视频「第一帧就是你给的图」（精确首帧锚定）→ 用 I2V（first_frame）。想让模型「参考某张图的外观/风格但自由生成」→ 用 R2V（ref_images + `<Picture i>`）。

**Q: 首帧图被拉伸变形？**
A: `first_frame` 是「直接拉伸到画布」（disabled crop），请先把首帧图裁成与生成分辨率相同的宽高比；`last_frame` 是 center-crop 保持比例，不受影响。

**Q: 报 `minimax_h3_fl2va` 找不到？**
A: 确认 `diffusion_models/` 下有 fl2va 模型，且工作流 `UNETLoader` 的 `unet_name` 与实际文件名一致。

**Q: R2V 报缺 audio_vae？**
A: `MiniMaxH3ReferenceToVideo` 的 `audio_vae` 是必需输入，需放置 `minimax_h3_audio_vae_fp32.safetensors`。

---

## 7. 相关文档

- 部署总览：[README.md](README.md) / [DEPLOYMENT.md](DEPLOYMENT.md)
- 全部 12+ 工作流说明：[WORKFLOWS.md](WORKFLOWS.md)
- 测速报告：[BENCHMARK.md](BENCHMARK.md)
- 上游源码：`comfy_extras/nodes_minimax_h3.py`（`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`）
