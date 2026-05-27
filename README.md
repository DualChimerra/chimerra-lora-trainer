<div align="center">

# LoRA Trainer · Colab

**A productivity-first web UI for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — train SDXL & Anima LoRAs from a browser tab, powered by a single Google Colab notebook.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DualChimerra/chimerra-lora-trainer/blob/main/colab/LoRA_Trainer.ipynb)
[![License: MIT](https://img.shields.io/badge/UI-MIT-blue.svg)](#license)
[![sd-scripts](https://img.shields.io/badge/backend-kohya--ss%2Fsd--scripts-orange)](https://github.com/kohya-ss/sd-scripts)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#)

</div>

---

## ✨ Why

Training LoRAs on Colab usually means hand-editing TOML configs, juggling CLI flags, and tailing logs in a notebook cell. **LoRA Trainer · Colab** wraps the unmodified kohya-ss trainer in a Linear/Attio-style UI: every option becomes a typed field, every run is live-tracked, and every sample lands in a clickable gallery — all served from a single FastAPI process behind a Cloudflare tunnel.

The training logic itself is **never patched**. We only assemble the same CLI arguments upstream `sd-scripts` already accepts, then hand them to `accelerate launch`.

## 🎯 Supported architectures

| Architecture | Trainer script | Notes |
|---|---|---|
| **SDXL** — NoobAI, Illustrious | `sdxl_train_network.py` | classic ε / v-prediction, clip-skip, SNR tricks |
| **Anima** — DiT · Qwen3-0.6B TE · Qwen-Image VAE | `anima_train_network.py` | FlowMatch (`weighting_scheme`, `timestep_sampling`, `discrete_flow_shift`, …) |

Network kinds: **LoRA**, **LoCon**, **LoHa**, **LoKr**, **DyLoRA**, **IA³** (via [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS)).

## 🚀 Quick start

1. Open [`colab/LoRA_Trainer.ipynb`](colab/LoRA_Trainer.ipynb) in Google Colab on a GPU runtime.
2. Run the four setup cells in order — they install `sd-scripts`, `lycoris-lora`, and `cloudflared`.
3. The last cell prints a `https://<random>.trycloudflare.com` URL — that's your UI.
4. Open it in any browser, fill in the rest, and hit **▶ Старт**.

That's it. The Colab kernel hosts the FastAPI backend; your browser is the only client.

## 🏗️ Architecture

```
       ┌────────────────────────────────────┐
       │  Browser  (Preact + htm, no build) │
       └────────────────┬───────────────────┘
                        │ HTTP + WebSocket (via cloudflared)
       ┌────────────────▼───────────────────┐
       │  FastAPI  (backend/app.py)         │
       │  • config persistence              │
       │  • sandboxed file-system browser   │
       │  • train-process manager + log bus │
       │  • thumbnail cache, Drive sync     │
       └────────────────┬───────────────────┘
                        │ subprocess: accelerate launch
       ┌────────────────▼───────────────────┐
       │  kohya-ss/sd-scripts  (UNTOUCHED)  │
       │  sdxl_train_network.py             │
       │  anima_train_network.py            │
       └────────────────────────────────────┘
```

## 🧭 What's in the UI

| Section | What it does |
|---|---|
| **Project** | paths and the output filename |
| **Model** | arch picker (SDXL / Anima) + weight selection; Anima reveals `qwen3`, `t5_tokenizer_path`, `vae` |
| **Dataset** | auto-scans `N_concept` subfolders, computes repeats, resolution, bucketing, caches |
| **Network** | LoRA / LoCon / LoHa / LoKr / DyLoRA / IA³ with live warnings (e.g. `alpha > dim`) |
| **Training** | optimizer, LR, scheduler, duration, mixed precision; FlowMatch knobs for Anima |
| **Samples** | prompt list + `every_n_epochs` / `every_n_steps` schedule |
| **Gallery** | prompt × epoch matrix of generated samples — click for full-screen viewer |
| **Logs** | live stdout stream over WebSocket, filterable |
| **Files** | trained `.safetensors` outputs with size, mtime, and inferred epoch |
| **Presets** | bundled starters (NoobAI / Illustrious / LoKr / Anima) + your own; JSON import/export |

Plus a one-click **"Очистка файлов"** in the header to wipe local samples, trained LoRAs, state dir, and the thumb cache (Drive is never touched).

## 🔬 Under the hood

The trainer is launched exactly as if you ran it from the terminal:

```bash
accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --pretrained_model_name_or_path=... \
  --dataset_config=dataset.toml \
  --network_module=networks.lora \
  --network_dim=32 --network_alpha=16 \
  --optimizer_type=AdamW8bit --learning_rate=1e-4 \
  ...
```

[`backend/config_builder.py`](backend/config_builder.py) assembles this command **one-to-one** against upstream's argument surface. No monkey-patching, no forked scripts.

## 🔒 Filesystem sandbox

The UI can only list and read files inside the four "root" paths configured in the Colab cell (`DATASET_ROOT`, `BASE_MODEL_ROOT`, `OUTPUT_ROOT`, `SAMPLES_ROOT`). Any request that resolves outside those roots is rejected by FastAPI.

## 💾 State & presets

- The current config auto-saves to `{OUTPUT_ROOT}/.lora_trainer/last_config.json` on every change (600 ms debounce).
- Presets live at `{OUTPUT_ROOT}/.lora_trainer/presets/<name>.json`.
- If Drive is unmounted, everything falls back to `/content/.lora_trainer/`.
- An optional **local-SSD cache** mode mirrors Drive paths into `/content/cache/` for ~10× faster I/O during training, with an on-demand `rsync` push back to Drive.

## 🛠️ Local development (no Colab)

```bash
pip install -r requirements.txt

export LT_DATASET_ROOT=/tmp/dataset
export LT_BASE_MODEL_ROOT=/tmp/models
export LT_OUTPUT_ROOT=/tmp/output
export LT_SD_SCRIPTS_DIR=/tmp/sd-scripts   # clone upstream here

uvicorn backend.app:app --reload --port 7860
```

Open <http://127.0.0.1:7860>.

## 🧪 Anima support

`anima_train_network.py` takes separate paths for each component:

- `--pretrained_model_name_or_path` — Anima DiT checkpoint
- `--qwen3` — Qwen3-0.6B text encoder
- `--t5_tokenizer_path` — T5 tokenizer
- `--vae` — Qwen-Image VAE

Selecting `arch = anima` in the UI auto-reveals these fields, hides SDXL-only options (`clip_skip`, `v_parameterization`), and swaps the Training tab to FlowMatch parameters instead of the noise/SNR set.

## 📜 License

- **UI** (this repo): MIT.
- **Training backend** ([sd-scripts](https://github.com/kohya-ss/sd-scripts)): Apache 2.0, © kohya-ss.
- **LyCORIS**: Apache 2.0, © KohakuBlueleaf.
