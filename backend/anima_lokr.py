"""Isolated training path for **Anima + LyCORIS LoKr**.

This module is the bridge to WalkingMeatAxolotl/AnimaLoraStudio. When (and only
when) the user picks arch=anima AND network.kind=lokr, training does NOT go
through kohya/sd-scripts at all. Instead we translate our `TrainConfig` into
AnimaLoraStudio's `TrainingConfig` YAML and launch *its* training engine
(`runtime/anima_train.py`), which drives the official `lycoris` library with an
Anima-aware preset (attn q/k/v/output_proj + mlp, excluding llm_adapter).

Everything here is deliberately self-contained so the main kohya pipeline in
`config_builder.build_command` stays byte-for-byte unchanged for every other
architecture/algorithm combination.

Mapping notes (our field -> their TrainingConfig field):
  * Their `data_dir` is a single root whose `N_xxx` subdirs encode per-folder
    repeats (kohya convention). Our dataset is a list of subsets with explicit
    `num_repeats`. We materialize a `<workdir>/anima_dataset/<repeats>_<name>`
    symlink tree so our UI repeats are honored verbatim by their parser.
  * Their text-encoder loader (`AutoModelForCausalLM.from_pretrained`) expects a
    Qwen *directory*, whereas our `anima_qwen3` may point at a single
    .safetensors. We forward it as-is; if it is a bare file the user must point
    the Qwen field at the HF model dir. See `text_encoder_path` below.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import TrainConfig


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------
def is_anima_lokr(cfg: TrainConfig) -> bool:
    """True only for the Anima + LyCORIS LoKr combo handled by this module."""
    return cfg.model.arch == "anima" and cfg.network.kind == "lokr"


# ---------------------------------------------------------------------------
# Small mapping helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """Filesystem-safe folder leaf for a symlinked dataset subset."""
    leaf = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
    return leaf or "concept"


def _opt_weight_decay(optimizer_args: List[str]) -> float:
    """Pull `weight_decay=<x>` out of our free-form kohya optimizer_args list."""
    for a in optimizer_args or []:
        m = re.match(r"\s*weight_decay\s*=\s*([0-9.eE+\-]+)\s*$", str(a))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return 0.0


def _map_optimizer(our_type: str) -> str:
    """Our OptimizerType -> their {adamw, prodigy, prodigy_plus_schedulefree}."""
    t = (our_type or "").lower()
    if t == "prodigyplusschedulefree" or "schedulefree" in t:
        return "prodigy_plus_schedulefree"
    if t.startswith("prodigy"):
        return "prodigy"
    # Their engine only ships adamw / prodigy / ppsf. Everything else (Lion,
    # DAdapt, SGD, CAME, AdaFactor, paged*) has no equivalent — fall back to
    # adamw, the safe DiT-LoRA default.
    return "adamw"


def _map_lr_scheduler(our_sched: str, their_opt: str) -> str:
    """Our LrScheduler -> their {none, cosine, cosine_with_restart}.

    Prodigy family fixes a constant LR in their engine (validator forbids a
    scheduler), so force `none` there.
    """
    if their_opt in ("prodigy", "prodigy_plus_schedulefree"):
        return "none"
    s = (our_sched or "").lower()
    if s == "cosine":
        return "cosine"
    if s in ("cosine_with_restarts", "cosine_with_restart"):
        return "cosine_with_restart"
    return "none"


def _map_timestep_sampling(our_ts: Optional[str]) -> str:
    """Our anima timestep_sampling -> their Literal set.

    Their options: logit_normal / uniform / logit_normal_low / mode /
    mixed_uniform_low / mixed_uniform_logit. Our vocabulary (shift / sigmoid /
    uniform / logit_normal / mode) only partially overlaps; map the ones that
    exist and default the rest to logit_normal (SD3/Anima default).
    """
    s = (our_ts or "").lower()
    if s in ("logit_normal", "uniform", "logit_normal_low", "mode",
             "mixed_uniform_low", "mixed_uniform_logit"):
        return s
    return "logit_normal"


def _map_mixed_precision(mp: str) -> str:
    s = (mp or "").lower()
    return s if s in ("bf16", "fp16", "no") else "bf16"


def _map_attention_backend(cfg: TrainConfig) -> str:
    """Their attention_backend: none(SDPA) / xformers / flash_attn."""
    tr = cfg.training
    if tr.xformers:
        return "xformers"
    if tr.sdpa or tr.mem_eff_attn:
        return "none"
    return "flash_attn"


def _map_loss(cfg: TrainConfig) -> Tuple[str, str]:
    """(loss_type, loss_weighting) in their vocabulary.

    The dedicated `cfg.anima_lokr.loss_weighting` wins when set to anything
    other than the default "none" — it carries detail_inv_t / cosmap which
    have no kohya equivalent. Otherwise fall back to min_snr when the user
    has a non-zero kohya min_snr_gamma.
    """
    lt = "huber" if cfg.training.loss_type in ("huber", "smooth_l1") else "mse"
    al = cfg.anima_lokr.loss_weighting
    if al and al != "none":
        return lt, al
    lw = "min_snr" if (cfg.training.min_snr_gamma or 0) > 0 else "none"
    return lt, lw


# ---------------------------------------------------------------------------
# Dataset: build an N_repeat symlink tree their parser understands
# ---------------------------------------------------------------------------
def _build_dataset_dir(cfg: TrainConfig, workdir: Path) -> str:
    """Materialize `<workdir>/anima_dataset/<repeats>_<name>` -> subset.image_dir.

    Their `AnimaDataset` derives per-folder repeats from the `N_` prefix, so we
    encode our explicit `num_repeats` into the folder name. Symlinks keep it
    cheap and avoid copying images. If symlinks are unavailable, fall back to
    the common parent of the subsets (repeats then come from existing folder
    names, if any).
    """
    subsets = cfg.dataset.subsets
    if not subsets:
        # Nothing to lay out — hand their loader the configured dataset root.
        return cfg.paths.dataset_root or "./dataset"

    root = workdir / "anima_dataset"
    try:
        root.mkdir(parents=True, exist_ok=True)
        # Clear stale symlinks from a previous build in the same workdir. We
        # only ever create symlinks here, so anything else (a stray real
        # directory dropped in by the user) we leave alone.
        for child in root.iterdir():
            if child.is_symlink():
                child.unlink()
        used: Dict[str, int] = {}
        for sub in subsets:
            src = Path(sub.image_dir)
            base = _safe_name(src.name or "concept")
            # Avoid collisions when two subsets share a leaf name.
            n = used.get(base, 0)
            used[base] = n + 1
            leaf = f"{max(1, sub.num_repeats)}_{base}" + (f"_{n}" if n else "")
            link = root / leaf
            link.symlink_to(src, target_is_directory=True)
        return str(root)
    except OSError:
        # Symlinks blocked (some Windows setups): fall back to the shared parent.
        parents = {str(Path(s.image_dir).parent) for s in subsets}
        if len(parents) == 1:
            return next(iter(parents))
        return cfg.paths.dataset_root or str(Path(subsets[0].image_dir).parent)


# ---------------------------------------------------------------------------
# Config translation: our TrainConfig -> their TrainingConfig dict
# ---------------------------------------------------------------------------
def build_studio_config(cfg: TrainConfig, workdir: Path) -> Dict[str, Any]:
    """Produce a dict keyed by AnimaLoraStudio's TrainingConfig field names.

    Only keys that exist in their schema are emitted (their loader ignores
    unknown keys, but staying within the schema keeps intent obvious).
    """
    net = cfg.network
    opt = cfg.optimizer
    tr = cfg.training
    sm = cfg.samples

    their_opt = _map_optimizer(opt.optimizer_type)
    loss_type, loss_weighting = _map_loss(cfg)

    # LoKr factor: their schema requires >= 2; our default sentinel is -1.
    factor = net.factor if (net.factor and net.factor >= 2) else 8

    data_dir = _build_dataset_dir(cfg, workdir)

    out: Dict[str, Any] = {
        # --- model paths ---
        "transformer_path": cfg.model.pretrained_model_name_or_path or "",
        "vae_path": cfg.model.anima_vae or "",
        # Their loader wants a Qwen *directory*; forwarded verbatim.
        "text_encoder_path": cfg.model.anima_qwen3 or "",
        "t5_tokenizer_path": cfg.model.anima_t5_tokenizer_path or "",

        # --- dataset ---
        "data_dir": data_dir,
        "resolution": cfg.dataset.resolution,
        "shuffle_caption": any(s.shuffle_caption for s in cfg.dataset.subsets),
        "keep_tokens": (cfg.dataset.subsets[0].keep_tokens if cfg.dataset.subsets else 0),
        "flip_augment": cfg.dataset.flip_aug,
        "tag_dropout": cfg.dataset.caption_tag_dropout_rate or 0.0,
        "cache_latents": cfg.dataset.cache_latents,

        # --- LoRA / LoKr ---
        "lora_type": "lokr",
        "lora_rank": net.network_dim,
        "lora_alpha": net.network_alpha,
        "lokr_factor": factor,
        # DoRA + rs-LoRA flags forwarded verbatim to AnimaLoraStudio. Their
        # `AnimaLycorisAdapter` passes these straight into `LycorisNetwork`.
        "lora_dora": bool(net.weight_decompose),
        "lora_rs": bool(net.rs_lora),
        "lora_dropout": net.dropout or 0.0,
        "lora_rank_dropout": net.rank_dropout or 0.0,
        "lora_module_dropout": net.module_dropout or 0.0,

        # --- training ---
        "epochs": tr.max_train_epochs,
        "max_steps": tr.max_train_steps or 0,
        "batch_size": tr.train_batch_size,
        "grad_accum": tr.gradient_accumulation_steps,
        "grad_checkpoint": tr.gradient_checkpointing,
        "learning_rate": opt.learning_rate,
        "optimizer_type": their_opt,
        "lr_scheduler": _map_lr_scheduler(opt.lr_scheduler, their_opt),
        "weight_decay": _opt_weight_decay(opt.optimizer_args),
        "grad_clip_max_norm": opt.max_grad_norm if opt.max_grad_norm is not None else 1.0,
        "mixed_precision": _map_mixed_precision(tr.mixed_precision),
        "attention_backend": _map_attention_backend(cfg),
        "num_workers": cfg.dataset.max_data_loader_n_workers or 0,

        # --- noise / timesteps / loss ---
        "noise_offset": float(tr.noise_offset or 0.0),
        # pyramid noise has no kohya field on the *AnimaLoraStudio* side
        # except via training.multires_noise_iterations/discount, so forward.
        "pyramid_noise_iters": int(tr.multires_noise_iterations or 0),
        "pyramid_noise_discount": float(
            tr.multires_noise_discount if tr.multires_noise_discount is not None else 0.35
        ),
        "timestep_sampling": _map_timestep_sampling(tr.timestep_sampling),
        "timestep_shift": float(tr.discrete_flow_shift or 3.0),
        "loss_type": loss_type,
        "loss_weighting": loss_weighting,

        # --- output ---
        "output_dir": cfg.paths.output_root or "./output",
        "output_name": cfg.project.output_name,
        "save_every_epochs": tr.save_every_n_epochs or 0,
        "seed": tr.seed,

        # CLI subprocess: plain throttled logging (not a tty).
        "no_progress": True,
    }

    if loss_type == "huber":
        out["huber_c"] = tr.huber_c

    # ---- AnimaLoraStudio-only knobs from the dedicated section --------------
    al = cfg.anima_lokr

    if loss_weighting == "min_snr":
        out["min_snr_gamma"] = tr.min_snr_gamma
    elif loss_weighting == "detail_inv_t":
        out["detail_inv_t_min"] = al.detail_inv_t_min
        out["detail_inv_t_max"] = al.detail_inv_t_max
    if loss_weighting != "none":
        out["weight_cap_ratio"] = al.weight_cap_ratio

    # PPSF optimizer block — emitted only when that optimizer is selected.
    if their_opt == "prodigy_plus_schedulefree":
        out.update({
            "ppsf_d_coef": al.ppsf_d_coef,
            "ppsf_prodigy_steps": al.ppsf_prodigy_steps,
            "ppsf_beta1": al.ppsf_beta1,
            "ppsf_beta2": al.ppsf_beta2,
            "ppsf_split_groups": al.ppsf_split_groups,
            "ppsf_split_groups_mean": al.ppsf_split_groups_mean,
            "ppsf_use_speed": al.ppsf_use_speed,
            "ppsf_fused_back_pass": al.ppsf_fused_back_pass,
            "ppsf_use_stableadamw": al.ppsf_use_stableadamw,
        })
    if their_opt == "prodigy":
        out["prodigy_d_coef"] = al.prodigy_d_coef
        out["prodigy_safeguard_warmup"] = al.prodigy_safeguard_warmup

    # InfoNoise — adaptive timestep sampling, full block only when enabled.
    if al.infonoise_enabled:
        out.update({
            "infonoise_enabled": True,
            "infonoise_K": al.infonoise_K,
            "infonoise_N_warm": al.infonoise_N_warm,
            "infonoise_M": al.infonoise_M,
            "infonoise_B": al.infonoise_B,
            "infonoise_beta": al.infonoise_beta,
            "infonoise_N_min": al.infonoise_N_min,
        })

    # Timestep extras (their dialect overrides the kohya one when set non-default).
    if al.timestep_sampling and al.timestep_sampling != "logit_normal":
        out["timestep_sampling"] = al.timestep_sampling
    if al.timestep_shift and al.timestep_shift != 3.0:
        out["timestep_shift"] = al.timestep_shift
    if al.timestep_mix_low_prob > 0:
        out["timestep_mix_low_prob"] = al.timestep_mix_low_prob
    if al.timestep_schedule_shift and al.timestep_schedule_shift != 1.0:
        out["timestep_schedule_shift"] = al.timestep_schedule_shift

    # Per-layer rank override (regex -> rank).
    if al.lora_reg_dims:
        out["lora_reg_dims"] = dict(al.lora_reg_dims)

    # Performance.
    if al.kv_trim:
        out["kv_trim"] = True

    # Regularization dataset.
    if al.reg_data_dir:
        out["reg_data_dir"] = al.reg_data_dir
        if al.reg_caption:
            out["reg_caption"] = al.reg_caption
        out["reg_weight"] = al.reg_weight

    # Resume.
    if net.network_weights:
        out["resume_lora"] = net.network_weights
    if tr.resume:
        out["resume_state"] = tr.resume

    # --- sampling ---
    if sm.enable and sm.prompts:
        first = sm.prompts[0]
        out["sample_every"] = sm.every_n_epochs or 0
        out["sample_steps"] = sm.every_n_steps or 0
        out["sample_infer_steps"] = first.steps
        out["sample_cfg_scale"] = first.scale
        out["sample_negative_prompt"] = first.negative
        out["sample_width"] = first.width
        out["sample_height"] = first.height
        out["sample_seed"] = first.seed
        prompts = [p.text for p in sm.prompts]
        if len(prompts) == 1:
            out["sample_prompt"] = prompts[0]
        else:
            out["sample_prompts"] = prompts
    else:
        out["sample_every"] = 0
        out["sample_steps"] = 0

    return out


# ---------------------------------------------------------------------------
# Launch argv
# ---------------------------------------------------------------------------
def build_anima_lokr_command(cfg: TrainConfig, workdir: Path) -> Tuple[List[str], dict]:
    """Build (argv, env) launching AnimaLoraStudio's trainer for Anima + LoKr."""
    workdir.mkdir(parents=True, exist_ok=True)

    studio_dir = cfg.paths.anima_studio_dir or "/content/AnimaLoraStudio"
    train_script = os.path.join(studio_dir, "runtime", "anima_train.py")

    studio_cfg = build_studio_config(cfg, workdir)
    cfg_path = workdir / "anima_studio_config.yaml"
    # JSON is valid YAML; their bootstrap reads it with yaml.safe_load, so we
    # avoid a PyYAML dependency on our side.
    cfg_path.write_text(json.dumps(studio_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    monitor_state = workdir / "monitor_state.json"

    argv: List[str] = [
        # Match the interpreter the backend is running under so torch /
        # lycoris / diffusers from our venv are visible. `python` can resolve
        # to a system interpreter on Colab and break imports.
        sys.executable, train_script,
        "--config", str(cfg_path),
        "--auto-install",
        "--no-live-curve",
        "--monitor-state-file", str(monitor_state),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Their bare-script launch self-injects repo root into sys.path; export it
    # too so `import studio` / `import training` resolve regardless of cwd.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = studio_dir + (os.pathsep + existing_pp if existing_pp else "")
    return argv, env
