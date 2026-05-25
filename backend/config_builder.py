"""Build kohya-ss/sd-scripts compatible TOML configs + CLI commands.

The intent: every value the user touches in the UI is forwarded to sd-scripts
under its EXACT upstream argument name. We never reinterpret training
semantics — we only assemble what sd-scripts already accepts.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

from .schemas import TrainConfig, NetworkSection, SamplesSection


# ---------------------------------------------------------------------------
# Network module / args
# ---------------------------------------------------------------------------
def _network_module(net: NetworkSection) -> Tuple[str, List[str]]:
    """Map (kind) -> (network_module, network_args[])."""
    if net.kind == "lora":
        return ("networks.lora", [])

    # Everything else lives in LyCORIS
    args = [
        f"preset={net.preset}",
        f"algo={net.kind}",  # locon, loha, lokr, dylora, ia3
    ]
    if net.conv_dim is not None:
        args.append(f"conv_dim={net.conv_dim}")
    if net.conv_alpha is not None:
        args.append(f"conv_alpha={net.conv_alpha}")
    if net.kind == "lokr":
        args.append(f"factor={net.factor}")
        if net.decompose_both:
            args.append("decompose_both=True")
    if net.use_tucker:
        args.append("use_tucker=True")
    if net.use_scalar:
        args.append("use_scalar=True")
    if net.rank_dropout and net.rank_dropout > 0:
        args.append(f"rank_dropout={net.rank_dropout}")
    if net.module_dropout and net.module_dropout > 0:
        args.append(f"module_dropout={net.module_dropout}")
    if net.dropout and net.dropout > 0:
        args.append(f"dropout={net.dropout}")
    if net.rank_dropout_scale:
        args.append("rank_dropout_scale=True")
    return ("lycoris.kohya", args)


# ---------------------------------------------------------------------------
# Dataset TOML — kohya's `--dataset_config` format
# ---------------------------------------------------------------------------
def write_dataset_toml(cfg: TrainConfig, dest: Path) -> Path:
    """Write the dataset_config TOML expected by sd-scripts."""
    ds = cfg.dataset
    lines: List[str] = []
    lines.append("[general]")
    lines.append(f"enable_bucket = {str(ds.enable_bucket).lower()}")
    lines.append(f"shuffle_caption = {str(any(s.shuffle_caption for s in ds.subsets) or False).lower()}")
    lines.append(f"caption_extension = \".txt\"")
    lines.append(f"keep_tokens = 0")
    lines.append("")
    lines.append("[[datasets]]")
    lines.append(f"resolution = {ds.resolution}")
    lines.append(f"batch_size = {cfg.training.train_batch_size}")
    lines.append(f"min_bucket_reso = {ds.min_bucket_reso}")
    lines.append(f"max_bucket_reso = {ds.max_bucket_reso}")
    lines.append(f"bucket_reso_steps = {ds.bucket_reso_steps}")
    lines.append(f"bucket_no_upscale = {str(ds.bucket_no_upscale).lower()}")
    lines.append("")
    for sub in ds.subsets:
        lines.append("  [[datasets.subsets]]")
        lines.append(f"  image_dir = \"{sub.image_dir}\"")
        lines.append(f"  num_repeats = {sub.num_repeats}")
        lines.append(f"  caption_extension = \"{sub.caption_extension}\"")
        if sub.class_tokens:
            esc = sub.class_tokens.replace('"', '\\"')
            lines.append(f"  class_tokens = \"{esc}\"")
        lines.append(f"  shuffle_caption = {str(sub.shuffle_caption).lower()}")
        lines.append(f"  keep_tokens = {sub.keep_tokens}")
        if sub.keep_tokens_separator:
            esc = sub.keep_tokens_separator.replace('"', '\\"')
            lines.append(f"  keep_tokens_separator = \"{esc}\"")
        lines.append("")

    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Sample prompts file — kohya format
# ---------------------------------------------------------------------------
def write_sample_prompts(samples: SamplesSection, dest: Path) -> Optional[Path]:
    if not samples.enable or not samples.prompts:
        return None
    out: List[str] = []
    for p in samples.prompts:
        line = (
            f"{p.text} "
            f"--n {p.negative} "
            f"--w {p.width} --h {p.height} "
            f"--d {p.seed} --l {p.scale} --s {p.steps}"
        )
        out.append(line)
    dest.write_text("\n".join(out), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Total steps (mirrors kohya's accounting; used for ETA & UI display)
# ---------------------------------------------------------------------------
def compute_total_steps(cfg: TrainConfig) -> int:
    ds = cfg.dataset
    tr = cfg.training
    if not ds.subsets:
        return 0
    weighted_images = sum(max(1, s.num_images) * max(1, s.num_repeats) for s in ds.subsets)
    bs = max(1, tr.train_batch_size) * max(1, tr.gradient_accumulation_steps)
    steps_per_epoch = max(1, weighted_images // bs)
    if tr.max_train_steps and tr.max_train_steps > 0:
        return tr.max_train_steps
    return steps_per_epoch * max(1, tr.max_train_epochs)


# ---------------------------------------------------------------------------
# Build the launch argv
# ---------------------------------------------------------------------------
def build_command(cfg: TrainConfig, workdir: Path) -> Tuple[List[str], dict]:
    """Build (argv, env) for launching kohya's network trainer.

    Two scripts are wired:
        - sdxl_train_network.py  (NoobAI / Illustrious / any SDXL fork)
        - anima_train_network.py (DiT + Qwen3 TE + Qwen-Image VAE)
    """
    workdir.mkdir(parents=True, exist_ok=True)
    dataset_toml = write_dataset_toml(cfg, workdir / "dataset.toml")
    sample_file = write_sample_prompts(cfg.samples, workdir / "samples.txt")

    sd = cfg.paths.sd_scripts_dir
    if cfg.model.arch == "anima":
        script = os.path.join(sd, "anima_train_network.py")
    else:
        script = os.path.join(sd, "sdxl_train_network.py")

    network_module, network_args = _network_module(cfg.network)

    argv: List[str] = ["accelerate", "launch", "--num_cpu_threads_per_process=2", script]

    def add(flag: str, value):
        if value is None:
            return
        if isinstance(value, bool):
            if value:
                argv.append(f"--{flag}")
            return
        argv.extend([f"--{flag}", str(value)])

    # --- paths / model ----------------------------------------------------
    add("pretrained_model_name_or_path", cfg.model.pretrained_model_name_or_path)
    if cfg.model.arch == "anima":
        add("qwen3", cfg.model.anima_qwen3)
        add("t5_tokenizer_path", cfg.model.anima_t5_tokenizer_path)
        add("vae", cfg.model.anima_vae)
    else:
        if cfg.model.vae:
            add("vae", cfg.model.vae)
        if cfg.model.v_parameterization:
            argv.append("--v_parameterization")
        add("clip_skip", cfg.model.clip_skip)

    add("dataset_config", str(dataset_toml))
    add("output_dir", cfg.paths.output_root)
    add("output_name", cfg.project.output_name)
    add("save_model_as", cfg.project.save_model_as)
    if cfg.project.log_with != "none":
        add("log_with", cfg.project.log_with)
        add("logging_dir", os.path.join(cfg.paths.output_root, "logs"))
    if cfg.project.log_prefix:
        add("log_prefix", cfg.project.log_prefix)

    # --- dataset cache flags ---------------------------------------------
    if cfg.dataset.cache_latents:
        argv.append("--cache_latents")
    if cfg.dataset.cache_latents_to_disk:
        argv.append("--cache_latents_to_disk")
    # cache_text_encoder_outputs is incompatible with shuffle_caption=True
    # AND with training the TE — silently drop it in those cases to avoid
    # the kohya assertion at startup.
    _any_shuffle = any(s.shuffle_caption for s in cfg.dataset.subsets)
    _trains_te = (
        not cfg.network.network_train_unet_only
        and (cfg.optimizer.text_encoder_lr is None or cfg.optimizer.text_encoder_lr > 0)
    )
    _cache_te_ok = cfg.dataset.cache_text_encoder_outputs and not _any_shuffle and not _trains_te
    if _cache_te_ok:
        argv.append("--cache_text_encoder_outputs")
        if cfg.dataset.cache_text_encoder_outputs_to_disk:
            argv.append("--cache_text_encoder_outputs_to_disk")
    add("max_token_length", cfg.dataset.max_token_length)
    if cfg.dataset.color_aug:
        argv.append("--color_aug")
    if cfg.dataset.flip_aug:
        argv.append("--flip_aug")
    if cfg.dataset.random_crop:
        argv.append("--random_crop")

    # --- network ----------------------------------------------------------
    add("network_module", network_module)
    add("network_dim", cfg.network.network_dim)
    add("network_alpha", cfg.network.network_alpha)
    if network_args:
        argv.append("--network_args")
        argv.extend(str(na) for na in network_args)
    if cfg.network.network_train_unet_only:
        argv.append("--network_train_unet_only")
    if cfg.network.network_train_text_encoder_only:
        argv.append("--network_train_text_encoder_only")
    if cfg.network.network_dropout and cfg.network.network_dropout > 0:
        add("network_dropout", cfg.network.network_dropout)
    if cfg.network.scale_weight_norms and cfg.network.scale_weight_norms > 0:
        add("scale_weight_norms", cfg.network.scale_weight_norms)

    # --- optimizer --------------------------------------------------------
    add("optimizer_type", cfg.optimizer.optimizer_type)
    if cfg.optimizer.optimizer_args:
        argv.append("--optimizer_args")
        argv.extend(str(oa) for oa in cfg.optimizer.optimizer_args)
    add("learning_rate", cfg.optimizer.learning_rate)
    add("unet_lr", cfg.optimizer.unet_lr)
    add("text_encoder_lr", cfg.optimizer.text_encoder_lr)
    add("lr_scheduler", cfg.optimizer.lr_scheduler)
    add("lr_warmup_steps", cfg.optimizer.lr_warmup_steps)
    add("lr_scheduler_num_cycles", cfg.optimizer.lr_scheduler_num_cycles)
    add("lr_scheduler_power", cfg.optimizer.lr_scheduler_power)

    # --- training ---------------------------------------------------------
    add("train_batch_size", cfg.training.train_batch_size)
    add("gradient_accumulation_steps", cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps and cfg.training.max_train_steps > 0:
        add("max_train_steps", cfg.training.max_train_steps)
    else:
        add("max_train_epochs", cfg.training.max_train_epochs)
    add("save_every_n_epochs", cfg.training.save_every_n_epochs)
    if cfg.training.save_last_n_epochs:
        add("save_last_n_epochs", cfg.training.save_last_n_epochs)
    add("seed", cfg.training.seed)
    add("mixed_precision", cfg.training.mixed_precision)
    add("save_precision", cfg.training.save_precision)
    if cfg.training.gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if cfg.training.xformers:
        argv.append("--xformers")
    if cfg.training.sdpa:
        argv.append("--sdpa")
    if cfg.training.full_bf16:
        argv.append("--full_bf16")
    if cfg.training.full_fp16:
        argv.append("--full_fp16")

    # noise / loss
    if cfg.training.min_snr_gamma is not None:
        add("min_snr_gamma", cfg.training.min_snr_gamma)
    if cfg.training.noise_offset:
        add("noise_offset", cfg.training.noise_offset)
    if cfg.training.adaptive_noise_scale is not None:
        add("adaptive_noise_scale", cfg.training.adaptive_noise_scale)
    if cfg.training.multires_noise_iterations:
        add("multires_noise_iterations", cfg.training.multires_noise_iterations)
        add("multires_noise_discount", cfg.training.multires_noise_discount)
    if cfg.training.ip_noise_gamma is not None:
        add("ip_noise_gamma", cfg.training.ip_noise_gamma)
    if cfg.training.debiased_estimation_loss:
        argv.append("--debiased_estimation_loss")
    if cfg.training.zero_terminal_snr:
        argv.append("--zero_terminal_snr")

    # Anima FlowMatch
    if cfg.model.arch == "anima":
        add("weighting_scheme", cfg.training.weighting_scheme)
        add("logit_mean", cfg.training.logit_mean)
        add("logit_std", cfg.training.logit_std)
        add("mode_scale", cfg.training.mode_scale)
        add("timestep_sampling", cfg.training.timestep_sampling)
        add("sigmoid_scale", cfg.training.sigmoid_scale)
        add("discrete_flow_shift", cfg.training.discrete_flow_shift)

    # samples
    if sample_file is not None:
        add("sample_prompts", str(sample_file))
        add("sample_sampler", cfg.samples.sampler)
        if cfg.samples.every_n_steps and cfg.samples.every_n_steps > 0:
            add("sample_every_n_steps", cfg.samples.every_n_steps)
        else:
            add("sample_every_n_epochs", cfg.samples.every_n_epochs)

    # extra advanced args (free-form)
    for k, v in (cfg.extra_args or {}).items():
        if v is None or v is False:
            continue
        if v is True:
            argv.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            # kohya nargs="*" flags (e.g. text_encoder_lr, lr_scheduler_args)
            # must be emitted once with all values, not repeated per value.
            if not v:
                continue
            argv.append(f"--{k}")
            argv.extend(str(x) for x in v)
        else:
            argv.extend([f"--{k}", str(v)])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return argv, env
