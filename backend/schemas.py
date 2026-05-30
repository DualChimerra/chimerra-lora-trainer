"""Pydantic schemas for the training UI.

Field names mirror sd-scripts CLI arguments 1:1 wherever possible so that
config_builder can pass them through without name translation.
"""
from __future__ import annotations

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_serializer


# ---------------------------------------------------------------------------
# Paths (Colab side)
# ---------------------------------------------------------------------------
class Paths(BaseModel):
    """Root directories selected in the Colab setup cell."""
    dataset_root: str = ""
    base_model_root: str = ""
    output_root: str = ""
    samples_root: str = ""
    sd_scripts_dir: str = "/content/sd-scripts"
    # AnimaLoraStudio checkout. Used ONLY for the isolated Anima + LyCORIS LoKr
    # path (its own training engine, not kohya). Ignored by every other combo.
    anima_studio_dir: str = "/content/AnimaLoraStudio"


# ---------------------------------------------------------------------------
# Model / architecture
# ---------------------------------------------------------------------------
ArchKind = Literal["sdxl", "anima"]
BaseModelPreset = Literal["noobai", "illustrious", "anima", "custom"]


class ModelSection(BaseModel):
    arch: ArchKind = "sdxl"
    preset: BaseModelPreset = "illustrious"

    # SDXL / NoobAI / Illustrious
    pretrained_model_name_or_path: str = ""
    vae: Optional[str] = None
    v_parameterization: bool = False
    clip_skip: int = 2

    # Anima-only paths
    anima_qwen3: Optional[str] = None
    anima_t5_tokenizer_path: Optional[str] = None
    anima_vae: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetSubset(BaseModel):
    """One sub-folder inside the dataset root, e.g. `10_concept`."""
    image_dir: str
    num_repeats: int = 10
    caption_extension: str = ".txt"
    class_tokens: Optional[str] = None
    shuffle_caption: bool = True
    keep_tokens: int = 1
    keep_tokens_separator: Optional[str] = None
    # auto-detected
    num_images: int = 0


class DatasetSection(BaseModel):
    resolution: int = 1024
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    # ⚠️ Cannot be True together with shuffle_caption=True or training the
    # text encoder (text_encoder_lr set / not network_train_unet_only).
    cache_text_encoder_outputs: bool = False
    cache_text_encoder_outputs_to_disk: bool = False
    color_aug: bool = False
    flip_aug: bool = False
    random_crop: bool = False
    max_token_length: int = 225

    # Data loading performance — kohya defaults to 0 workers (single-threaded),
    # which is a major cause of slow epochs on Colab. 8 + persistent is the
    # standard "fast" config.
    max_data_loader_n_workers: int = 8
    persistent_data_loader_workers: bool = True

    # Caption augmentation (global). Optional because the Num inputs in the UI
    # send null when the user clears the field — semantically "off". The
    # config_builder already short-circuits None for these via `and`-checks, so
    # accepting None lets the form save without a 422 when fields are blanked.
    caption_dropout_rate: Optional[float] = 0.0
    caption_tag_dropout_rate: Optional[float] = 0.0
    caption_dropout_every_n_epochs: Optional[int] = 0
    token_warmup_min: Optional[int] = 1
    token_warmup_step: Optional[int] = 0
    weighted_captions: bool = False

    subsets: List[DatasetSubset] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Network (LoRA / LyCORIS / LoKr)
# ---------------------------------------------------------------------------
NetworkKind = Literal["lora", "locon", "loha", "lokr", "dylora", "ia3"]


class NetworkSection(BaseModel):
    kind: NetworkKind = "lora"
    network_dim: int = 32
    network_alpha: float = 16.0

    # LyCORIS-specific
    conv_dim: Optional[int] = 16
    conv_alpha: Optional[float] = 8.0
    preset: str = "full"  # full, attn-mlp, attn-only, full-lin
    factor: int = -1  # LoKr only
    decompose_both: bool = False  # LoKr: also decompose w1 (second matrix)
    # DoRA — direction/magnitude decomposition of the LoRA update. Stabilises
    # training and often beats vanilla LoRA at the same rank.
    weight_decompose: bool = False
    # rs-LoRA — scale = alpha/sqrt(r) instead of alpha/r; recommended when
    # rank > 32 because the standard scaling underweights the update.
    rs_lora: bool = False
    use_tucker: bool = False
    use_scalar: bool = False
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    dropout: float = 0.0
    rank_dropout_scale: bool = False

    # Common
    network_train_unet_only: bool = False
    network_train_text_encoder_only: bool = False
    network_dropout: float = 0.0
    scale_weight_norms: float = 0.0

    # Continue training from existing LoRA weights
    network_weights: Optional[str] = None
    dim_from_weights: bool = False
    base_weights: List[str] = Field(default_factory=list)
    base_weights_multiplier: List[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Optimizer / training schedule
# ---------------------------------------------------------------------------
OptimizerType = Literal[
    "AdamW", "AdamW8bit", "Lion", "Lion8bit",
    "Prodigy", "DAdaptation", "DAdaptAdam", "DAdaptLion",
    "SGDNesterov", "SGDNesterov8bit",
    "AdaFactor", "PagedAdamW8bit", "PagedLion8bit",
    "pytorch_optimizer.CAME",
    # AnimaLoraStudio-only (Anima + LoKr path). Selecting this with kohya will
    # error out — the UI gates it to (arch=anima, kind=lokr).
    "ProdigyPlusScheduleFree",
]
LrScheduler = Literal[
    "constant", "constant_with_warmup", "linear",
    "cosine", "cosine_with_restarts", "polynomial",
    "adafactor", "warmup_stable_decay", "piecewise_constant",
]
MixedPrecision = Literal["no", "fp16", "bf16"]


class PiecewisePoint(BaseModel):
    at: float = 0.0    # fraction of total steps where this level begins (0..1)
    mult: float = 1.0  # LR multiplier for this level (0..1)


class OptimizerSection(BaseModel):
    optimizer_type: OptimizerType = "AdamW8bit"
    optimizer_args: List[str] = Field(default_factory=list)  # ["weight_decay=0.1", ...]
    learning_rate: float = 1e-4
    unet_lr: Optional[float] = None
    text_encoder_lr: Optional[float] = None
    lr_scheduler: LrScheduler = "cosine"
    lr_scheduler_args: List[str] = Field(default_factory=list)
    lr_warmup_steps: int = 0
    # WSD only: length of the final decay phase. Float in (0,1] = fraction of
    # total steps (0.2 = last 20%); int = absolute steps. Required by kohya for
    # warmup_stable_decay, ignored by other schedulers.
    lr_decay_steps: Optional[float] = None
    # piecewise_constant only: user-drawn LR levels. Each point = (at, mult);
    # the level holds `mult` from its `at` until the next point's `at`. Only
    # consumed when lr_scheduler == "piecewise_constant".
    lr_piecewise: List[PiecewisePoint] = Field(default_factory=lambda: [
        PiecewisePoint(at=0.0, mult=1.0),
        PiecewisePoint(at=0.5, mult=0.3),
        PiecewisePoint(at=0.8, mult=0.1),
    ])
    lr_scheduler_num_cycles: int = 1
    lr_scheduler_power: float = 1.0
    max_grad_norm: float = 1.0


class TrainingSection(BaseModel):
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_train_epochs: int = 10
    max_train_steps: int = 0  # 0 → derive from epochs
    save_every_n_epochs: int = 1
    save_last_n_epochs: int = 0
    seed: int = 42
    mixed_precision: MixedPrecision = "bf16"
    save_precision: MixedPrecision = "fp16"
    gradient_checkpointing: bool = True
    xformers: bool = False
    sdpa: bool = True
    mem_eff_attn: bool = False
    full_bf16: bool = False
    full_fp16: bool = False
    lowram: bool = False
    highvram: bool = False
    fused_backward_pass: bool = False
    vae_batch_size: int = 0  # 0 → kohya default; raise for speed if VRAM allows

    # loss / timesteps
    loss_type: Literal["l2", "l1", "huber", "smooth_l1"] = "l2"
    huber_schedule: Literal["snr", "exponential", "constant"] = "snr"
    huber_c: float = 0.1
    prior_loss_weight: float = 1.0
    min_timestep: int = 0
    max_timestep: int = 1000

    # state save / resume
    save_state: bool = False
    save_state_on_train_end: bool = False
    resume: Optional[str] = None

    # noise / loss extras
    min_snr_gamma: Optional[float] = 5.0
    noise_offset: Optional[float] = 0.0357
    adaptive_noise_scale: Optional[float] = None
    noise_offset_random_strength: bool = False
    multires_noise_iterations: Optional[int] = None
    multires_noise_discount: Optional[float] = None
    ip_noise_gamma: Optional[float] = None
    ip_noise_gamma_random_strength: bool = False
    debiased_estimation_loss: bool = False
    zero_terminal_snr: bool = False

    # Anima FlowMatch extras
    weighting_scheme: Optional[str] = "logit_normal"
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    mode_scale: Optional[float] = 1.29
    timestep_sampling: Optional[str] = "shift"
    sigmoid_scale: Optional[float] = 1.0
    discrete_flow_shift: Optional[float] = 3.0


# ---------------------------------------------------------------------------
# Anima + LyCORIS LoKr only — fields specific to AnimaLoraStudio's engine.
# Inert when training runs through kohya (every non-Anima-LoKr combo).
# ---------------------------------------------------------------------------
LossWeighting = Literal["none", "min_snr", "detail_inv_t", "cosmap"]
AnimaTimestepSampling = Literal[
    "logit_normal", "uniform", "logit_normal_low",
    "mode", "mixed_uniform_low", "mixed_uniform_logit",
]


class AnimaLokrSection(BaseModel):
    # ---- ProdigyPlusScheduleFree (PPSF) optimizer extras --------------------
    # Active only when optimizer.optimizer_type == "ProdigyPlusScheduleFree".
    ppsf_d_coef: float = 1.0
    ppsf_prodigy_steps: int = 0
    ppsf_beta1: float = 0.9
    ppsf_beta2: float = 0.99
    ppsf_split_groups: bool = True
    ppsf_split_groups_mean: bool = False
    ppsf_use_speed: bool = False
    ppsf_fused_back_pass: bool = False
    ppsf_use_stableadamw: bool = True

    # ---- Prodigy extras (regular Prodigy via AnimaLoraStudio) ---------------
    prodigy_d_coef: float = 1.0
    prodigy_safeguard_warmup: bool = True

    # ---- InfoNoise (adaptive timestep sampling) -----------------------------
    infonoise_enabled: bool = False
    infonoise_K: int = 64
    infonoise_N_warm: int = 0
    infonoise_M: int = 100
    infonoise_B: int = 256
    infonoise_beta: float = 0.9
    infonoise_N_min: int = 50

    # ---- Loss weighting -----------------------------------------------------
    # When != "none" this overrides the kohya-style min_snr_gamma plumbing.
    loss_weighting: LossWeighting = "none"
    weight_cap_ratio: float = 0.0  # 0 = disabled
    detail_inv_t_min: float = 1.0
    detail_inv_t_max: float = 5.0

    # ---- Timestep sampling (AnimaLoraStudio dialect) ------------------------
    # Their schema has its own set of values, separate from our kohya FlowMatch
    # `training.timestep_sampling`. The mapper prefers this one when set.
    timestep_sampling: AnimaTimestepSampling = "logit_normal"
    timestep_shift: float = 3.0
    timestep_mix_low_prob: float = 0.0  # mixed_* modes only
    timestep_schedule_shift: float = 1.0  # post-sample σ schedule shift

    # ---- LoKr per-layer rank override --------------------------------------
    # regex -> rank, e.g. {"lora_unet_.*double.*": 16}
    lora_reg_dims: Optional[Dict[str, int]] = None

    # ---- Performance --------------------------------------------------------
    kv_trim: bool = False  # Cross-attn KV trim to nearest bucket

    # ---- Regularization dataset --------------------------------------------
    reg_data_dir: Optional[str] = None
    reg_caption: Optional[str] = None
    reg_weight: float = 1.0


# ---------------------------------------------------------------------------
# Sample generation during training
# ---------------------------------------------------------------------------
SampleSampler = Literal[
    "ddim", "pndm", "lms", "euler", "euler_a",
    "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++",
    "dpmsingle", "k_lms", "k_euler", "k_euler_a",
    "k_dpm_2", "k_dpm_2_a",
]


class SamplePrompt(BaseModel):
    text: str
    negative: str = "(worst quality, low quality:1.2)"
    width: int = 1024
    height: int = 1024
    steps: int = 24
    scale: float = 5.0
    seed: int = 42
    sampler: SampleSampler = "euler_a"


class SamplesSection(BaseModel):
    enable: bool = True
    every_n_epochs: int = 1
    every_n_steps: int = 0
    sampler: SampleSampler = "euler_a"
    prompts: List[SamplePrompt] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Project (output naming)
# ---------------------------------------------------------------------------
class ProjectSection(BaseModel):
    output_name: str = "my-lora"
    save_model_as: Literal["safetensors", "ckpt", "pt"] = "safetensors"
    log_with: Literal["none", "tensorboard", "wandb"] = "tensorboard"
    log_prefix: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregate config
# ---------------------------------------------------------------------------
class TrainConfig(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    project: ProjectSection = Field(default_factory=ProjectSection)
    model: ModelSection = Field(default_factory=ModelSection)
    dataset: DatasetSection = Field(default_factory=DatasetSection)
    network: NetworkSection = Field(default_factory=NetworkSection)
    optimizer: OptimizerSection = Field(default_factory=OptimizerSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    samples: SamplesSection = Field(default_factory=SamplesSection)
    anima_lokr: AnimaLokrSection = Field(default_factory=AnimaLokrSection)

    # arbitrary additional kohya args the user might add via the "advanced" pane
    extra_args: Dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _strip_arch_irrelevant(self, handler):
        # Strip fields that don't apply to the selected architecture so the
        # exported/saved JSON matches what the UI shows. Anima is DiT (no conv
        # layers) and uses Qwen3 TE (no 75-token CLIP limit). Using a wrap
        # serializer ensures this also fires when TrainConfig is nested inside
        # another model (e.g. Preset.config).
        data = handler(self)
        if self.model.arch == "anima":
            net = data.get("network") or {}
            net.pop("conv_dim", None)
            net.pop("conv_alpha", None)
            ds = data.get("dataset") or {}
            ds.pop("max_token_length", None)
        # `anima_lokr` is only consumed when (arch=anima, kind=lokr). Strip it
        # from every other config so saved JSON / presets stay tidy.
        is_anima_lokr = self.model.arch == "anima" and self.network.kind == "lokr"
        if not is_anima_lokr:
            data.pop("anima_lokr", None)
        else:
            # Anima+LoKr trains via AnimaLoraStudio, not kohya. A pile of
            # kohya-only knobs never reach that engine — keeping them in the
            # saved JSON is misleading (e.g. scale_weight_norms looks like an
            # anti-overcook clamp but does nothing here; the LR-schedule fields
            # are dead because PPSF/Prodigy force a constant LR). Drop them so
            # the persisted config honestly reflects what the engine receives.
            net = data.get("network") or {}
            for k in ("scale_weight_norms", "network_dropout", "use_tucker",
                      "use_scalar", "rank_dropout_scale", "decompose_both",
                      "dim_from_weights", "base_weights", "base_weights_multiplier",
                      "network_train_text_encoder_only", "preset"):
                net.pop(k, None)
            opt = data.get("optimizer") or {}
            for k in ("lr_piecewise", "lr_decay_steps", "lr_warmup_steps",
                      "lr_scheduler_args", "lr_scheduler_num_cycles",
                      "lr_scheduler_power", "unet_lr", "text_encoder_lr"):
                opt.pop(k, None)
        return data


# ---------------------------------------------------------------------------
# Runtime status pushed to UI over WebSocket
# ---------------------------------------------------------------------------
class TrainStatus(BaseModel):
    state: Literal["idle", "starting", "running", "stopping", "finished", "error"] = "idle"
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    lr: Optional[float] = None
    eta_seconds: Optional[int] = None
    started_at: Optional[float] = None
    pid: Optional[int] = None
    message: Optional[str] = None


class Preset(BaseModel):
    name: str
    description: str = ""
    created_at: float
    config: TrainConfig
