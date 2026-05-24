// Built-in dafault presets — fast/quality starting points per architecture.
// These are applied on top of the user's current config (deep-merged).

export const BUILTIN_PRESETS = {
    illustrious_lora: {
        label: 'Illustrious · LoRA · 1024',
        model: { arch: 'sdxl', preset: 'illustrious', clip_skip: 2 },
        network: { kind: 'lora', network_dim: 32, network_alpha: 16, network_train_unet_only: false },
        optimizer: { optimizer_type: 'AdamW8bit', learning_rate: 1e-4, unet_lr: 1e-4, text_encoder_lr: 5e-5, lr_scheduler: 'cosine' },
        training: { mixed_precision: 'bf16', save_precision: 'fp16', gradient_checkpointing: true, sdpa: true,
                    min_snr_gamma: 5.0, noise_offset: 0.0357 },
        dataset: { resolution: 1024, cache_latents: true, cache_text_encoder_outputs: true },
    },
    noobai_lora: {
        label: 'NoobAI · LoRA · 1024',
        model: { arch: 'sdxl', preset: 'noobai', clip_skip: 2 },
        network: { kind: 'lora', network_dim: 32, network_alpha: 16 },
        optimizer: { optimizer_type: 'AdamW8bit', learning_rate: 1e-4, lr_scheduler: 'cosine' },
        training: { mixed_precision: 'bf16', save_precision: 'fp16', gradient_checkpointing: true, sdpa: true,
                    min_snr_gamma: 5.0, noise_offset: 0.0357 },
        dataset: { resolution: 1024, cache_latents: true, cache_text_encoder_outputs: true },
    },
    illustrious_lokr: {
        label: 'Illustrious · LoKr · low-rank style',
        model: { arch: 'sdxl', preset: 'illustrious' },
        network: { kind: 'lokr', network_dim: 10000, network_alpha: 1, factor: 16,
                   conv_dim: 10000, conv_alpha: 1, preset: 'full', use_tucker: true },
        optimizer: { optimizer_type: 'Prodigy', learning_rate: 1.0, lr_scheduler: 'cosine',
                     optimizer_args: ['decouple=True', 'weight_decay=0.01', 'd_coef=2', 'use_bias_correction=True', 'safeguard_warmup=True'] },
        training: { mixed_precision: 'bf16', save_precision: 'fp16', gradient_checkpointing: true, sdpa: true,
                    min_snr_gamma: 5.0, noise_offset: 0.0357 },
        dataset: { resolution: 1024, cache_latents: true, cache_text_encoder_outputs: true },
    },
    anima_lora: {
        label: 'Anima · LoRA · DiT FlowMatch',
        model: { arch: 'anima', preset: 'anima' },
        network: { kind: 'lora', network_dim: 16, network_alpha: 16 },
        optimizer: { optimizer_type: 'AdamW8bit', learning_rate: 1e-4, lr_scheduler: 'constant' },
        training: { mixed_precision: 'bf16', save_precision: 'bf16', gradient_checkpointing: true, sdpa: true,
                    weighting_scheme: 'logit_normal', logit_mean: 0.0, logit_std: 1.0,
                    timestep_sampling: 'shift', discrete_flow_shift: 3.0,
                    min_snr_gamma: null, noise_offset: null },
        dataset: { resolution: 1024, cache_latents: true, cache_text_encoder_outputs: true },
    },
};

// Deep-merge `patch` into `target` (returns new object).
export function deepMerge(target, patch) {
    if (Array.isArray(patch)) return patch.slice();
    if (patch && typeof patch === 'object') {
        const out = { ...(target || {}) };
        for (const k of Object.keys(patch)) {
            const v = patch[k];
            if (v && typeof v === 'object' && !Array.isArray(v))
                out[k] = deepMerge(target?.[k], v);
            else
                out[k] = v;
        }
        return out;
    }
    return patch;
}
