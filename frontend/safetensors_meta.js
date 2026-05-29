// Read sd-scripts / kohya training metadata from a .safetensors LoRA and turn
// it into a partial TrainConfig patch (deep-merged over the current config).
//
// Works with any LoRA produced by sd-scripts (the `ss_*` keys live in the
// safetensors header's `__metadata__` map), not just ones trained here.
//
// safetensors layout: [8-byte LE u64 header length][UTF-8 JSON header].
// All `__metadata__` values are STRINGS per spec — scalars are str(x),
// nested structures (network_args, datasets) are json.dumps(x). So every
// accessor below tolerates both a raw string and an already-parsed value.

const MAX_HEADER_BYTES = 64 * 1024 * 1024; // sanity cap; LoRA headers are small

export async function readSafetensorsMetadata(file) {
    const lenBuf = await file.slice(0, 8).arrayBuffer();
    const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
    if (!Number.isFinite(headerLen) || headerLen <= 0 || headerLen > MAX_HEADER_BYTES)
        throw new Error('не похоже на safetensors (некорректная длина заголовка)');
    const headerBuf = await file.slice(8, 8 + headerLen).arrayBuffer();
    const header = JSON.parse(new TextDecoder().decode(headerBuf));
    const meta = header.__metadata__ || {};
    if (!meta || typeof meta !== 'object')
        throw new Error('в файле нет __metadata__');
    return meta;
}

// ---- coercion helpers (every metadata value may be a string) --------------
const isNone = v => v === undefined || v === null || v === 'None' || v === '';
const asNum = v => { if (isNone(v)) return null; const n = Number(v); return Number.isFinite(n) ? n : null; };
const asInt = v => { const n = asNum(v); return n === null ? null : Math.round(n); };
const asBool = v => { if (isNone(v)) return null; if (typeof v === 'boolean') return v; return v === 'True' || v === 'true' || v === '1'; };
const asStr = v => (isNone(v) ? null : String(v));
const asJson = v => {
    if (isNone(v)) return null;
    if (typeof v === 'object') return v;
    try { return JSON.parse(v); } catch { return null; }
};

// Assign only when value is non-null, so we never overwrite a good current
// value with a metadata gap.
const put = (obj, key, val) => { if (val !== null && val !== undefined) obj[key] = val; };

// "bitsandbytes.optim.adamw.AdamW8bit(weight_decay=0.01,...)" → {type, args}
const OPT_ALIASES = {
    adamw: 'AdamW', adamw8bit: 'AdamW8bit',
    lion: 'Lion', lion8bit: 'Lion8bit',
    pagedadamw8bit: 'PagedAdamW8bit', pagedlion8bit: 'PagedLion8bit',
    prodigy: 'Prodigy',
    dadaptation: 'DAdaptation', dadaptadam: 'DAdaptAdam', dadaptlion: 'DAdaptLion',
    sgdnesterov: 'SGDNesterov', sgdnesterov8bit: 'SGDNesterov8bit',
    adafactor: 'AdaFactor',
    came: 'pytorch_optimizer.CAME',
};

function parseOptimizer(raw) {
    const s = asStr(raw);
    if (!s) return { type: null, args: null };
    const parenIdx = s.indexOf('(');
    const head = (parenIdx === -1 ? s : s.slice(0, parenIdx)).trim();
    const cls = head.split('.').pop() || head;
    const type = OPT_ALIASES[cls.toLowerCase()] || null;
    let args = null;
    if (parenIdx !== -1) {
        const inner = s.slice(parenIdx + 1, s.lastIndexOf(')'));
        args = inner.split(',').map(x => x.trim()).filter(Boolean);
        if (args.length === 0) args = null;
    }
    return { type, args };
}

// lycoris.kohya algo → our NetworkKind
const ALGO_KIND = { lora: 'locon', locon: 'locon', loha: 'loha', lokr: 'lokr', ia3: 'ia3', dylora: 'dylora' };

function detectNetwork(meta, net) {
    const mod = asStr(meta.ss_network_module) || '';
    const args = asJson(meta.ss_network_args) || {};
    if (mod.includes('lycoris')) {
        const algo = asStr(args.algo);
        put(net, 'kind', (algo && ALGO_KIND[algo.toLowerCase()]) || 'locon');
    } else if (mod.includes('dylora')) {
        put(net, 'kind', 'dylora');
    } else if (mod.includes('lora')) {
        put(net, 'kind', 'lora'); // covers networks.lora and networks.lora_anima
    }
    // LyCORIS network_args (conv layers, lokr factor, presets, dropouts)
    put(net, 'conv_dim', asInt(args.conv_dim));
    put(net, 'conv_alpha', asNum(args.conv_alpha));
    put(net, 'factor', asInt(args.factor));
    put(net, 'preset', asStr(args.preset));
    put(net, 'use_tucker', asBool(args.use_tucker));
    put(net, 'use_scalar', asBool(args.use_scalar));
    put(net, 'decompose_both', asBool(args.decompose_both));
    put(net, 'rank_dropout', asNum(args.rank_dropout));
    put(net, 'module_dropout', asNum(args.module_dropout));
    put(net, 'rank_dropout_scale', asBool(args.rank_dropout_scale));
    put(net, 'dropout', asNum(args.dropout));
}

// Build a partial config patch from parsed metadata. Returns {patch, warnings}.
export function metadataToPatch(meta) {
    const patch = { model: {}, project: {}, network: {}, optimizer: {}, training: {}, dataset: {} };
    const warnings = [];

    // ---- architecture ----
    const baseVer = asStr(meta.ss_base_model_version) || '';
    const netMod = asStr(meta.ss_network_module) || '';
    const isAnima = baseVer.includes('anima') || netMod.includes('anima');
    patch.model.arch = isAnima ? 'anima' : 'sdxl';
    if (isAnima) patch.model.preset = 'anima';

    // ---- model ----
    put(patch.model, 'clip_skip', asInt(meta.ss_clip_skip));
    const predType = asStr(meta['modelspec.prediction_type']);
    if (predType) patch.model.v_parameterization = predType === 'v';

    // ---- project ----
    put(patch.project, 'output_name', asStr(meta.ss_output_name));

    // ---- network ----
    put(patch.network, 'network_dim', asInt(meta.ss_network_dim));
    put(patch.network, 'network_alpha', asNum(meta.ss_network_alpha));
    put(patch.network, 'network_dropout', asNum(meta.ss_network_dropout));
    put(patch.network, 'scale_weight_norms', asNum(meta.ss_scale_weight_norms));
    detectNetwork(meta, patch.network);

    // ---- optimizer / schedule ----
    const { type: optType, args: optArgs } = parseOptimizer(meta.ss_optimizer);
    put(patch.optimizer, 'optimizer_type', optType);
    if (optArgs) patch.optimizer.optimizer_args = optArgs;
    if (asStr(meta.ss_optimizer) && !optType)
        warnings.push(`оптимизатор "${asStr(meta.ss_optimizer)}" не распознан — выставьте вручную`);
    put(patch.optimizer, 'learning_rate', asNum(meta.ss_learning_rate));
    put(patch.optimizer, 'unet_lr', asNum(meta.ss_unet_lr));
    put(patch.optimizer, 'text_encoder_lr', asNum(meta.ss_text_encoder_lr));
    put(patch.optimizer, 'lr_scheduler', asStr(meta.ss_lr_scheduler));
    put(patch.optimizer, 'lr_warmup_steps', asInt(meta.ss_lr_warmup_steps));
    put(patch.optimizer, 'max_grad_norm', asNum(meta.ss_max_grad_norm));

    // ---- training ----
    const t = patch.training;
    put(t, 'max_train_epochs', asInt(meta.ss_num_epochs));
    put(t, 'seed', asInt(meta.ss_seed));
    put(t, 'mixed_precision', asStr(meta.ss_mixed_precision));
    put(t, 'gradient_checkpointing', asBool(meta.ss_gradient_checkpointing));
    put(t, 'gradient_accumulation_steps', asInt(meta.ss_gradient_accumulation_steps));
    put(t, 'full_fp16', asBool(meta.ss_full_fp16));
    put(t, 'full_bf16', asBool(meta.ss_full_bf16));
    put(t, 'lowram', asBool(meta.ss_lowram));
    put(t, 'loss_type', asStr(meta.ss_loss_type));
    put(t, 'huber_schedule', asStr(meta.ss_huber_schedule));
    put(t, 'huber_c', asNum(meta.ss_huber_c));
    put(t, 'prior_loss_weight', asNum(meta.ss_prior_loss_weight));
    put(t, 'min_snr_gamma', asNum(meta.ss_min_snr_gamma));
    put(t, 'noise_offset', asNum(meta.ss_noise_offset));
    put(t, 'adaptive_noise_scale', asNum(meta.ss_adaptive_noise_scale));
    put(t, 'noise_offset_random_strength', asBool(meta.ss_noise_offset_random_strength));
    put(t, 'multires_noise_iterations', asInt(meta.ss_multires_noise_iterations));
    put(t, 'multires_noise_discount', asNum(meta.ss_multires_noise_discount));
    put(t, 'ip_noise_gamma', asNum(meta.ss_ip_noise_gamma));
    put(t, 'ip_noise_gamma_random_strength', asBool(meta.ss_ip_noise_gamma_random_strength));
    put(t, 'debiased_estimation_loss', asBool(meta.ss_debiased_estimation));
    put(t, 'zero_terminal_snr', asBool(meta.ss_zero_terminal_snr));
    // Anima FlowMatch extras
    put(t, 'weighting_scheme', asStr(meta.ss_weighting_scheme));
    put(t, 'discrete_flow_shift', asNum(meta.ss_discrete_flow_shift));
    put(t, 'timestep_sampling', asStr(meta.ss_timestep_sampling) || asStr(meta.ss_timestep_sample_method));
    put(t, 'sigmoid_scale', asNum(meta.ss_sigmoid_scale));
    put(t, 'logit_mean', asNum(meta.ss_logit_mean));
    put(t, 'logit_std', asNum(meta.ss_logit_std));
    put(t, 'mode_scale', asNum(meta.ss_mode_scale));

    // ---- dataset (global) ----
    const d = patch.dataset;
    put(d, 'cache_latents', asBool(meta.ss_cache_latents));
    put(d, 'caption_dropout_rate', asNum(meta.ss_caption_dropout_rate));
    put(d, 'caption_tag_dropout_rate', asNum(meta.ss_caption_tag_dropout_rate));
    put(d, 'caption_dropout_every_n_epochs', asInt(meta.ss_caption_dropout_every_n_epochs));
    put(d, 'max_token_length', asInt(meta.ss_max_token_length));

    // ---- dataset (resolution / buckets / subsets) from ss_datasets[0] ----
    const datasets = asJson(meta.ss_datasets);
    const ds0 = Array.isArray(datasets) && datasets[0] ? datasets[0] : null;
    if (ds0) {
        if (Array.isArray(ds0.resolution)) put(d, 'resolution', asInt(ds0.resolution[0]));
        put(d, 'enable_bucket', asBool(ds0.enable_bucket));
        put(d, 'min_bucket_reso', asInt(ds0.min_bucket_reso));
        put(d, 'max_bucket_reso', asInt(ds0.max_bucket_reso));
        if (Array.isArray(ds0.subsets) && ds0.subsets.length) {
            d.subsets = ds0.subsets.map(s => {
                const sub = { image_dir: asStr(s.image_dir) || '' };
                put(sub, 'num_repeats', asInt(s.num_repeats));
                put(sub, 'shuffle_caption', asBool(s.shuffle_caption));
                put(sub, 'keep_tokens', asInt(s.keep_tokens));
                put(sub, 'class_tokens', asStr(s.class_tokens));
                return sub;
            });
        }
    }

    // Drop empty sections so deepMerge stays minimal.
    for (const k of Object.keys(patch))
        if (patch[k] && typeof patch[k] === 'object' && Object.keys(patch[k]).length === 0)
            delete patch[k];

    return { patch, warnings };
}

export async function loraFileToPatch(file) {
    const meta = await readSafetensorsMetadata(file);
    const ssKeys = Object.keys(meta).filter(k => k.startsWith('ss_'));
    if (ssKeys.length === 0)
        throw new Error('нет тренировочной метадаты (ss_*) — возможно, это не LoRA от sd-scripts');
    return { meta, ...metadataToPatch(meta) };
}
