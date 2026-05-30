// LoRA Trainer · Preact + htm, no build step.
import { h, render } from 'https://esm.sh/preact@10.22.0';
import { useState, useEffect, useMemo, useRef, useCallback } from 'https://esm.sh/preact@10.22.0/hooks';
import htm from 'https://esm.sh/htm@3.1.1';

import { api, openSocket } from './api.js';
import { TIPS, tipFor } from './tooltips.js';
import { validate, findWarn } from './validations.js';
import { BUILTIN_PRESETS, deepMerge } from './presets.js';
import { loraFileToPatch } from './safetensors_meta.js';

const html = htm.bind(h);

// =============================================================================
// helpers
// =============================================================================
function setIn(obj, path, value) {
    const parts = Array.isArray(path) ? path : path.split('.');
    if (parts.length === 0) return value;
    const [head, ...rest] = parts;
    const isIndex = /^\d+$/.test(head);
    const key = isIndex ? Number(head) : head;
    if (rest.length === 0) {
        if (Array.isArray(obj)) {
            const next = obj.slice();
            next[key] = value;
            return next;
        }
        return { ...(obj || {}), [key]: value };
    }
    const child = obj ? obj[key] : undefined;
    const updated = setIn(child, rest, value);
    if (Array.isArray(obj)) {
        const next = obj.slice();
        next[key] = updated;
        return next;
    }
    return { ...(obj || {}), [key]: updated };
}

function getIn(obj, path) {
    const parts = Array.isArray(path) ? path : path.split('.');
    let cur = obj;
    for (const p of parts) {
        if (cur == null) return undefined;
        cur = cur[/^\d+$/.test(p) ? Number(p) : p];
    }
    return cur;
}

function fmtBytes(n) {
    if (!n && n !== 0) return '—';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

function fmtTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts * 1000);
    return d.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtDuration(s) {
    if (s == null || s < 0) return '—';
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
    if (h > 0) return `${h}ч ${m}м`;
    if (m > 0) return `${m}м ${sec}с`;
    return `${sec}с`;
}

// Strip arch-irrelevant fields from a config for clean export / display.
// SDXL configs shouldn't carry FlowMatch fields or anima_* paths; Anima
// configs shouldn't carry SDXL noise/loss fields, vae, or clip_skip.
function cleanForExport(cfg) {
    const c = JSON.parse(JSON.stringify(cfg));
    const isAnima = c.model?.arch === 'anima';
    if (isAnima) {
        // SDXL-only model fields
        delete c.model.vae;
        delete c.model.v_parameterization;
        delete c.model.clip_skip;
        // SDXL-only training noise/loss fields. The Anima+LoKr engine reuses a
        // handful of these (min_snr_gamma, noise_offset, multires_noise_*) via
        // its dedicated panel, so keep them for that combo — otherwise saved
        // presets / exports would silently drop the user's loss/noise settings.
        const isLokr = c.network?.kind === 'lokr';
        const tdel = isLokr
            ? [
                'adaptive_noise_scale', 'ip_noise_gamma', 'debiased_estimation_loss',
                'zero_terminal_snr', 'noise_offset_random_strength', 'ip_noise_gamma_random_strength',
            ]
            : [
                'min_snr_gamma', 'noise_offset', 'adaptive_noise_scale',
                'multires_noise_iterations', 'multires_noise_discount',
                'ip_noise_gamma', 'debiased_estimation_loss', 'zero_terminal_snr',
                'noise_offset_random_strength', 'ip_noise_gamma_random_strength',
            ];
        tdel.forEach(k => delete c.training[k]);
    } else {
        // Anima-only model fields
        delete c.model.anima_qwen3;
        delete c.model.anima_t5_tokenizer_path;
        delete c.model.anima_vae;
        // Anima FlowMatch training fields
        const fdel = [
            'weighting_scheme', 'logit_mean', 'logit_std', 'mode_scale',
            'timestep_sampling', 'sigmoid_scale', 'discrete_flow_shift',
        ];
        fdel.forEach(k => delete c.training[k]);
    }
    return c;
}

function computeTotalStepsLocal(cfg) {
    const subs = cfg.dataset?.subsets || [];
    if (subs.length === 0) return 0;
    const weighted = subs.reduce((s, x) => s + Math.max(1, x.num_images || 0) * Math.max(1, x.num_repeats || 0), 0);
    const bs = Math.max(1, cfg.training.train_batch_size) * Math.max(1, cfg.training.gradient_accumulation_steps);
    const perEpoch = Math.max(1, Math.floor(weighted / bs));
    if (cfg.training.max_train_steps && cfg.training.max_train_steps > 0) return cfg.training.max_train_steps;
    return perEpoch * Math.max(1, cfg.training.max_train_epochs);
}

// Step multiplier at fraction x for piecewise_constant (kohya semantics:
// a level holds from its `at` until the next point's `at`).
function piecewiseMult(points, x) {
    const pts = [...(points || [])].sort((a, b) => a.at - b.at);
    if (pts.length === 0) return 1;
    let m = pts[0].mult;
    for (const p of pts) { if (x >= p.at) m = p.mult; else break; }
    return Math.max(0, Math.min(1, m));
}

// Normalized LR shape (y in 0..1, peak = configured learning_rate) over the run.
// Mirrors kohya/transformers scheduler math. Returns [] when not plottable.
function lrCurvePoints(cfg, totalSteps, n = 240) {
    if (!totalSteps || totalSteps <= 0) return [];
    const o = cfg.optimizer;
    const sched = o.lr_scheduler;
    if (sched === 'adafactor') return []; // optimizer-internal, no static curve
    const ws = o.lr_warmup_steps || 0;
    const warm = ws > 0 && ws < 1 ? ws : ws / totalSteps; // fraction of total
    const cycles = Math.max(1, o.lr_scheduler_num_cycles || 1);
    const power = o.lr_scheduler_power > 0 ? o.lr_scheduler_power : 1.0;
    const ds = o.lr_decay_steps;
    const decayFrac = ds == null ? 0 : (ds > 0 && ds < 1 ? ds : ds / totalSteps);

    const val = (x) => {
        if (sched === 'piecewise_constant') return piecewiseMult(o.lr_piecewise || [], x);
        if (warm > 0 && x < warm) return x / warm;          // linear warmup
        const p = warm >= 1 ? 1 : (x - warm) / (1 - warm);  // post-warmup progress 0..1
        switch (sched) {
            case 'constant':
            case 'constant_with_warmup': return 1;
            case 'linear': return Math.max(0, 1 - p);
            case 'cosine': return 0.5 * (1 + Math.cos(Math.PI * p));
            case 'cosine_with_restarts': {
                let ph = (cycles * p) % 1;
                if (p > 0 && ph === 0) ph = 1; // cycle boundary = trough, not a jump to peak
                return 0.5 * (1 + Math.cos(Math.PI * ph));
            }
            case 'polynomial': return Math.pow(Math.max(0, 1 - p), power);
            case 'warmup_stable_decay': {
                const start = 1 - decayFrac;
                if (decayFrac <= 0 || x < start) return 1;       // stable plateau
                const d = Math.min(1, (x - start) / decayFrac);
                return 0.5 * (1 + Math.cos(Math.PI * d));         // short cosine decay
            }
            default: return 0.5 * (1 + Math.cos(Math.PI * p));
        }
    };

    const pts = [];
    for (let i = 0; i <= n; i++) {
        const x = i / n;
        pts.push({ x, y: Math.max(0, Math.min(1, val(x))) });
    }
    return pts;
}

// =============================================================================
// generic widgets
// =============================================================================
const Field = ({ label, tipKey, children, warn, err, columns = 'split' }) => {
    const tip = tipKey ? TIPS[tipKey] : null;
    const cls = `row${columns === 'stack' ? ' row-stack' : ''}`;
    return html`
        <div class=${cls}>
            <label class="field-label">
                ${label}
                ${tip && html`<span class="tip tooltip-trigger" data-tip=${tip}>ⓘ</span>`}
            </label>
            <div>
                ${children}
                ${err && html`<div class="field-err">${err}</div>`}
                ${warn && !err && html`<div class="field-warn">${warn}</div>`}
            </div>
        </div>
    `;
};

const Text = ({ value, onInput, placeholder, warn, err }) => html`
    <input type="text" class=${err ? 'err' : warn ? 'warn' : ''}
        value=${value ?? ''} placeholder=${placeholder ?? ''}
        onInput=${e => onInput(e.target.value)} />
`;

const Num = ({ value, onInput, step = 1, min, max, warn, err }) => {
    const [buf, setBuf] = useState(value == null ? '' : String(value));
    const focused = useRef(false);
    useEffect(() => {
        if (!focused.current) setBuf(value == null ? '' : String(value));
    }, [value]);
    const commit = (v, finalize) => {
        setBuf(v);
        if (v === '' || v === '-' || v === '.' || v === '-.') {
            if (finalize) {
                onInput(null);
                if (value != null) setBuf(String(value));
            }
            return;
        }
        const parsed = step < 1 ? parseFloat(v) : parseInt(v, 10);
        if (!isNaN(parsed)) onInput(parsed);
    };
    return html`
        <input type="number" step=${step} min=${min} max=${max}
            class=${'num ' + (err ? 'err' : warn ? 'warn' : '')}
            value=${buf}
            onFocus=${() => { focused.current = true; }}
            onBlur=${e => { focused.current = false; commit(e.target.value, true); }}
            onInput=${e => commit(e.target.value, false)} />
    `;
};

const Select = ({ value, onInput, options, warn, err }) => html`
    <select class=${err ? 'err' : warn ? 'warn' : ''} value=${value} onChange=${e => onInput(e.target.value)}>
        ${options.map(opt => {
            const [val, label] = Array.isArray(opt) ? opt : [opt, opt];
            return html`<option value=${val}>${label}</option>`;
        })}
    </select>
`;

const Switch = ({ value, onInput }) => html`
    <div class=${'switch' + (value ? ' on' : '')} onClick=${() => onInput(!value)}></div>
`;

const Check = ({ value, onInput, label }) => html`
    <label class="check">
        <input type="checkbox" checked=${!!value} onChange=${e => onInput(e.target.checked)} />
        <span>${label}</span>
    </label>
`;

const Btn = ({ children, onClick, variant = '', size = '', disabled }) => html`
    <button class=${`btn ${variant ? 'btn-' + variant : ''} ${size ? 'btn-' + size : ''}`}
        onClick=${onClick} disabled=${disabled}>${children}</button>
`;

// =============================================================================
// Toasts
// =============================================================================
const ToastCtx = {
    listeners: [],
    push(t) { this.listeners.forEach(fn => fn(t)); },
};
function useToasts() {
    const [items, setItems] = useState([]);
    useEffect(() => {
        const fn = (t) => {
            const id = Math.random().toString(36).slice(2);
            setItems(prev => [...prev, { id, ...t }]);
            setTimeout(() => setItems(prev => prev.filter(x => x.id !== id)), t.timeout || 4000);
        };
        ToastCtx.listeners.push(fn);
        return () => { ToastCtx.listeners = ToastCtx.listeners.filter(x => x !== fn); };
    }, []);
    return items;
}
const toast = (kind, message) => ToastCtx.push({ kind, message });

const ToastStack = () => {
    const items = useToasts();
    return html`
        <div class="toast-stack">
            ${items.map(t => html`<div class=${'toast ' + t.kind} key=${t.id}>${t.message}</div>`)}
        </div>
    `;
};

// =============================================================================
// Section: PROJECT (paths + output name)
// =============================================================================
const SectionProject = ({ cfg, set, val }) => {
    return html`
        <h2>Проект</h2>
        <div class="subtitle">Базовые пути и имя выходного файла.</div>

        <div class="card">
            <div class="card-title">Пути <span class="hint">— заданы в Colab-ячейке, можно поправить ниже</span></div>
            <${Field} label="dataset_root" tipKey="paths.dataset_root" err=${val.errMap['paths.dataset_root']}>
                <${Text} value=${cfg.paths.dataset_root} onInput=${v => set('paths.dataset_root', v)} placeholder="/content/drive/MyDrive/datasets/my-lora" />
            </${Field}>
            <${Field} label="base_model_root" tipKey="paths.base_model_root">
                <${Text} value=${cfg.paths.base_model_root} onInput=${v => set('paths.base_model_root', v)} placeholder="/content/drive/MyDrive/models" />
            </${Field}>
            <${Field} label="output_root" tipKey="paths.output_root" err=${val.errMap['paths.output_root']}>
                <${Text} value=${cfg.paths.output_root} onInput=${v => set('paths.output_root', v)} placeholder="/content/drive/MyDrive/output/my-lora" />
            </${Field}>
            <${Field} label="samples_root" tipKey="paths.samples_root">
                <${Text} value=${cfg.paths.samples_root} onInput=${v => set('paths.samples_root', v)} placeholder="(оставьте пустым — kohya пишет в output_root/sample)" />
            </${Field}>
            <${Field} label="sd_scripts_dir" tipKey="paths.sd_scripts_dir" err=${val.errMap['paths.sd_scripts_dir']}>
                <${Text} value=${cfg.paths.sd_scripts_dir} onInput=${v => set('paths.sd_scripts_dir', v)} placeholder="/content/sd-scripts" />
            </${Field}>
        </div>

        <div class="card">
            <div class="card-title">Имя проекта</div>
            <${Field} label="output_name" tipKey="project.output_name">
                <${Text} value=${cfg.project.output_name} onInput=${v => set('project.output_name', v)} placeholder="my-lora-v1" />
            </${Field}>
            <${Field} label="save_model_as" tipKey="project.save_model_as">
                <${Select} value=${cfg.project.save_model_as} onInput=${v => set('project.save_model_as', v)}
                    options=${['safetensors', 'ckpt', 'pt']} />
            </${Field}>
            <${Field} label="log_with" tipKey="project.log_with">
                <${Select} value=${cfg.project.log_with} onInput=${v => set('project.log_with', v)}
                    options=${[['none', 'нет'], 'tensorboard', 'wandb']} />
            </${Field}>
        </div>
    `;
};

// =============================================================================
// Section: MODEL
// =============================================================================
const SectionModel = ({ cfg, set, val, models }) => {
    const isAnima = cfg.model.arch === 'anima';
    const archOpts = [['sdxl', 'SDXL (NoobAI / Illustrious)'], ['anima', 'Anima (DiT · Qwen3 + Qwen-Image VAE)']];
    const presetOpts = isAnima
        ? [['anima', 'Anima'], ['custom', 'Custom']]
        : [['noobai', 'NoobAI'], ['illustrious', 'Illustrious'], ['custom', 'Custom']];

    return html`
        <h2>Модель</h2>
        <div class="subtitle">Архитектура и веса базовой модели.</div>

        <div class="card">
            <div class="card-title">
                Архитектура
                <span class="tag accent">${isAnima ? 'Anima · DiT FlowMatch' : 'SDXL'}</span>
            </div>
            <${Field} label="arch" tipKey="model.arch">
                <${Select} value=${cfg.model.arch} onInput=${v => set('model.arch', v)} options=${archOpts} />
            </${Field}>
            <${Field} label="preset" tipKey="model.preset">
                <${Select} value=${cfg.model.preset} onInput=${v => set('model.preset', v)} options=${presetOpts} />
            </${Field}>
        </div>

        <div class="card">
            <div class="card-title">Пути к весам</div>
            <${Field} label="pretrained_model_name_or_path" tipKey="model.pretrained_model_name_or_path"
                err=${val.errMap['model.pretrained_model_name_or_path']}>
                <${ModelPicker} value=${cfg.model.pretrained_model_name_or_path}
                    onInput=${v => set('model.pretrained_model_name_or_path', v)} models=${models} />
            </${Field}>
            ${!isAnima && html`
                <${Field} label="vae" tipKey="model.vae">
                    <${ModelPicker} value=${cfg.model.vae || ''} onInput=${v => set('model.vae', v || null)} models=${models} />
                </${Field}>
                <${Field} label="v_parameterization" tipKey="model.v_parameterization">
                    <${Check} value=${cfg.model.v_parameterization} onInput=${v => set('model.v_parameterization', v)} label="v-prediction" />
                </${Field}>
                <${Field} label="clip_skip" tipKey="model.clip_skip">
                    <${Num} value=${cfg.model.clip_skip} onInput=${v => set('model.clip_skip', v)} min=${1} max=${12} />
                </${Field}>
            `}
            ${isAnima && html`
                <${Field} label="qwen3 (text encoder)" tipKey="model.anima_qwen3"
                    err=${val.errMap['model.anima_qwen3']}>
                    <${Text} value=${cfg.model.anima_qwen3 || ''} onInput=${v => set('model.anima_qwen3', v || null)}
                        placeholder="/content/drive/MyDrive/models/anima/qwen3-0.6B" />
                </${Field}>
                <${Field} label="t5_tokenizer_path" tipKey="model.anima_t5_tokenizer_path"
                    err=${val.errMap['model.anima_t5_tokenizer_path']}>
                    <${Text} value=${cfg.model.anima_t5_tokenizer_path || ''} onInput=${v => set('model.anima_t5_tokenizer_path', v || null)} />
                </${Field}>
                <${Field} label="vae (Qwen-Image)" tipKey="model.anima_vae"
                    err=${val.errMap['model.anima_vae']}>
                    <${ModelPicker} value=${cfg.model.anima_vae || ''} onInput=${v => set('model.anima_vae', v || null)} models=${models} />
                </${Field}>
            `}
        </div>
    `;
};

const ModelPicker = ({ value, onInput, models }) => {
    const [open, setOpen] = useState(false);
    return html`
        <div style="position: relative;">
            <${Text} value=${value} onInput=${onInput} placeholder="(введите путь или выберите ↘)" />
            ${models && models.length > 0 && html`
                <div style="display:flex; gap:6px; margin-top:6px; flex-wrap:wrap;">
                    ${models.slice(0, 16).map(m => html`
                        <button class="btn btn-sm btn-ghost mono" onClick=${() => onInput(m.path)}
                            title=${m.path}>${m.name}</button>
                    `)}
                </div>
            `}
        </div>
    `;
};

// =============================================================================
// Section: DATASET
// =============================================================================
const SectionDataset = ({ cfg, set, val, rescan, scanResult }) => {
    const totalImgs = (cfg.dataset.subsets || []).reduce((s, x) => s + (x.num_images || 0), 0);
    const totalWeighted = (cfg.dataset.subsets || []).reduce((s, x) => s + (x.num_images || 0) * (x.num_repeats || 0), 0);

    return html`
        <h2>Датасет</h2>
        <div class="subtitle">
            ${totalImgs} картинок · с повторами: <b>${totalWeighted}</b>
        </div>

        <div class="card">
            <div class="card-title">
                Подпапки
                <span class="hint">— любое название, repeats задаётся в поле справа</span>
                <div style="margin-left:auto; display:flex; gap:6px;">
                    <${Btn} size="sm" variant="ghost" onClick=${rescan}>↻ Пересканировать папку</${Btn}>
                    <${Btn} size="sm" variant="ghost" onClick=${() => set('dataset.subsets', [
                        ...(cfg.dataset.subsets || []),
                        { image_dir: '', num_repeats: 10, caption_extension: '.txt', shuffle_caption: true, keep_tokens: 1, num_images: 0 }
                    ])}>+ Добавить вручную</${Btn}>
                </div>
            </div>
            ${val.errMap['dataset.subsets'] && html`<div class="field-err" style="margin-bottom:8px;">${val.errMap['dataset.subsets']}</div>`}

            <div class="dataset-row" style="color: var(--text-mute); font-size: 11px; padding: 0 0 4px;">
                <div>Картинок</div><div>Папка</div><div>Repeats</div><div>Подписи (.txt)</div><div></div>
            </div>
            ${(cfg.dataset.subsets || []).map((s, i) => html`
                <div key=${i} style="border-bottom: 1px solid var(--line); padding-bottom:6px; margin-bottom:6px;">
                    <div class="dataset-row">
                        <div class="mono tabnum" style="color: var(--text-dim);">${s.num_images || '—'}</div>
                        <input class="mono" type="text" value=${s.image_dir} placeholder="/content/drive/MyDrive/dataset/concept"
                            onInput=${e => set(`dataset.subsets.${i}.image_dir`, e.target.value)} />
                        <input type="number" min="1" value=${s.num_repeats}
                            onInput=${e => set(`dataset.subsets.${i}.num_repeats`, parseInt(e.target.value, 10) || 1)} />
                        <input type="text" value=${s.caption_extension}
                            onInput=${e => set(`dataset.subsets.${i}.caption_extension`, e.target.value)} />
                        <button class="btn btn-sm btn-ghost" title="Удалить"
                            onClick=${() => set('dataset.subsets', cfg.dataset.subsets.filter((_, j) => j !== i))}>✕</button>
                    </div>
                    <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:center; padding-left:4px; margin-top:4px; font-size:11.5px;">
                        <${Check} value=${s.shuffle_caption !== false} onInput=${v => set(`dataset.subsets.${i}.shuffle_caption`, v)} label="shuffle_caption" />
                        <label style="display:flex; align-items:center; gap:4px;">
                            <span class="dim">keep_tokens</span>
                            <input type="number" min="0" style="width:60px;" value=${s.keep_tokens ?? 1}
                                onInput=${e => set(`dataset.subsets.${i}.keep_tokens`, parseInt(e.target.value, 10) || 0)} />
                        </label>
                        <label style="display:flex; align-items:center; gap:4px; flex:1; min-width:220px;">
                            <span class="dim">class_tokens</span>
                            <input type="text" style="flex:1;" value=${s.class_tokens || ''} placeholder="(опционально, для картинок без .txt)"
                                onInput=${e => set(`dataset.subsets.${i}.class_tokens`, e.target.value || null)} />
                        </label>
                        <label style="display:flex; align-items:center; gap:4px;">
                            <span class="dim">keep_tokens_separator</span>
                            <input type="text" style="width:60px;" value=${s.keep_tokens_separator || ''} placeholder="|||"
                                onInput=${e => set(`dataset.subsets.${i}.keep_tokens_separator`, e.target.value || null)} />
                        </label>
                    </div>
                </div>
            `)}
        </div>

        <div class="card">
            <div class="card-title">Разрешение и bucket</div>
            <div class="grid-3">
                <${Field} label="resolution" tipKey="dataset.resolution" warn=${val.warnMap['dataset.resolution']}>
                    <${Num} value=${cfg.dataset.resolution} step=${64} onInput=${v => set('dataset.resolution', v)} warn=${val.warnMap['dataset.resolution']} />
                </${Field}>
                <${Field} label="min_bucket_reso" tipKey="dataset.min_bucket_reso">
                    <${Num} value=${cfg.dataset.min_bucket_reso} step=${64} onInput=${v => set('dataset.min_bucket_reso', v)} />
                </${Field}>
                <${Field} label="max_bucket_reso" tipKey="dataset.max_bucket_reso">
                    <${Num} value=${cfg.dataset.max_bucket_reso} step=${64} onInput=${v => set('dataset.max_bucket_reso', v)} />
                </${Field}>
            </div>
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                <${Check} value=${cfg.dataset.enable_bucket} onInput=${v => set('dataset.enable_bucket', v)} label="enable_bucket" />
                <${Check} value=${cfg.dataset.bucket_no_upscale} onInput=${v => set('dataset.bucket_no_upscale', v)} label="bucket_no_upscale" />
                <${Check} value=${cfg.dataset.cache_latents} onInput=${v => set('dataset.cache_latents', v)} label="cache_latents" />
                <${Check} value=${cfg.dataset.cache_latents_to_disk} onInput=${v => set('dataset.cache_latents_to_disk', v)} label="cache_latents_to_disk" />
                <${Check} value=${cfg.dataset.cache_text_encoder_outputs} onInput=${v => set('dataset.cache_text_encoder_outputs', v)} label="cache_text_encoder_outputs" />
                <${Check} value=${cfg.dataset.cache_text_encoder_outputs_to_disk} onInput=${v => set('dataset.cache_text_encoder_outputs_to_disk', v)} label="cache_te_to_disk" />
                <${Check} value=${cfg.dataset.color_aug} onInput=${v => set('dataset.color_aug', v)} label="color_aug" />
                <${Check} value=${cfg.dataset.flip_aug} onInput=${v => set('dataset.flip_aug', v)} label="flip_aug" />
                <${Check} value=${cfg.dataset.random_crop} onInput=${v => set('dataset.random_crop', v)} label="random_crop" />
            </div>
        </div>

        <div class="card">
            <div class="card-title">Подписи и DataLoader</div>
            <div class="grid-3">
                ${cfg.model.arch !== 'anima' && html`
                <${Field} label="max_token_length" tipKey="dataset.max_token_length">
                    <${Select} value=${cfg.dataset.max_token_length == null ? 'none' : String(cfg.dataset.max_token_length)}
                        onInput=${v => set('dataset.max_token_length', v === 'none' ? null : parseInt(v, 10))}
                        options=${[['none','без лимита'],['75','75'],['150','150'],['225','225']]} />
                </${Field}>
                `}
                <${Field} label="max_data_loader_n_workers" tipKey="dataset.max_data_loader_n_workers">
                    <${Num} value=${cfg.dataset.max_data_loader_n_workers ?? 8} min=${0} max=${32}
                        onInput=${v => set('dataset.max_data_loader_n_workers', v)} />
                </${Field}>
                <${Field} label="persistent_data_loader_workers" tipKey="dataset.persistent_data_loader_workers">
                    <${Switch} value=${cfg.dataset.persistent_data_loader_workers !== false}
                        onInput=${v => set('dataset.persistent_data_loader_workers', v)} />
                </${Field}>
                <${Field} label="caption_dropout_rate" tipKey="dataset.caption_dropout_rate">
                    <${Num} value=${cfg.dataset.caption_dropout_rate ?? 0} step=${0.01} min=${0} max=${1}
                        onInput=${v => set('dataset.caption_dropout_rate', v)} />
                </${Field}>
                <${Field} label="caption_tag_dropout_rate" tipKey="dataset.caption_tag_dropout_rate">
                    <${Num} value=${cfg.dataset.caption_tag_dropout_rate ?? 0} step=${0.01} min=${0} max=${1}
                        onInput=${v => set('dataset.caption_tag_dropout_rate', v)} />
                </${Field}>
                <${Field} label="caption_dropout_every_n_epochs" tipKey="dataset.caption_dropout_every_n_epochs">
                    <${Num} value=${cfg.dataset.caption_dropout_every_n_epochs ?? 0} min=${0}
                        onInput=${v => set('dataset.caption_dropout_every_n_epochs', v)} />
                </${Field}>
                <${Field} label="token_warmup_min" tipKey="dataset.token_warmup_min">
                    <${Num} value=${cfg.dataset.token_warmup_min ?? 1} min=${1}
                        onInput=${v => set('dataset.token_warmup_min', v)} />
                </${Field}>
                <${Field} label="token_warmup_step" tipKey="dataset.token_warmup_step">
                    <${Num} value=${cfg.dataset.token_warmup_step ?? 0} min=${0}
                        onInput=${v => set('dataset.token_warmup_step', v)} />
                </${Field}>
                <${Field} label="weighted_captions" tipKey="dataset.weighted_captions">
                    <${Switch} value=${!!cfg.dataset.weighted_captions}
                        onInput=${v => set('dataset.weighted_captions', v)} />
                </${Field}>
            </div>
            <div class="dim" style="font-size:11.5px; margin-top:6px;">
                💡 Скорость: <b>max_data_loader_n_workers</b> = 4–8 + <b>persistent_data_loader_workers</b> = ON
                часто ускоряют эпохи на 30–80% на Colab. Если эпоха длится 2× дольше ожидаемого — проверьте эти два поля.
            </div>
        </div>
    `;
};

// =============================================================================
// Section: NETWORK (LoRA / LyCORIS / LoKr)
// =============================================================================
const SectionNetwork = ({ cfg, set, val }) => {
    const n = cfg.network;
    const isAlphaWarn = !!val.warnMap['network.network_alpha'];
    const isLycoris = n.kind !== 'lora';
    // Anima + LoKr runs through AnimaLoraStudio's engine, NOT kohya. A pile
    // of kohya-only knobs (network_dropout, scale_weight_norms, use_tucker,
    // train_unet/te_only switches, base_weights, dim_from_weights) have no
    // counterpart there — gate them so the form only shows what actually
    // reaches the trainer.
    const isAnimaLokr = cfg.model?.arch === 'anima' && n.kind === 'lokr';
    return html`
        <h2>Сеть</h2>
        <div class="subtitle">LoRA или LyCORIS (LoCon · LoHA · LoKr · DyLoRA · IA³).</div>

        <div class="card">
            <div class="card-title">Тип и размер</div>
            <div class="grid-3">
                <${Field} label="network kind" tipKey="network.kind">
                    <${Select} value=${n.kind} onInput=${v => set('network.kind', v)}
                        options=${[
                            ['lora','LoRA'],['locon','LyCORIS LoCon'],['loha','LyCORIS LoHA'],
                            ['lokr','LyCORIS LoKr'],['dylora','LyCORIS DyLoRA'],['ia3','LyCORIS IA³']
                        ]} />
                </${Field}>
                <${Field} label="network_dim" tipKey="network.network_dim">
                    <${Num} value=${n.network_dim} onInput=${v => set('network.network_dim', v)} min=${1} />
                </${Field}>
                <${Field} label="network_alpha" tipKey="network.network_alpha"
                    warn=${val.warnMap['network.network_alpha']}>
                    <${Num} value=${n.network_alpha} step=${0.5} onInput=${v => set('network.network_alpha', v)}
                        warn=${isAlphaWarn} />
                </${Field}>
            </div>
            ${isLycoris && cfg.model.arch !== 'anima' && html`
                <div class="grid-3">
                    <${Field} label="preset" tipKey="network.preset">
                        <${Select} value=${n.preset} onInput=${v => set('network.preset', v)}
                            options=${['full','full-lin','attn-mlp','attn-only','unet-transformer-only','unet-convblock-only']} />
                    </${Field}>
                    <${Field} label="conv_dim" tipKey="network.conv_dim" warn=${val.warnMap['network.conv_dim']}>
                        <${Num} value=${n.conv_dim} onInput=${v => set('network.conv_dim', v)} warn=${!!val.warnMap['network.conv_dim']} />
                    </${Field}>
                    <${Field} label="conv_alpha" tipKey="network.conv_alpha">
                        <${Num} value=${n.conv_alpha} step=${0.5} onInput=${v => set('network.conv_alpha', v)} />
                    </${Field}>
                </div>
            `}
            ${n.kind === 'lokr' && html`
                <div class="grid-3">
                    <${Field} label="factor" tipKey="network.factor" warn=${val.warnMap['network.factor']}>
                        <${Num} value=${n.factor} onInput=${v => set('network.factor', v)} warn=${!!val.warnMap['network.factor']} />
                    </${Field}>
                    <${Field} label="decompose_both" tipKey="network.decompose_both">
                        <${Switch} value=${n.decompose_both} onInput=${v => set('network.decompose_both', v)} />
                    </${Field}>
                </div>
                <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                    <${Check} value=${n.weight_decompose} onInput=${v => set('network.weight_decompose', v)} label="DoRA (weight_decompose)" />
                    <${Check} value=${n.rs_lora} onInput=${v => set('network.rs_lora', v)} label="rs-LoRA" />
                </div>
                ${cfg.model.arch === 'anima' && html`
                    <div class="dim" style="margin-top:6px; padding:6px 10px; border-left:3px solid #5fb04a; background:rgba(95,176,74,0.08); font-size:12.5px;">
                        Anima + LoKr → тренировка идёт через AnimaLoraStudio (не kohya).
                        Активный пресет таргетинга: <b>anima_full</b>
                        — q/k/v/output_proj + mlp.layer1/2, без llm_adapter.
                        Поля <code>preset</code> / <code>conv_dim</code> / <code>conv_alpha</code> игнорируются.
                    </div>
                `}
            `}
        </div>

        <div class="card">
            <div class="card-title">Регуляризация и охват</div>
            <div class="grid-3">
                <${Field} label="rank_dropout" tipKey="network.rank_dropout">
                    <${Num} value=${n.rank_dropout} step=${0.05} min=${0} max=${1} onInput=${v => set('network.rank_dropout', v)} />
                </${Field}>
                <${Field} label="module_dropout" tipKey="network.module_dropout">
                    <${Num} value=${n.module_dropout} step=${0.05} min=${0} max=${1} onInput=${v => set('network.module_dropout', v)} />
                </${Field}>
                ${!isAnimaLokr && html`
                    <${Field} label="network_dropout" tipKey="network.network_dropout">
                        <${Num} value=${n.network_dropout} step=${0.05} min=${0} max=${1} onInput=${v => set('network.network_dropout', v)} />
                    </${Field}>
                    <${Field} label="scale_weight_norms" tipKey="network.scale_weight_norms">
                        <${Num} value=${n.scale_weight_norms} step=${0.1} onInput=${v => set('network.scale_weight_norms', v)} />
                    </${Field}>
                `}
            </div>
            ${!isAnimaLokr && html`
                <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                    <${Check} value=${n.use_tucker} onInput=${v => set('network.use_tucker', v)} label="use_tucker" />
                    <${Check} value=${n.use_scalar} onInput=${v => set('network.use_scalar', v)} label="use_scalar" />
                    <${Check} value=${n.rank_dropout_scale} onInput=${v => set('network.rank_dropout_scale', v)} label="rank_dropout_scale" />
                    <${Check} value=${n.network_train_unet_only} onInput=${v => set('network.network_train_unet_only', v)} label="train_unet_only" />
                    <${Check} value=${n.network_train_text_encoder_only} onInput=${v => set('network.network_train_text_encoder_only', v)} label="train_text_encoder_only" />
                </div>
                ${val.errMap['network.network_train_text_encoder_only'] && html`
                    <div class="field-err" style="margin-top:6px;">${val.errMap['network.network_train_text_encoder_only']}</div>
                `}
            `}
        </div>

        ${isAnimaLokr && html`<${AnimaLokrNetworkPanel} cfg=${cfg} set=${set} />`}

        <div class="card">
            <div class="card-title">Продолжить с весов (опционально)</div>
            <${Field} label="network_weights" tipKey="network.network_weights">
                <${Text} value=${n.network_weights || ''} onInput=${v => set('network.network_weights', v || null)}
                    placeholder="(пустое — обучаем с нуля; или путь к .safetensors LoRA)" />
            </${Field}>
            ${!isAnimaLokr && html`
                <${Field} label="dim_from_weights" tipKey="network.dim_from_weights">
                    <${Check} value=${!!n.dim_from_weights} onInput=${v => set('network.dim_from_weights', v)}
                        label="взять network_dim/alpha из весов" />
                </${Field}>
            `}
        </div>
    `;
};

// =============================================================================
// Section: ANIMA + LOKR EXTRAS (LyCORIS per-layer rank, regularization dataset)
// =============================================================================
// Visible only when (arch=anima, kind=lokr). Maps 1:1 to AnimaLoraStudio's
// TrainingConfig fields under our cfg.anima_lokr section.
const AnimaLokrNetworkPanel = ({ cfg, set }) => {
    const al = cfg.anima_lokr || {};
    const regDimsText = al.lora_reg_dims
        ? Object.entries(al.lora_reg_dims).map(([k, v]) => `${k} = ${v}`).join('\n')
        : '';
    const onRegDims = (text) => {
        const out = {};
        for (const line of (text || '').split(/\r?\n/)) {
            const m = line.match(/^\s*(.+?)\s*=\s*(\d+)\s*$/);
            if (m) out[m[1]] = parseInt(m[2], 10);
        }
        set('anima_lokr.lora_reg_dims', Object.keys(out).length ? out : null);
    };
    return html`
        <div class="card">
            <div class="card-title">Anima · LoKr · дополнительно</div>
            <${Field} label="lora_reg_dims" tipKey="anima_lokr.lora_reg_dims" columns="stack">
                <textarea class="ta" rows="3" style="font-family:monospace;"
                    placeholder="regex = rank, по одному правилу на строку. Пример:&#10;lora_unet_.*double.* = 16"
                    value=${regDimsText}
                    onInput=${e => onRegDims(e.target.value)} />
            </${Field}>
            <div class="grid-3">
                <${Field} label="kv_trim" tipKey="anima_lokr.kv_trim">
                    <${Switch} value=${!!al.kv_trim} onInput=${v => set('anima_lokr.kv_trim', v)} />
                </${Field}>
                <${Field} label="reg_data_dir" tipKey="anima_lokr.reg_data_dir">
                    <${Text} value=${al.reg_data_dir || ''}
                        onInput=${v => set('anima_lokr.reg_data_dir', v || null)}
                        placeholder="(опц.) папка с regularization-картинками" />
                </${Field}>
                <${Field} label="reg_weight" tipKey="anima_lokr.reg_weight">
                    <${Num} value=${al.reg_weight ?? 1.0} step=${0.05} min=${0} max=${1}
                        onInput=${v => set('anima_lokr.reg_weight', v)} />
                </${Field}>
            </div>
        </div>
    `;
};

// PPSF (ProdigyPlusScheduleFree) — exposed only when optimizer_type matches and
// (arch=anima, kind=lokr). All fields land in cfg.anima_lokr.ppsf_*.
const PPSFPanel = ({ al, set }) => html`
    <div class="grid-3" style="margin-top:6px;">
        <${Field} label="ppsf_d_coef" tipKey="anima_lokr.ppsf_d_coef">
            <${Num} value=${al.ppsf_d_coef ?? 1.0} step=${0.1} min=${0.1} max=${10}
                onInput=${v => set('anima_lokr.ppsf_d_coef', v)} />
        </${Field}>
        <${Field} label="ppsf_prodigy_steps" tipKey="anima_lokr.ppsf_prodigy_steps">
            <${Num} value=${al.ppsf_prodigy_steps ?? 0} min=${0}
                onInput=${v => set('anima_lokr.ppsf_prodigy_steps', v)} />
        </${Field}>
        <${Field} label="ppsf_beta1" tipKey="anima_lokr.ppsf_beta1">
            <${Num} value=${al.ppsf_beta1 ?? 0.9} step=${0.01} min=${0} max=${1}
                onInput=${v => set('anima_lokr.ppsf_beta1', v)} />
        </${Field}>
        <${Field} label="ppsf_beta2" tipKey="anima_lokr.ppsf_beta2">
            <${Num} value=${al.ppsf_beta2 ?? 0.99} step=${0.001} min=${0} max=${1}
                onInput=${v => set('anima_lokr.ppsf_beta2', v)} />
        </${Field}>
    </div>
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
        <${Check} value=${!!al.ppsf_split_groups} onInput=${v => set('anima_lokr.ppsf_split_groups', v)} label="split_groups" />
        <${Check} value=${!!al.ppsf_split_groups_mean} onInput=${v => set('anima_lokr.ppsf_split_groups_mean', v)} label="split_groups_mean" />
        <${Check} value=${!!al.ppsf_use_speed} onInput=${v => set('anima_lokr.ppsf_use_speed', v)} label="use_speed" />
        <${Check} value=${!!al.ppsf_fused_back_pass} onInput=${v => set('anima_lokr.ppsf_fused_back_pass', v)} label="fused_back_pass" />
        <${Check} value=${!!al.ppsf_use_stableadamw} onInput=${v => set('anima_lokr.ppsf_use_stableadamw', v)} label="use_stableadamw" />
    </div>
`;

// AnimaLoraStudio loss weighting + InfoNoise + timestep extras. Replaces the
// "FlowMatch (Anima)" card for the Anima+LoKr combo since their engine has its
// own dialect that conflicts with kohya's.
const AnimaLokrTrainingPanel = ({ al, set, cfg }) => {
    const lw = al.loss_weighting || 'none';
    const ts = al.timestep_sampling || 'logit_normal';
    return html`
        <div class="card">
            <div class="card-title">Anima · LoKr · Loss + Timestep</div>
            <div class="grid-3">
                <${Field} label="loss_weighting" tipKey="anima_lokr.loss_weighting">
                    <${Select} value=${lw} onInput=${v => set('anima_lokr.loss_weighting', v)}
                        options=${['none','min_snr','detail_inv_t','cosmap']} />
                </${Field}>
                ${lw === 'min_snr' && html`
                    <${Field} label="min_snr_gamma" tipKey="training.min_snr_gamma">
                        <${Num} value=${cfg.training.min_snr_gamma ?? 5.0} step=${0.5} min=${0.1} max=${20}
                            onInput=${v => set('training.min_snr_gamma', v)} />
                    </${Field}>
                `}
                ${lw === 'detail_inv_t' && html`
                    <${Field} label="detail_inv_t_min" tipKey="anima_lokr.detail_inv_t_min">
                        <${Num} value=${al.detail_inv_t_min ?? 1.0} step=${0.1} min=${1.0}
                            onInput=${v => set('anima_lokr.detail_inv_t_min', v)} />
                    </${Field}>
                    <${Field} label="detail_inv_t_max" tipKey="anima_lokr.detail_inv_t_max">
                        <${Num} value=${al.detail_inv_t_max ?? 5.0} step=${0.1} min=${0.1}
                            onInput=${v => set('anima_lokr.detail_inv_t_max', v)} />
                    </${Field}>
                `}
                ${lw !== 'none' && html`
                    <${Field} label="weight_cap_ratio" tipKey="anima_lokr.weight_cap_ratio">
                        <${Num} value=${al.weight_cap_ratio ?? 0} step=${0.5} min=${0} max=${50}
                            onInput=${v => set('anima_lokr.weight_cap_ratio', v)} />
                    </${Field}>
                `}
            </div>
            <div class="grid-3" style="margin-top:6px;">
                <${Field} label="timestep_sampling" tipKey="anima_lokr.timestep_sampling">
                    <${Select} value=${ts} onInput=${v => set('anima_lokr.timestep_sampling', v)}
                        options=${['logit_normal','uniform','logit_normal_low','mode','mixed_uniform_low','mixed_uniform_logit']} />
                </${Field}>
                <${Field} label="timestep_shift" tipKey="anima_lokr.timestep_shift">
                    <${Num} value=${al.timestep_shift ?? 3.0} step=${0.1} min=${0.1} max=${10}
                        onInput=${v => set('anima_lokr.timestep_shift', v)} />
                </${Field}>
                ${(ts === 'mixed_uniform_low' || ts === 'mixed_uniform_logit') && html`
                    <${Field} label="mix_low_prob" tipKey="anima_lokr.timestep_mix_low_prob">
                        <${Num} value=${al.timestep_mix_low_prob ?? 0} step=${0.05} min=${0} max=${1}
                            onInput=${v => set('anima_lokr.timestep_mix_low_prob', v)} />
                    </${Field}>
                `}
                <${Field} label="timestep_schedule_shift" tipKey="anima_lokr.timestep_schedule_shift">
                    <${Num} value=${al.timestep_schedule_shift ?? 1.0} step=${0.1} min=${0.1} max=${10}
                        onInput=${v => set('anima_lokr.timestep_schedule_shift', v)} />
                </${Field}>
            </div>
            <div class="grid-3" style="margin-top:6px;">
                <${Field} label="noise_offset" tipKey="training.noise_offset">
                    <${Num} value=${cfg.training.noise_offset ?? 0} step=${0.001} min=${0} max=${0.2}
                        onInput=${v => set('training.noise_offset', v)} />
                </${Field}>
                <${Field} label="pyramid_noise_iters" tipKey="training.multires_noise_iterations">
                    <${Num} value=${cfg.training.multires_noise_iterations ?? 0} min=${0} max=${6}
                        onInput=${v => set('training.multires_noise_iterations', v)} />
                </${Field}>
                <${Field} label="pyramid_noise_discount" tipKey="training.multires_noise_discount">
                    <${Num} value=${cfg.training.multires_noise_discount ?? 0.35} step=${0.05} min=${0.1} max=${0.9}
                        onInput=${v => set('training.multires_noise_discount', v)} />
                </${Field}>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                Anima · InfoNoise (адаптивный timestep)
                <span style="margin-left:auto;">
                    <${Switch} value=${!!al.infonoise_enabled} onInput=${v => set('anima_lokr.infonoise_enabled', v)} />
                </span>
            </div>
            ${al.infonoise_enabled && html`
                <div class="grid-3">
                    <${Field} label="K (бины)" tipKey="anima_lokr.infonoise_K">
                        <${Num} value=${al.infonoise_K ?? 64} min=${16} max=${256}
                            onInput=${v => set('anima_lokr.infonoise_K', v)} />
                    </${Field}>
                    <${Field} label="N_warm (warmup шагов; 0=авто)" tipKey="anima_lokr.infonoise_N_warm">
                        <${Num} value=${al.infonoise_N_warm ?? 0} min=${0}
                            onInput=${v => set('anima_lokr.infonoise_N_warm', v)} />
                    </${Field}>
                    <${Field} label="M (период обновления)" tipKey="anima_lokr.infonoise_M">
                        <${Num} value=${al.infonoise_M ?? 100} min=${10}
                            onInput=${v => set('anima_lokr.infonoise_M', v)} />
                    </${Field}>
                    <${Field} label="B (FIFO ёмкость bin)" tipKey="anima_lokr.infonoise_B">
                        <${Num} value=${al.infonoise_B ?? 256} min=${32}
                            onInput=${v => set('anima_lokr.infonoise_B', v)} />
                    </${Field}>
                    <${Field} label="β (EMA новый вес)" tipKey="anima_lokr.infonoise_beta">
                        <${Num} value=${al.infonoise_beta ?? 0.9} step=${0.05} min=${0.1} max=${0.999}
                            onInput=${v => set('anima_lokr.infonoise_beta', v)} />
                    </${Field}>
                    <${Field} label="N_min (мин. сэмплов в bin)" tipKey="anima_lokr.infonoise_N_min">
                        <${Num} value=${al.infonoise_N_min ?? 50} min=${1}
                            onInput=${v => set('anima_lokr.infonoise_N_min', v)} />
                    </${Field}>
                </div>
            `}
        </div>
    `;
};

// =============================================================================
// Section: OPTIMIZER / TRAINING
// =============================================================================
// Interactive editor for piecewise_constant: drag handles to set levels;
// numbers (step/epoch · multiplier) are derived automatically.
const PiecewiseEditor = ({ cfg, set }) => {
    const totalSteps = computeTotalStepsLocal(cfg);
    const epochs = Math.max(1, cfg.training.max_train_epochs);
    const pts = cfg.optimizer.lr_piecewise || [];
    const svgRef = useRef(null);
    const [drag, setDrag] = useState(-1);

    const W = 560, H = 200, padX = 14, padTop = 14, padBot = 24;
    const plotW = W - padX * 2, plotH = H - padTop - padBot;
    const X = (at) => padX + Math.max(0, Math.min(1, at)) * plotW;
    const Y = (m) => padTop + (1 - Math.max(0, Math.min(1, m))) * plotH;
    const r2 = (v) => Math.round(v * 100) / 100;

    const toFrac = (evt) => {
        const rect = svgRef.current.getBoundingClientRect();
        const at = ((evt.clientX - rect.left) / rect.width * W - padX) / plotW;
        const m = 1 - ((evt.clientY - rect.top) / rect.height * H - padTop) / plotH;
        return { at: Math.max(0, Math.min(1, at)), mult: Math.max(0.01, Math.min(1, m)) };
    };
    const update = (arr) => set('optimizer.lr_piecewise',
        [...arr].sort((a, b) => a.at - b.at));

    const onDown = (i) => (e) => {
        e.stopPropagation();
        svgRef.current.setPointerCapture(e.pointerId);
        setDrag(i);
    };
    const onMove = (e) => {
        if (drag < 0) return;
        const f = toFrac(e);
        update(pts.map((p, i) => i === drag ? { at: r2(f.at), mult: r2(f.mult) } : p));
    };
    const onUp = () => setDrag(-1);
    const addPoint = (e) => {
        const f = toFrac(e);
        update([...pts, { at: r2(f.at), mult: r2(f.mult) }]);
    };
    const removePoint = (i) => (e) => {
        e.stopPropagation();
        if (pts.length > 2) update(pts.filter((_, j) => j !== i));
    };

    const sorted = [...pts].sort((a, b) => a.at - b.at);
    // step polyline
    const poly = [];
    let prev = sorted.length ? sorted[0].mult : 1;
    poly.push([0, prev]);
    for (const p of sorted) { poly.push([p.at, prev]); poly.push([p.at, p.mult]); prev = p.mult; }
    poly.push([1, prev]);
    const polyStr = poly.map(([x, m]) => `${X(x).toFixed(1)},${Y(m).toFixed(1)}`).join(' ');

    return html`
        <div class="card" style="margin-top:8px;">
            <div style="font-weight:600; margin-bottom:4px;">Кастомный график LR (piecewise)</div>
            <div class="dim" style="font-size:11.5px; margin-bottom:6px;">
                Тяни точки мышкой · клик по пустому месту — добавить точку · двойной клик по точке — удалить.
                Уровень держится от своей точки до следующей.
            </div>
            <svg ref=${svgRef} viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block; touch-action:none; cursor:crosshair;"
                onPointerMove=${onMove} onPointerUp=${onUp} onPointerLeave=${onUp}>
                <rect x=${padX} y=${padTop} width=${plotW} height=${plotH} fill="rgba(255,255,255,0.02)"
                    stroke="rgba(255,255,255,0.12)" stroke-width="1" onClick=${addPoint}/>
                ${[0.25, 0.5, 0.75].map(g => html`<line x1=${X(g)} y1=${padTop} x2=${X(g)} y2=${padTop + plotH}
                    stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`)}
                <polyline points=${polyStr} fill="none" stroke="#5fb04a" stroke-width="2" stroke-linejoin="round"/>
                ${pts.map((p, i) => html`<circle cx=${X(p.at)} cy=${Y(p.mult)} r="6"
                    fill=${drag === i ? '#ffd24a' : '#5fb04a'} stroke="#1c1c1c" stroke-width="1.5"
                    style="cursor:grab;" onPointerDown=${onDown(i)} onDblClick=${removePoint(i)}/>`)}
                <text x=${padX} y=${padTop - 3} fill="rgba(255,255,255,0.5)" font-size="9">LR ×</text>
                <text x=${W - padX} y=${H - 6} text-anchor="end" fill="rgba(255,255,255,0.5)" font-size="9">шаги · ${totalSteps}</text>
            </svg>
            <div class="dim" style="font-size:11.5px; display:flex; gap:8px; flex-wrap:wrap; margin-top:4px;">
                ${sorted.map(p => html`<span style="background:rgba(95,176,74,0.15); padding:2px 6px; border-radius:4px;">
                    эп ${Math.round(p.at * epochs)} · шаг ${Math.round(p.at * totalSteps)} → ×${p.mult}</span>`)}
            </div>
        </div>`;
};

const SectionTraining = ({ cfg, set, val }) => {
    const isAnima = cfg.model.arch === 'anima';
    // AnimaLoraStudio's engine only ships adamw / prodigy / ProdigyPlusScheduleFree.
    // Show ONLY those three when training Anima+LoKr; otherwise the full kohya list.
    const isAnimaLokr = isAnima && cfg.network.kind === 'lokr';
    const opts = isAnimaLokr
        ? ['AdamW', 'Prodigy', 'ProdigyPlusScheduleFree']
        : ['AdamW','AdamW8bit','Lion','Lion8bit','Prodigy','DAdaptation','DAdaptAdam','DAdaptLion',
           'SGDNesterov','SGDNesterov8bit','AdaFactor','PagedAdamW8bit','PagedLion8bit','pytorch_optimizer.CAME'];
    // AnimaLoraStudio's schema literal is {none, cosine, cosine_with_restart}.
    // Map "constant" → "none" in their YAML; the UI surfaces "constant" because
    // it's our wording. Hide everything outside their support set when in combo.
    const lrOpts = isAnimaLokr
        ? ['constant', 'cosine', 'cosine_with_restarts']
        : ['constant','constant_with_warmup','linear','cosine','cosine_with_restarts',
           'polynomial','adafactor','warmup_stable_decay','piecewise_constant'];
    const opt = cfg.optimizer.optimizer_type;
    const isPPSF = isAnimaLokr && opt === 'ProdigyPlusScheduleFree';
    const isProdigy = isAnimaLokr && opt === 'Prodigy';
    const al = cfg.anima_lokr || {};
    return html`
        <h2>Оптимизатор и обучение</h2>
        <div class="subtitle">LR, оптимизатор, расписание, длительность.</div>

        <div class="card">
            <div class="card-title">Оптимизатор</div>
            <div class="grid-3">
                <${Field} label="optimizer_type" tipKey="optimizer.optimizer_type">
                    <${Select} value=${cfg.optimizer.optimizer_type} onInput=${v => set('optimizer.optimizer_type', v)} options=${opts} />
                </${Field}>
                <${Field} label="learning_rate" tipKey="optimizer.learning_rate"
                    warn=${val.warnMap['optimizer.learning_rate']}>
                    <${Num} value=${cfg.optimizer.learning_rate} step=${1e-5}
                        warn=${!!val.warnMap['optimizer.learning_rate']}
                        onInput=${v => set('optimizer.learning_rate', v)} />
                </${Field}>
                <${Field} label="lr_scheduler" tipKey="optimizer.lr_scheduler">
                    <${Select} value=${cfg.optimizer.lr_scheduler} onInput=${v => {
                        set('optimizer.lr_scheduler', v);
                        // Drop stale scheduler-specific fields so they don't linger in the
                        // saved JSON and confuse the next reader.
                        if (v !== 'piecewise_constant') {
                            set('optimizer.lr_piecewise', [
                                { at: 0.0, mult: 1.0 },
                                { at: 0.5, mult: 0.3 },
                                { at: 0.8, mult: 0.1 },
                            ]);
                        }
                        if (v !== 'warmup_stable_decay') set('optimizer.lr_decay_steps', null);
                    }}
                        options=${lrOpts} />
                </${Field}>
            </div>
            <div class="grid-3">
                ${!isAnimaLokr && html`
                    <${Field} label="unet_lr" tipKey="optimizer.unet_lr">
                        <${Num} value=${cfg.optimizer.unet_lr ?? ''} step=${1e-5} onInput=${v => set('optimizer.unet_lr', v)} />
                    </${Field}>
                    <${Field} label="text_encoder_lr" tipKey="optimizer.text_encoder_lr">
                        <${Num} value=${cfg.optimizer.text_encoder_lr ?? ''} step=${1e-5} onInput=${v => set('optimizer.text_encoder_lr', v)} />
                    </${Field}>
                    <${Field} label="lr_warmup_steps" tipKey="optimizer.lr_warmup_steps">
                        <${Num} value=${cfg.optimizer.lr_warmup_steps} onInput=${v => set('optimizer.lr_warmup_steps', v)} />
                    </${Field}>
                    ${cfg.optimizer.lr_scheduler === 'warmup_stable_decay' && html`
                        <${Field} label="lr_decay_steps" tipKey="optimizer.lr_decay_steps">
                            <${Num} value=${cfg.optimizer.lr_decay_steps ?? ''} step=${0.05} min=${0}
                                onInput=${v => set('optimizer.lr_decay_steps', v)} />
                        </${Field}>`}
                    <${Field} label="lr_scheduler_num_cycles" tipKey="optimizer.lr_scheduler_num_cycles"
                        warn=${val.warnMap['optimizer.lr_scheduler_num_cycles']}>
                        <${Num} value=${cfg.optimizer.lr_scheduler_num_cycles} onInput=${v => set('optimizer.lr_scheduler_num_cycles', v)} />
                    </${Field}>
                    <${Field} label="lr_scheduler_power" tipKey="optimizer.lr_scheduler_power">
                        <${Num} value=${cfg.optimizer.lr_scheduler_power} step=${0.1} onInput=${v => set('optimizer.lr_scheduler_power', v)} />
                    </${Field}>
                `}
                <${Field} label="max_grad_norm" tipKey="optimizer.max_grad_norm">
                    <${Num} value=${cfg.optimizer.max_grad_norm ?? 1.0} step=${0.1} min=${0}
                        onInput=${v => set('optimizer.max_grad_norm', v)} />
                </${Field}>
            </div>
            ${!isAnimaLokr && cfg.optimizer.lr_scheduler === 'piecewise_constant' && html`<${PiecewiseEditor} cfg=${cfg} set=${set} />`}
            ${isAnimaLokr ? html`
                <${Field} label="weight_decay" tipKey="optimizer.optimizer_args">
                    <${Num} value=${(function(){
                        // Pull weight_decay out of kohya-style optimizer_args so users can
                        // tweak it as a plain number without editing a textarea.
                        const a = (cfg.optimizer.optimizer_args || []).find(s => /^\s*weight_decay\s*=/.test(s));
                        return a ? parseFloat(a.split('=')[1]) : 0;
                    })()} step=${0.001} min=${0}
                        onInput=${v => {
                            const rest = (cfg.optimizer.optimizer_args || []).filter(s => !/^\s*weight_decay\s*=/.test(s));
                            set('optimizer.optimizer_args', v > 0 ? [...rest, `weight_decay=${v}`] : rest);
                        }} />
                </${Field}>
                ${isProdigy && html`
                    <div class="grid-3" style="margin-top:6px;">
                        <${Field} label="prodigy_d_coef" tipKey="anima_lokr.prodigy_d_coef">
                            <${Num} value=${al.prodigy_d_coef ?? 1.0} step=${0.1} min=${0.1} max=${10}
                                onInput=${v => set('anima_lokr.prodigy_d_coef', v)} />
                        </${Field}>
                        <${Field} label="safeguard_warmup" tipKey="anima_lokr.prodigy_safeguard_warmup">
                            <${Switch} value=${!!al.prodigy_safeguard_warmup} onInput=${v => set('anima_lokr.prodigy_safeguard_warmup', v)} />
                        </${Field}>
                    </div>
                `}
                ${isPPSF && html`<${PPSFPanel} al=${al} set=${set} />`}
            ` : html`
                <${Field} label="optimizer_args" tipKey="optimizer.optimizer_args" columns="stack">
                    <textarea placeholder=${"weight_decay=0.1\nbetas=0.9,0.999"}
                        value=${(cfg.optimizer.optimizer_args || []).join('\n')}
                        onInput=${e => set('optimizer.optimizer_args',
                            e.target.value
                                .replace(/&#10;|&#xA;/gi, '\n')
                                .split(/\n+/)
                                .map(s => s.trim())
                                .filter(Boolean))}></textarea>
                </${Field}>
                <${Field} label="lr_scheduler_args" tipKey="optimizer.lr_scheduler_args" columns="stack">
                    <textarea placeholder=${"num_cycles=1\npower=1.0"}
                        value=${(cfg.optimizer.lr_scheduler_args || []).join('\n')}
                        onInput=${e => set('optimizer.lr_scheduler_args',
                            e.target.value.replace(/&#10;|&#xA;/gi, '\n').split(/\n+/).map(s => s.trim()).filter(Boolean))}></textarea>
                </${Field}>
            `}
        </div>

        <div class="card">
            <div class="card-title">Длительность и батч</div>
            <div class="grid-4">
                <${Field} label="train_batch_size" tipKey="training.train_batch_size"
                    err=${val.errMap['training.train_batch_size']}>
                    <${Num} value=${cfg.training.train_batch_size} min=${1}
                        onInput=${v => set('training.train_batch_size', v)} />
                </${Field}>
                <${Field} label="grad_accum_steps" tipKey="training.gradient_accumulation_steps">
                    <${Num} value=${cfg.training.gradient_accumulation_steps} min=${1}
                        onInput=${v => set('training.gradient_accumulation_steps', v)} />
                </${Field}>
                <${Field} label="max_train_epochs" tipKey="training.max_train_epochs"
                    err=${val.errMap['training.max_train_epochs']}>
                    <${Num} value=${cfg.training.max_train_epochs}
                        onInput=${v => set('training.max_train_epochs', v)} />
                </${Field}>
                <${Field} label="max_train_steps" tipKey="training.max_train_steps">
                    <${Num} value=${cfg.training.max_train_steps}
                        onInput=${v => set('training.max_train_steps', v)} />
                </${Field}>
            </div>
            <div class="grid-4">
                <${Field} label="save_every_n_epochs" tipKey="training.save_every_n_epochs">
                    <${Num} value=${cfg.training.save_every_n_epochs}
                        onInput=${v => set('training.save_every_n_epochs', v)} />
                </${Field}>
                <${Field} label="save_last_n_epochs" tipKey="training.save_last_n_epochs">
                    <${Num} value=${cfg.training.save_last_n_epochs}
                        onInput=${v => set('training.save_last_n_epochs', v)} />
                </${Field}>
                <${Field} label="seed" tipKey="training.seed">
                    <${Num} value=${cfg.training.seed} onInput=${v => set('training.seed', v)} />
                </${Field}>
                <${Field} label="mixed_precision" tipKey="training.mixed_precision"
                    warn=${val.warnMap['training.mixed_precision']}>
                    <${Select} value=${cfg.training.mixed_precision}
                        warn=${!!val.warnMap['training.mixed_precision']}
                        onInput=${v => set('training.mixed_precision', v)}
                        options=${['no','fp16','bf16']} />
                </${Field}>
                <${Field} label="save_precision" tipKey="training.save_precision">
                    <${Select} value=${cfg.training.save_precision} onInput=${v => set('training.save_precision', v)}
                        options=${['no','fp16','bf16']} />
                </${Field}>
            </div>
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                <${Check} value=${cfg.training.gradient_checkpointing} onInput=${v => set('training.gradient_checkpointing', v)} label="gradient_checkpointing" />
                <${Check} value=${cfg.training.sdpa} onInput=${v => set('training.sdpa', v)} label="sdpa" />
                <${Check} value=${cfg.training.xformers} onInput=${v => set('training.xformers', v)} label="xformers" />
                ${!isAnimaLokr && html`
                    <${Check} value=${cfg.training.mem_eff_attn} onInput=${v => set('training.mem_eff_attn', v)} label="mem_eff_attn" />
                    <${Check} value=${cfg.training.full_bf16} onInput=${v => set('training.full_bf16', v)} label="full_bf16" />
                    <${Check} value=${cfg.training.full_fp16} onInput=${v => set('training.full_fp16', v)} label="full_fp16" />
                    <${Check} value=${cfg.training.lowram} onInput=${v => set('training.lowram', v)} label="lowram" />
                    <${Check} value=${cfg.training.highvram} onInput=${v => set('training.highvram', v)} label="highvram" />
                    <${Check} value=${cfg.training.fused_backward_pass} onInput=${v => set('training.fused_backward_pass', v)} label="fused_backward_pass" />
                `}
            </div>
            ${!isAnimaLokr && html`
                <div class="grid-3" style="margin-top:6px;">
                    <${Field} label="vae_batch_size" tipKey="training.vae_batch_size">
                        <${Num} value=${cfg.training.vae_batch_size ?? 0} min=${0}
                            onInput=${v => set('training.vae_batch_size', v)} />
                    </${Field}>
                    <${Field} label="prior_loss_weight" tipKey="training.prior_loss_weight">
                        <${Num} value=${cfg.training.prior_loss_weight ?? 1.0} step=${0.1} min=${0}
                            onInput=${v => set('training.prior_loss_weight', v)} />
                    </${Field}>
                </div>
            `}
            <div class="dim" style="font-size:11.5px; margin-top:6px;">
                💡 Скорость: отключите <b>gradient_checkpointing</b> если VRAM хватает (+20–25% throughput).
                Включите <b>highvram</b> на A100/L4 чтобы держать модели в GPU между прогонами.
            </div>
        </div>

        <div class="card">
            <div class="card-title">Loss · Timestep · Save State</div>
            <div class="grid-3">
                <${Field} label="loss_type" tipKey="training.loss_type">
                    <${Select} value=${cfg.training.loss_type || 'l2'} onInput=${v => set('training.loss_type', v)}
                        options=${isAnimaLokr ? ['l2','huber'] : ['l2','l1','huber','smooth_l1']} />
                </${Field}>
                ${(cfg.training.loss_type === 'huber' || cfg.training.loss_type === 'smooth_l1') && html`
                    ${!isAnimaLokr && html`
                        <${Field} label="huber_schedule" tipKey="training.huber_schedule">
                            <${Select} value=${cfg.training.huber_schedule || 'snr'} onInput=${v => set('training.huber_schedule', v)}
                                options=${['snr','exponential','constant']} />
                        </${Field}>
                    `}
                    <${Field} label="huber_c" tipKey="training.huber_c">
                        <${Num} value=${cfg.training.huber_c ?? 0.1} step=${0.01} onInput=${v => set('training.huber_c', v)} />
                    </${Field}>
                `}
                ${!isAnimaLokr && html`
                    <${Field} label="min_timestep" tipKey="training.min_timestep">
                        <${Num} value=${cfg.training.min_timestep ?? 0} min=${0} max=${1000}
                            onInput=${v => set('training.min_timestep', v)} />
                    </${Field}>
                    <${Field} label="max_timestep" tipKey="training.max_timestep">
                        <${Num} value=${cfg.training.max_timestep ?? 1000} min=${0} max=${1000}
                            onInput=${v => set('training.max_timestep', v)} />
                    </${Field}>
                `}
            </div>
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                <${Check} value=${cfg.training.save_state} onInput=${v => set('training.save_state', v)} label="save_state (на каждом save_every_n_epochs)" />
                <${Check} value=${cfg.training.save_state_on_train_end} onInput=${v => set('training.save_state_on_train_end', v)} label="save_state_on_train_end" />
            </div>
            <${Field} label="resume (путь к state-папке)" tipKey="training.resume">
                <${Text} value=${cfg.training.resume || ''} placeholder="(пустое — старт с нуля)"
                    onInput=${v => set('training.resume', v || null)} />
            </${Field}>
        </div>

        ${isAnimaLokr && html`<${AnimaLokrTrainingPanel} al=${al} set=${set} cfg=${cfg} />`}

        <div class="card" style=${isAnimaLokr ? 'display:none' : ''}>
            <div class="card-title">${isAnima ? 'FlowMatch (Anima)' : 'Шум и Loss'}</div>
            ${isAnima
                ? html`
                    <div class="grid-3">
                        <${Field} label="weighting_scheme" tipKey="training.weighting_scheme">
                            <${Select} value=${cfg.training.weighting_scheme} onInput=${v => set('training.weighting_scheme', v)}
                                options=${['uniform','sigma_sqrt','logit_normal','mode','cosmap','none']} />
                        </${Field}>
                        <${Field} label="timestep_sampling" tipKey="training.timestep_sampling">
                            <${Select} value=${cfg.training.timestep_sampling} onInput=${v => set('training.timestep_sampling', v)}
                                options=${['sigma','uniform','sigmoid','shift','flux_shift','logit_normal']} />
                        </${Field}>
                        <${Field} label="discrete_flow_shift" tipKey="training.discrete_flow_shift">
                            <${Num} value=${cfg.training.discrete_flow_shift} step=${0.1} onInput=${v => set('training.discrete_flow_shift', v)} />
                        </${Field}>
                        <${Field} label="logit_mean">
                            <${Num} value=${cfg.training.logit_mean} step=${0.1} onInput=${v => set('training.logit_mean', v)} />
                        </${Field}>
                        <${Field} label="logit_std">
                            <${Num} value=${cfg.training.logit_std} step=${0.1} onInput=${v => set('training.logit_std', v)} />
                        </${Field}>
                        <${Field} label="mode_scale">
                            <${Num} value=${cfg.training.mode_scale} step=${0.01} onInput=${v => set('training.mode_scale', v)} />
                        </${Field}>
                        <${Field} label="sigmoid_scale">
                            <${Num} value=${cfg.training.sigmoid_scale} step=${0.1} onInput=${v => set('training.sigmoid_scale', v)} />
                        </${Field}>
                    </div>
                `
                : html`
                    <div class="grid-3">
                        <${Field} label="min_snr_gamma" tipKey="training.min_snr_gamma">
                            <${Num} value=${cfg.training.min_snr_gamma ?? ''} step=${0.5} onInput=${v => set('training.min_snr_gamma', v)} />
                        </${Field}>
                        <${Field} label="noise_offset" tipKey="training.noise_offset">
                            <${Num} value=${cfg.training.noise_offset ?? ''} step=${0.001} onInput=${v => set('training.noise_offset', v)} />
                        </${Field}>
                        <${Field} label="adaptive_noise_scale" tipKey="training.adaptive_noise_scale">
                            <${Num} value=${cfg.training.adaptive_noise_scale ?? ''} step=${0.001} onInput=${v => set('training.adaptive_noise_scale', v)} />
                        </${Field}>
                        <${Field} label="multires_noise_iterations" tipKey="training.multires_noise_iterations">
                            <${Num} value=${cfg.training.multires_noise_iterations ?? ''} onInput=${v => set('training.multires_noise_iterations', v)} />
                        </${Field}>
                        <${Field} label="multires_noise_discount" tipKey="training.multires_noise_discount">
                            <${Num} value=${cfg.training.multires_noise_discount ?? ''} step=${0.05} onInput=${v => set('training.multires_noise_discount', v)} />
                        </${Field}>
                        <${Field} label="ip_noise_gamma" tipKey="training.ip_noise_gamma">
                            <${Num} value=${cfg.training.ip_noise_gamma ?? ''} step=${0.01} onInput=${v => set('training.ip_noise_gamma', v)} />
                        </${Field}>
                    </div>
                    <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                        <${Check} value=${cfg.training.debiased_estimation_loss} onInput=${v => set('training.debiased_estimation_loss', v)} label="debiased_estimation_loss" />
                        <${Check} value=${cfg.training.zero_terminal_snr} onInput=${v => set('training.zero_terminal_snr', v)} label="zero_terminal_snr" />
                        <${Check} value=${cfg.training.noise_offset_random_strength} onInput=${v => set('training.noise_offset_random_strength', v)} label="noise_offset_random_strength" />
                        <${Check} value=${cfg.training.ip_noise_gamma_random_strength} onInput=${v => set('training.ip_noise_gamma_random_strength', v)} label="ip_noise_gamma_random_strength" />
                    </div>
                `
            }
        </div>
    `;
};

// =============================================================================
// Section: SAMPLES
// =============================================================================
const SectionSamples = ({ cfg, set, val }) => {
    const s = cfg.samples;
    return html`
        <h2>Сэмплы</h2>
        <div class="subtitle">Генерация превью-картинок прямо во время обучения.</div>

        <div class="card">
            <div class="card-title">
                Расписание
                <span style="margin-left:auto;">
                    <${Switch} value=${s.enable} onInput=${v => set('samples.enable', v)} />
                </span>
            </div>
            <div class="grid-3">
                <${Field} label="every_n_epochs" tipKey="samples.every_n_epochs">
                    <${Num} value=${s.every_n_epochs} onInput=${v => set('samples.every_n_epochs', v)} />
                </${Field}>
                <${Field} label="every_n_steps" tipKey="samples.every_n_steps">
                    <${Num} value=${s.every_n_steps} onInput=${v => set('samples.every_n_steps', v)} />
                </${Field}>
                <${Field} label="sampler" tipKey="samples.sampler">
                    <${Select} value=${s.sampler} onInput=${v => set('samples.sampler', v)}
                        options=${['ddim','pndm','lms','euler','euler_a','heun','dpm_2','dpm_2_a','dpmsolver','dpmsolver++','k_lms','k_euler','k_euler_a']} />
                </${Field}>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                Промпты
                <span class="hint">— ${s.prompts.length} шт.</span>
                <div style="margin-left:auto;">
                    <${Btn} size="sm" variant="ghost" onClick=${() => set('samples.prompts', [
                        ...(s.prompts || []),
                        { text: '', negative: '(worst quality, low quality:1.2)', width: 1024, height: 1024, steps: 24, scale: 5.0, seed: 42, sampler: s.sampler }
                    ])}>+ Промпт</${Btn}>
                </div>
            </div>
            ${val.warnMap['samples.prompts'] && html`<div class="field-warn" style="margin-bottom:8px;">${val.warnMap['samples.prompts']}</div>`}
            ${s.prompts.map((p, i) => html`
                <div class="card" style="background: var(--bg-2); margin-bottom:8px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                        <span class="tag">#${i + 1}</span>
                        <span class="dim mono">w×h=${p.width}×${p.height} · steps=${p.steps} · cfg=${p.scale} · seed=${p.seed}</span>
                        <div style="margin-left:auto;">
                            <${Btn} size="sm" variant="ghost" onClick=${() => set('samples.prompts', s.prompts.filter((_, j) => j !== i))}>✕</${Btn}>
                        </div>
                    </div>
                    <${Field} label="prompt" tipKey="samples.prompt.text" columns="stack">
                        <textarea value=${p.text} onInput=${e => set(`samples.prompts.${i}.text`, e.target.value)}></textarea>
                    </${Field}>
                    <${Field} label="negative" tipKey="samples.prompt.negative" columns="stack">
                        <textarea style="min-height:36px;" value=${p.negative} onInput=${e => set(`samples.prompts.${i}.negative`, e.target.value)}></textarea>
                    </${Field}>
                    <div class="grid-4">
                        <${Field} label="width"><${Num} value=${p.width} step=${64} onInput=${v => set(`samples.prompts.${i}.width`, v)} /></${Field}>
                        <${Field} label="height"><${Num} value=${p.height} step=${64} onInput=${v => set(`samples.prompts.${i}.height`, v)} /></${Field}>
                        <${Field} label="steps" tipKey="samples.prompt.steps"><${Num} value=${p.steps} onInput=${v => set(`samples.prompts.${i}.steps`, v)} /></${Field}>
                        <${Field} label="cfg scale" tipKey="samples.prompt.scale"><${Num} value=${p.scale} step=${0.1} onInput=${v => set(`samples.prompts.${i}.scale`, v)} /></${Field}>
                        <${Field} label="seed" tipKey="samples.prompt.seed"><${Num} value=${p.seed} onInput=${v => set(`samples.prompts.${i}.seed`, v)} /></${Field}>
                    </div>
                </div>
            `)}
        </div>
    `;
};

// =============================================================================
// Section: GALLERY (samples grid)
// =============================================================================
const SectionGallery = ({ cfg, samples, refresh }) => {
    const [zoom, setZoom] = useState(null);
    const [syncing, setSyncing] = useState(false);
    const [syncMsg, setSyncMsg] = useState('');
    const [syncEnabled, setSyncEnabled] = useState(false);

    useEffect(() => {
        api.syncStatus().then(s => { setSyncEnabled(s.enabled); }).catch(() => {});
    }, []);

    const pushNow = async () => {
        try {
            setSyncing(true); setSyncMsg('запущено…');
            const r = await api.syncPush();
            if (r.skipped) setSyncMsg('output уже на Drive — синк не нужен');
            else if (r.already_running) setSyncMsg('уже идёт фоновый синк');
            else setSyncMsg('запущено в фоне, ~30-120 с');
            // poll until finished
            const t = setInterval(async () => {
                try {
                    const s = await api.syncStatus();
                    if (!s.running) {
                        clearInterval(t);
                        setSyncing(false);
                        setSyncMsg(s.last_msg === 'ok' ? '✓ синхронизировано' : `✗ ${s.last_msg || 'неизвестно'}`);
                    }
                } catch { clearInterval(t); setSyncing(false); }
            }, 1500);
        } catch (e) { setSyncing(false); setSyncMsg(`✗ ${e.message}`); }
    };

    // Group by prompt index; within each prompt show a wrapping grid of
    // sample tiles sorted by epoch descending (newest first). Tile label is
    // "ep N · step M". Avoids the giant horizontal-scrolling matrix.
    const { groups, promptLabels } = useMemo(() => {
        const promptCount = Math.max(1, cfg.samples.prompts?.length || 1);
        const promptLabels = (cfg.samples.prompts || []).map((p, i) =>
            `#${i + 1} · ${(p.text || '').slice(0, 60)}${(p.text || '').length > 60 ? '…' : ''}`);
        while (promptLabels.length < promptCount) promptLabels.push(`#${promptLabels.length + 1}`);

        const epochCounters = {};
        const buckets = Array.from({ length: promptCount }, () => []);
        samples
            .slice()
            .sort((a, b) => (a.epoch - b.epoch) || a.name.localeCompare(b.name))
            .forEach(s => {
                const ep = s.epoch || 0;
                // Prefer the prompt index parsed from the filename; fall back
                // to round-robin within the epoch.
                let idx;
                const tag = parseInt(s.prompt_tag, 10);
                if (!Number.isNaN(tag) && tag >= 0 && tag < promptCount) {
                    idx = tag;
                } else {
                    idx = (epochCounters[ep] = (epochCounters[ep] ?? -1) + 1) % promptCount;
                }
                buckets[idx].push(s);
            });
        // newest epoch first inside each prompt bucket
        buckets.forEach(b => b.sort((a, b) => (b.epoch - a.epoch) || (b.step - a.step)));
        const groups = buckets.map((tiles, i) => ({ idx: i, label: promptLabels[i], tiles }));
        return { groups, promptLabels };
    }, [samples, cfg.samples.prompts]);

    return html`
        <h2>Галерея сэмплов</h2>
        <div class="subtitle">
            ${samples.length} картинок · группировка по промптам, новые эпохи сверху.
            <button class="btn btn-sm btn-ghost" style="margin-left:8px;" onClick=${refresh}>↻ Обновить</button>
            ${syncEnabled && html`
                <button class="btn btn-sm btn-ghost" style="margin-left:4px;"
                    disabled=${syncing} onClick=${pushNow}
                    title="Скинуть локальный output (чекпоинты + сэмплы) на Google Drive">
                    ${syncing ? '⟳ Sync…' : '↑ Push to Drive'}
                </button>
                ${syncMsg && html`<span class="dim" style="margin-left:8px;font-size:11px;">${syncMsg}</span>`}
            `}
        </div>

        ${samples.length === 0
            ? html`<div class="card dim">Пока ничего не сгенерировано. Сэмплы появятся здесь по мере прохождения эпох.</div>`
            : html`
                <div class="gallery-shell">
                    ${groups.filter(g => g.tiles.length > 0).map(g => html`
                        <section class="gallery-group" key=${g.idx}>
                            <header class="gallery-group-head" title=${g.label}>${g.label}</header>
                            <div class="gallery-tiles">
                                ${g.tiles.map(s => html`
                                    <figure class="gallery-tile" key=${s.path} onClick=${() => setZoom(s)}>
                                        <img src=${api.thumbUrl(s.path, 256)}
                                            loading="lazy" decoding="async"
                                            title=${`epoch ${s.epoch} · step ${s.step} · ${s.name}`} />
                                        <figcaption>ep ${s.epoch}${s.step ? ` · ${s.step}` : ''}</figcaption>
                                    </figure>
                                `)}
                            </div>
                        </section>
                    `)}
                </div>
            `}

        ${zoom && html`
            <div class="image-viewer" onClick=${() => setZoom(null)}>
                <img src=${api.fileUrl(zoom.path)} />
                <div class="meta">
                    ${zoom.name} · epoch ${zoom.epoch} · ${fmtBytes(zoom.size)} · ${fmtTime(zoom.mtime)}
                </div>
            </div>
        `}
    `;
};

// =============================================================================
// Section: LOGS
// =============================================================================
const SectionLogs = ({ logs, onClear }) => {
    const ref = useRef(null);
    useEffect(() => {
        if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
    }, [logs]);
    return html`
        <h2>Логи обучения</h2>
        <div class="subtitle">
            ${logs.length} строк
            <button class="btn btn-sm btn-ghost" style="margin-left:8px;" onClick=${onClear}>Очистить</button>
        </div>
        <div class="logs" ref=${ref}>
            ${logs.map((l, i) => {
                const t = new Date(l.ts * 1000).toLocaleTimeString('ru-RU');
                const cls = l.stream === 'system' ? 'l-system'
                          : /error|traceback|\bExit\b/i.test(l.line || '') ? 'l-err'
                          : /warn/i.test(l.line || '') ? 'l-warn'
                          : '';
                return html`<div key=${i}><span class="ts">${t}</span><span class=${cls}>${l.line}</span></div>`;
            })}
        </div>
    `;
};

// =============================================================================
// Section: FILES (outputs)
// =============================================================================
const SectionFiles = ({ outputs, refresh, cfg }) => html`
    <h2>Файлы модели</h2>
    <div class="subtitle">
        Каталог: <span class="mono">${cfg.paths.output_root || '—'}</span>
        <button class="btn btn-sm btn-ghost" style="margin-left:8px;" onClick=${refresh}>↻ Обновить</button>
    </div>

    <div class="card" style="padding:0;">
        <div class="file-row" style="background: var(--bg-2); color: var(--text-mute); font-size: 11px;">
            <div>Имя</div><div>Размер</div><div>Изменён</div><div>Эпоха</div>
        </div>
        ${outputs.length === 0
            ? html`<div style="padding:14px; color: var(--text-dim);">Файлов модели пока нет.</div>`
            : outputs.map(o => {
                const m = o.name.match(/[-_]?(?:e|epoch[_-]?)(\d+)/i) || o.name.match(/[-_](\d+)\./);
                const ep = m ? m[1] : '—';
                return html`
                    <div class="file-row" key=${o.path}>
                        <div class="name mono" title=${o.path}>${o.name}</div>
                        <div class="size">${fmtBytes(o.size)}</div>
                        <div class="mtime">${fmtTime(o.mtime)}</div>
                        <div class="epoch">${ep}</div>
                    </div>
                `;
            })}
    </div>
`;

// =============================================================================
// Section: PRESETS
// =============================================================================
const SectionPresets = ({ cfg, applyPatch, replaceCfg, presets, refresh }) => {
    const [name, setName] = useState('');
    const [desc, setDesc] = useState('');
    const fileRef = useRef(null);

    const onSave = async () => {
        if (!name.trim()) { toast('warn', 'Введите имя пресета'); return; }
        try {
            await api.presetsSave({ name: name.trim(), description: desc.trim(), config: cleanForExport(cfg) });
            await refresh();
            toast('ok', 'Пресет сохранён');
            setName(''); setDesc('');
        } catch (e) { toast('err', String(e.message)); }
    };

    const onExport = () => {
        const cleaned = cleanForExport(cfg);
        const blob = new Blob([JSON.stringify(cleaned, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${cfg.project.output_name || 'lora'}-config.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const onImportFile = async (file) => {
        const isSafetensors = /\.safetensors$/i.test(file.name);
        if (isSafetensors) {
            try {
                const { patch, warnings } = await loraFileToPatch(file);
                replaceCfg(deepMerge(cfg, patch));
                toast('ok', `Настройки восстановлены из чекпоинта${patch.project?.output_name ? ` (${patch.project.output_name})` : ''}`);
                warnings.forEach(w => toast('warn', w));
            } catch (e) { toast('err', 'Не удалось прочитать чекпоинт: ' + e.message); }
            return;
        }
        try {
            const text = await file.text();
            const parsed = JSON.parse(text);
            // Either a TrainConfig or a Preset (with .config inside)
            const next = parsed.config && parsed.created_at ? parsed.config : parsed;
            // Deep-merge over the current config so a partial / cleaned JSON
            // (e.g. an arch-stripped export) can't leave required sections
            // undefined and crash the render.
            replaceCfg(deepMerge(cfg, next));
            toast('ok', 'Конфиг импортирован');
        } catch (e) { toast('err', 'Не удалось разобрать JSON: ' + e.message); }
    };

    return html`
        <h2>Шаблоны (пресеты)</h2>
        <div class="subtitle">Сохраняйте и переключайте конфигурации, экспортируйте в JSON. Импорт принимает JSON-конфиг или готовый <code>.safetensors</code> чекпоинт LoRA — настройки тренировки восстановятся из метадаты.</div>

        <div class="card">
            <div class="card-title">Встроенные стартовые пресеты</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                ${Object.entries(BUILTIN_PRESETS).map(([key, p]) => html`
                    <${Btn} key=${key} variant="ghost" onClick=${() => applyPatch(p)}>${p.label}</${Btn}>
                `)}
            </div>
            <div class="dim" style="margin-top:6px; font-size:11.5px;">Пресет накладывается поверх текущих настроек.</div>
        </div>

        <div class="card">
            <div class="card-title">Мои пресеты <span class="hint">(${presets.length})</span></div>
            ${presets.length === 0
                ? html`<div class="dim">Пока нет сохранённых пресетов.</div>`
                : presets.map(p => html`
                    <div class="file-row" key=${p.name}>
                        <div class="name"><b>${p.name}</b>${p.description ? html`<span class="dim"> — ${p.description}</span>` : ''}</div>
                        <div class="size dim">${fmtTime(p.created_at)}</div>
                        <div></div>
                        <div style="display:flex; gap:4px;">
                            <button class="btn btn-sm btn-ghost" onClick=${() => replaceCfg(p.config)}>Применить</button>
                            <button class="btn btn-sm btn-ghost" onClick=${async () => {
                                if (!confirm(`Удалить пресет "${p.name}"?`)) return;
                                await api.presetsDelete(p.name);
                                await refresh();
                                toast('ok', 'Пресет удалён');
                            }}>✕</button>
                        </div>
                    </div>
                `)}
        </div>

        <div class="card">
            <div class="card-title">Сохранить текущий конфиг</div>
            <${Field} label="name"><${Text} value=${name} onInput=${setName} placeholder="my-lora-noobai-v3" /></${Field}>
            <${Field} label="description"><${Text} value=${desc} onInput=${setDesc} placeholder="опционально" /></${Field}>
            <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
                <${Btn} variant="primary" onClick=${onSave}>💾 Сохранить пресет</${Btn}>
                <${Btn} variant="ghost" onClick=${onExport}>↓ Экспорт JSON</${Btn}>
                <${Btn} variant="ghost" onClick=${() => fileRef.current?.click()}>↑ Импорт JSON / LoRA</${Btn}>
                <input ref=${fileRef} type="file" accept="application/json,.json,.safetensors" style="display:none;"
                    onChange=${e => { if (e.target.files[0]) { onImportFile(e.target.files[0]); e.target.value = ''; } }} />
            </div>
        </div>
    `;
};

// =============================================================================
// Right stats panel
// =============================================================================
const LrChart = ({ cfg, totalSteps, status, lrTrace }) => {
    const W = 300, H = 140, padX = 10, padTop = 10, padBot = 18;
    const plotW = W - padX * 2, plotH = H - padTop - padBot;
    const X = (xf) => padX + Math.max(0, Math.min(1, xf)) * plotW;
    const Y = (yn) => padTop + (1 - Math.max(0, Math.min(1, yn))) * plotH;

    const sched = cfg.optimizer.lr_scheduler;
    const pts = lrCurvePoints(cfg, totalSteps);
    const epochs = Math.max(1, cfg.training.max_train_epochs);
    const saveEvery = cfg.training.save_every_n_epochs || 0;
    const peakLr = Math.max(cfg.optimizer.learning_rate || 0,
                            cfg.optimizer.unet_lr || 0,
                            cfg.optimizer.text_encoder_lr || 0) || 1e-9;

    if (sched === 'adafactor' || pts.length === 0) {
        return html`<div class="dim" style="font-size:11.5px;">
            График LR недоступен для adafactor (LR задаётся самим оптимизатором).
        </div>`;
    }

    const planned = pts.map(p => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');

    // save-point ticks (skip if too dense)
    const saveLines = [];
    if (saveEvery > 0 && Math.floor(epochs / saveEvery) <= 50) {
        for (let e = saveEvery; e <= epochs; e += saveEvery) {
            const xf = e / epochs;
            saveLines.push(X(xf));
        }
    }

    // actual LR trace (normalized by peak)
    const trace = (lrTrace || []).filter(p => p.lr != null && p.step > 0);
    const actual = trace.length >= 2
        ? trace.map(p => `${X(p.step / totalSteps).toFixed(1)},${Y(p.lr / peakLr).toFixed(1)}`).join(' ')
        : null;

    // current position
    const running = status.state === 'running' || status.state === 'starting';
    const curX = totalSteps > 0 && status.step > 0 ? X(status.step / totalSteps) : null;
    const curY = curX != null && status.lr != null ? Y(status.lr / peakLr) : null;
    const lrText = status.lr?.toExponential?.(2) ?? null;
    const labelX = curX != null ? Math.max(padX + 14, Math.min(W - padX - 14, curX)) : 0;

    return html`
        <svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block;">
            <rect x=${padX} y=${padTop} width=${plotW} height=${plotH} fill="none"
                stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
            ${saveLines.map(x => html`<line x1=${x} y1=${padTop} x2=${x} y2=${padTop + plotH}
                stroke="rgba(255,255,255,0.10)" stroke-width="1"/>`)}
            <polyline points=${planned} fill="none" stroke="#5fb04a" stroke-width="2"
                stroke-linejoin="round"/>
            ${actual && html`<polyline points=${actual} fill="none" stroke="#e8a33d"
                stroke-width="1.6" stroke-linejoin="round" opacity="0.9"/>`}
            ${curX != null && html`<line x1=${curX} y1=${padTop} x2=${curX} y2=${padTop + plotH}
                stroke="#ff5d5d" stroke-width="1.5"/>`}
            ${curY != null && html`<circle cx=${curX} cy=${curY} r="2.6" fill="#ff5d5d"/>`}
            ${lrText && html`<text x=${labelX} y=${padTop + 9} text-anchor="middle"
                fill="#ff8a8a" font-size="9" font-weight="bold">${lrText}</text>`}
            <text x=${padX} y=${padTop - 2} fill="rgba(255,255,255,0.5)" font-size="9">LR</text>
            <text x=${W - padX} y=${H - 5} text-anchor="end" fill="rgba(255,255,255,0.5)"
                font-size="9">шаги · ${totalSteps}</text>
        </svg>
        <div class="dim" style="font-size:11px; display:flex; gap:12px; flex-wrap:wrap; margin-top:2px;">
            <span style="color:#5fb04a;">— план (${sched})</span>
            ${actual && html`<span style="color:#e8a33d;">— факт</span>`}
            ${curX != null && html`<span style="color:#ff5d5d;">| сейчас ${Math.round(status.step / totalSteps * 100)}%${lrText ? ` · LR ${lrText}` : ''}</span>`}
        </div>
    `;
};

const StatPanel = ({ cfg, status, totalSteps, etaSec, lrTrace }) => {
    const pct = totalSteps > 0 ? Math.min(100, (status.step / totalSteps) * 100) : 0;
    const stateLabel = {
        idle: 'не запущено',
        starting: 'старт…',
        running: 'идёт обучение',
        stopping: 'останавливается…',
        finished: 'завершено',
        error: 'ошибка',
    }[status.state] || status.state;

    return html`
        <div class="stat-card">
            <div style="display:flex; align-items:center; justify-content: space-between;">
                <div style="font-weight:600;">Статус</div>
                <div class=${`state-pill ${status.state}`}>
                    <span class="dot"></span>${stateLabel}
                </div>
            </div>
            <div class="progress"><div class="fill" style=${`width:${pct}%`}></div></div>
            <div class="stat-row"><span class="label">Шаг</span><span class="val tabnum">${status.step}/${totalSteps || '?'}</span></div>
            <div class="stat-row"><span class="label">Эпоха</span><span class="val tabnum">${status.epoch}/${cfg.training.max_train_epochs}</span></div>
            <div class="stat-row"><span class="label">Loss</span><span class="val tabnum">${status.loss?.toFixed?.(4) ?? '—'}</span></div>
            <div class="stat-row"><span class="label">LR</span><span class="val tabnum">${status.lr?.toExponential?.(2) ?? '—'}</span></div>
            <div class="stat-row"><span class="label">ETA</span><span class="val tabnum">${fmtDuration(etaSec ?? status.eta_seconds)}</span></div>
        </div>

        <div class="stat-card">
            <div style="font-weight:600; margin-bottom:6px;">Расчёт</div>
            <div class="stat-row"><span class="label">Total steps</span><span class="val accent tabnum">${totalSteps}</span></div>
            <div class="stat-row"><span class="label">Steps/epoch</span><span class="val tabnum">
                ${cfg.training.max_train_epochs > 0 ? Math.floor(totalSteps / cfg.training.max_train_epochs) : 0}
            </span></div>
            <div class="stat-row"><span class="label">Eff. batch</span><span class="val tabnum">
                ${cfg.training.train_batch_size * cfg.training.gradient_accumulation_steps}
            </span></div>
            <div class="stat-row"><span class="label">Эпох</span><span class="val tabnum">${cfg.training.max_train_epochs}</span></div>
        </div>

        <div class="stat-card">
            <div style="font-weight:600; margin-bottom:6px;">GPU / Colab</div>
            <div class="dim" style="font-size:11.5px;">
                Оценка времени берётся из реальных шагов в логах. До старта обучения ETA = «—».
            </div>
        </div>

        <div class="stat-card">
            <div style="font-weight:600; margin-bottom:6px;">График LR</div>
            <${LrChart} cfg=${cfg} totalSteps=${totalSteps} status=${status} lrTrace=${lrTrace} />
        </div>
    `;
};

// =============================================================================
// Topbar
// =============================================================================
const TopBar = ({ cfg, status, onStart, onStop, dirty, onSaveConfig, onCleanup }) => html`
    <div class="topbar">
        <div class="brand">
            <span class="dot"></span>
            <span>LoRA Trainer</span>
            <span class="dim">· Colab</span>
        </div>
        <div class="sep"></div>
        <div class="meta">
            <span class="mono">${cfg.model.arch === 'anima' ? 'Anima · DiT' : 'SDXL'}</span>
            <span class="dim"> · </span>
            <span class="mono">${cfg.network.kind}</span>
            <span class="dim"> · dim=</span><span class="mono">${cfg.network.network_dim}</span>
            <span class="dim"> α=</span><span class="mono">${cfg.network.network_alpha}</span>
        </div>
        <div class="grow"></div>
        <div class="actions">
            ${dirty && html`<span class="state-pill" title="несохранённые изменения"><span class="dot" style="background: var(--warn);"></span>изменения</span>`}
            <button class="btn btn-ghost" onClick=${onCleanup} title="Удалить сэмплы, LoRA, .lora_trainer и кэш миниатюр (локально, Drive не трогается)" disabled=${status.state === 'running' || status.state === 'starting'}>Очистка файлов</button>
            <button class="btn btn-ghost" onClick=${onSaveConfig}>Сохранить</button>
            ${status.state === 'running' || status.state === 'starting'
                ? html`<button class="btn btn-danger" onClick=${onStop}>■ Стоп</button>`
                : html`<button class="btn btn-primary" onClick=${onStart}>▶ Старт</button>`}
        </div>
    </div>
`;

// =============================================================================
// Root App
// =============================================================================
const SECTIONS = [
    { id: 'project',  label: 'Проект',      icon: '◇' },
    { id: 'model',    label: 'Модель',      icon: '◈' },
    { id: 'dataset',  label: 'Датасет',     icon: '⊞' },
    { id: 'network',  label: 'Сеть LoRA',   icon: '⊕' },
    { id: 'training', label: 'Обучение',    icon: '↻' },
    { id: 'samples',  label: 'Сэмплы',      icon: '⊡' },
    { id: 'gallery',  label: 'Галерея',     icon: '▦' },
    { id: 'logs',     label: 'Логи',        icon: '≡' },
    { id: 'files',    label: 'Файлы',       icon: '⎘' },
    { id: 'presets',  label: 'Шаблоны',     icon: '★' },
];

// AnimaLoraStudio's recommended defaults for the Anima + LoKr combo. Each
// entry maps to a path in cfg. Only fields whose CURRENT value still equals
// the OLD (kohya) default get overwritten — anything the user already touched
// stays put. Triggered on the first transition into (anima, lokr) per session.
const ANIMA_LOKR_DEFAULTS = [
    ['optimizer.optimizer_type',    'AdamW',        'AdamW'],
    ['optimizer.learning_rate',     1e-4,           1e-4],
    ['optimizer.lr_scheduler',      'cosine',       'constant'],   // their "none"
    ['network.network_dim',         32,             32],
    ['network.network_alpha',       16.0,           32.0],         // their default = rank
    ['network.factor',              -1,             8],
    ['network.weight_decompose',    false,          false],
    ['network.rs_lora',             false,          false],
    ['network.network_train_unet_only', false,      true],         // their engine = unet only
    ['training.train_batch_size',   2,              1],
    ['training.gradient_accumulation_steps', 1,     4],
    ['training.max_train_epochs',   10,             10],
    ['training.save_every_n_epochs',1,              2],
    ['training.mixed_precision',    'bf16',         'bf16'],
    ['training.sdpa',               true,           true],
    ['training.xformers',           false,          false],
    ['training.noise_offset',       0.0357,         0],
    ['training.min_snr_gamma',      5.0,            5.0],
    ['training.timestep_sampling',  'shift',        'logit_normal'],
    ['training.discrete_flow_shift',3.0,            3.0],
    ['dataset.flip_aug',            false,          true],
    ['dataset.cache_latents',       true,           true],
    ['dataset.cache_latents_to_disk', true,         true],
    ['dataset.max_data_loader_n_workers', 8,        8],            // Colab: keep our higher default
    ['dataset.persistent_data_loader_workers', true, true],
];

const App = () => {
    const [cfg, setCfg] = useState(null);
    // One-shot guard so re-renders or save bounces don't re-apply defaults.
    const animaLokrApplied = useRef(false);
    const [section, setSection] = useState('project');
    const [status, setStatus] = useState({ state: 'idle', step: 0, total_steps: 0, epoch: 0 });
    const [lrTrace, setLrTrace] = useState([]);
    const [logs, setLogs] = useState([]);
    const [presets, setPresets] = useState([]);
    const [outputs, setOutputs] = useState([]);
    const [samples, setSamples] = useState([]);
    const [models, setModels] = useState([]);
    const [scanResult, setScanResult] = useState([]);
    const [dirty, setDirty] = useState(false);
    const saveTimer = useRef(null);
    // Latest cfg for callbacks that must NOT re-subscribe on every edit (the
    // WebSocket handler). Updated on every render; read via .current.
    const cfgRef = useRef(null);
    cfgRef.current = cfg;

    // initial load
    useEffect(() => { (async () => {
        try {
            const loaded = await api.getConfig();
            setCfg(loaded);
            await Promise.all([refreshPresets(), refreshOutputs(loaded), refreshSamples(loaded), refreshModels(loaded)]);
        } catch (e) { toast('err', 'Не удалось загрузить конфиг: ' + e.message); }
    })(); }, []);

    // ws
    useEffect(() => {
        const close = openSocket((ev) => {
            if (ev.type === 'snapshot') {
                if (ev.status) setStatus(ev.status);
                if (ev.tail) setLogs(ev.tail);
                return;
            }
            if (ev.type === 'status') {
                setStatus(ev.status);
                const st = ev.status;
                if (st && st.step > 0 && st.lr != null) {
                    setLrTrace(prev => {
                        const last = prev[prev.length - 1];
                        if (last && last.step === st.step) return prev;
                        if (last && st.step < last.step) return [{ step: st.step, lr: st.lr }]; // new run
                        const next = [...prev, { step: st.step, lr: st.lr }];
                        return next.length > 2000 ? next.slice(-2000) : next;
                    });
                }
            }
            if (ev.type === 'log') setLogs(prev => {
                const next = [...prev, ev];
                return next.length > 5000 ? next.slice(-5000) : next;
            });
            // when training emits a save line, refresh outputs/samples
            if (ev.type === 'log' && /saving (checkpoint|state)|saved model|saving images/i.test(ev.line || '')) {
                const c = cfgRef.current;
                if (c) { refreshOutputs(c); refreshSamples(c); }
            }
            // sample generation lags behind the "saving checkpoint" line by 20-60s,
            // so also kick off a delayed refresh when kohya announces generation.
            if (ev.type === 'log' && /generating sample images/i.test(ev.line || '')) {
                setTimeout(() => { const c = cfgRef.current; if (c) refreshSamples(c); }, 45000);
            }
        });
        return close;
    }, []);

    // debounce save
    const set = useCallback((path, value) => {
        setCfg(prev => {
            const next = setIn(prev, path, value);
            setDirty(true);
            clearTimeout(saveTimer.current);
            saveTimer.current = setTimeout(() => {
                api.putConfig(next).then(() => setDirty(false)).catch(() => {});
            }, 600);
            return next;
        });
    }, []);

    const replaceCfg = useCallback((nextCfg) => {
        setCfg(prev => {
            // keep current paths, since the imported preset may have stale ones
            const merged = { ...nextCfg, paths: { ...nextCfg.paths, ...(prev?.paths || {}) } };
            setDirty(true);
            api.putConfig(merged).then(() => setDirty(false)).catch(() => {});
            return merged;
        });
    }, []);

    // Auto-apply Anima+LoKr defaults on the first transition into the combo.
    // Fields the user has already touched (current value differs from the
    // kohya default we shipped) are preserved. Re-arming the guard on combo
    // change lets the user re-trigger by toggling out and back in.
    useEffect(() => {
        if (!cfg) return;
        const isCombo = cfg.model?.arch === 'anima' && cfg.network?.kind === 'lokr';
        if (!isCombo) { animaLokrApplied.current = false; return; }
        if (animaLokrApplied.current) return;
        animaLokrApplied.current = true;
        setCfg(prev => {
            let next = prev;
            let changed = 0;
            for (const [path, oldDef, newDef] of ANIMA_LOKR_DEFAULTS) {
                const cur = getIn(next, path);
                if (cur === oldDef || cur === undefined) {
                    if (cur !== newDef) {
                        next = setIn(next, path, newDef);
                        changed++;
                    }
                }
            }
            if (changed > 0) {
                setDirty(true);
                api.putConfig(next).then(() => setDirty(false)).catch(() => {});
                toast('ok', `Применены дефолты Anima + LoKr (${changed} поле/полей)`);
            }
            return next;
        });
    }, [cfg?.model?.arch, cfg?.network?.kind]);

    const applyPatch = useCallback((patch) => {
        setCfg(prev => {
            const next = deepMerge(prev, patch);
            setDirty(true);
            api.putConfig(next).then(() => setDirty(false)).catch(() => {});
            return next;
        });
        toast('ok', 'Пресет применён');
    }, []);

    const refreshPresets = async () => { try { setPresets(await api.presetsList()); } catch {} };
    const refreshOutputs = async (c) => { try { setOutputs(await api.fsOutputs((c || cfg)?.paths?.output_root)); } catch {} };
    const refreshSamples = async (c) => {
        const target = (c || cfg);
        if (!target) return;
        // Pass empty path so the backend merges output_root + samples_root.
        // kohya always writes to <output_dir>/sample, so output_root is required.
        try { setSamples(await api.fsSamples('')); } catch {}
    };
    const refreshModels = async (c) => {
        const target = (c || cfg);
        if (!target?.paths?.base_model_root) return;
        try { setModels(await api.fsModels(target.paths.base_model_root)); } catch {}
    };
    const rescanDataset = async () => {
        if (!cfg.paths.dataset_root) return;
        try {
            const found = await api.fsScanDataset(cfg.paths.dataset_root);
            setScanResult(found);
            // overwrite subsets but try to preserve user-edited fields by image_dir
            const prev = cfg.dataset.subsets || [];
            const merged = found.map(f => {
                const existing = prev.find(p => p.image_dir === f.image_dir);
                return existing
                    ? { ...existing, num_images: f.num_images }
                    : { image_dir: f.image_dir, num_repeats: f.num_repeats, caption_extension: '.txt',
                        shuffle_caption: true, keep_tokens: 1, num_images: f.num_images };
            });
            set('dataset.subsets', merged);
            toast('ok', `Найдено ${found.length} подпапок · ${found.reduce((s, x) => s + x.num_images, 0)} картинок`);
        } catch (e) { toast('err', 'Скан не удался: ' + e.message); }
    };

    // periodic sample/output refresh while running
    useEffect(() => {
        if (status.state !== 'running' && status.state !== 'starting') return;
        const t = setInterval(() => {
            if (cfg) { refreshOutputs(cfg); refreshSamples(cfg); }
        }, 10_000);
        return () => clearInterval(t);
    }, [status.state, cfg]);

    const onStart = async () => {
        try {
            const v = validate(cfg);
            if (v.errors.length > 0) {
                toast('err', `Не могу запустить: ${v.errors[0].message}`);
                return;
            }
            await api.putConfig(cfg);
            await api.trainStart();
            setLogs([]);
            setLrTrace([]);
            toast('ok', 'Обучение запущено');
        } catch (e) { toast('err', String(e.message)); }
    };
    const onStop = async () => { try { await api.trainStop(); toast('ok', 'Запрошена остановка'); } catch (e) { toast('err', String(e.message)); } };
    const onSaveConfig = async () => {
        try { await api.putConfig(cfg); setDirty(false); toast('ok', 'Конфиг сохранён'); }
        catch (e) { toast('err', String(e.message)); }
    };
    const onCleanup = async () => {
        const msg = 'Удалить локальные файлы?\n\n'
            + '• сэмплы (output_root/sample, samples_root)\n'
            + '• обученные LoRA (*.safetensors в output_root)\n'
            + '• папку .lora_trainer (конфиг + пресеты)\n'
            + '• кэш миниатюр\n\n'
            + 'Файлы на Google Drive не затрагиваются. Операция необратима.';
        if (!window.confirm(msg)) return;
        try {
            const r = await api.cleanup();
            const rm = r.removed || {};
            toast('ok', `Удалено: сэмплы ${rm.samples||0}, LoRA ${rm.loras||0}, кэш ${rm.thumb_cache||0}`);
            if (r.errors && r.errors.length) {
                toast('warn', `Ошибок: ${r.errors.length} (см. консоль)`);
                console.warn('[cleanup] errors:', r.errors);
            }
            await Promise.all([refreshOutputs(cfg), refreshSamples(cfg), refreshPresets()]);
        } catch (e) { toast('err', String(e.message)); }
    };

    const val = useMemo(() => {
        if (!cfg) return { errors: [], warnings: [], errMap: {}, warnMap: {} };
        const v = validate(cfg);
        const errMap = Object.fromEntries(v.errors.map(x => [x.path, x.message]));
        const warnMap = Object.fromEntries(v.warnings.map(x => [x.path, x.message]));
        return { ...v, errMap, warnMap };
    }, [cfg]);

    const totalSteps = useMemo(() => cfg ? computeTotalStepsLocal(cfg) : 0, [cfg]);

    if (!cfg) {
        return html`<div style="padding: 40px; color: var(--text-dim);">Загрузка конфига…</div>`;
    }

    const SectionEl = (() => {
        switch (section) {
            case 'project':  return html`<${SectionProject} cfg=${cfg} set=${set} val=${val} />`;
            case 'model':    return html`<${SectionModel} cfg=${cfg} set=${set} val=${val} models=${models} />`;
            case 'dataset':  return html`<${SectionDataset} cfg=${cfg} set=${set} val=${val} rescan=${rescanDataset} scanResult=${scanResult} />`;
            case 'network':  return html`<${SectionNetwork} cfg=${cfg} set=${set} val=${val} />`;
            case 'training': return html`<${SectionTraining} cfg=${cfg} set=${set} val=${val} />`;
            case 'samples':  return html`<${SectionSamples} cfg=${cfg} set=${set} val=${val} />`;
            case 'gallery':  return html`<${SectionGallery} cfg=${cfg} samples=${samples} refresh=${() => refreshSamples(cfg)} />`;
            case 'logs':     return html`<${SectionLogs} logs=${logs} onClear=${async () => { await api.clearLogs(); setLogs([]); }} />`;
            case 'files':    return html`<${SectionFiles} outputs=${outputs} refresh=${() => refreshOutputs(cfg)} cfg=${cfg} />`;
            case 'presets':  return html`<${SectionPresets} cfg=${cfg} applyPatch=${applyPatch} replaceCfg=${replaceCfg} presets=${presets} refresh=${refreshPresets} />`;
            default:         return null;
        }
    })();

    return html`
        <div class="app">
            <${TopBar} cfg=${cfg} status=${status} onStart=${onStart} onStop=${onStop} dirty=${dirty} onSaveConfig=${onSaveConfig} onCleanup=${onCleanup} />

            <aside class="sidebar">
                <div class="group-label">Настройки</div>
                ${SECTIONS.slice(0, 6).map(s => html`
                    <div class=${'nav-item' + (section === s.id ? ' active' : '')} onClick=${() => setSection(s.id)} key=${s.id}>
                        <span class="icon">${s.icon}</span>
                        <span>${s.label}</span>
                    </div>
                `)}
                <div class="group-label">Процесс</div>
                ${SECTIONS.slice(6).map(s => html`
                    <div class=${'nav-item' + (section === s.id ? ' active' : '')} onClick=${() => setSection(s.id)} key=${s.id}>
                        <span class="icon">${s.icon}</span>
                        <span>${s.label}</span>
                        ${s.id === 'logs' && logs.length > 0 && html`<span class="badge">${logs.length}</span>`}
                        ${s.id === 'files' && outputs.length > 0 && html`<span class="badge">${outputs.length}</span>`}
                        ${s.id === 'gallery' && samples.length > 0 && html`<span class="badge">${samples.length}</span>`}
                    </div>
                `)}
                <div style="flex:1;"></div>
                <div style="padding: 10px; font-size: 11px; color: var(--text-mute); border-top: 1px solid var(--line);">
                    state: <span class="mono">${status.state}</span><br/>
                    pid: <span class="mono">${status.pid ?? '—'}</span>
                </div>
            </aside>

            <main class="main">${SectionEl}</main>

            <aside class="statbar">
                <${StatPanel} cfg=${cfg} status=${status} totalSteps=${totalSteps} lrTrace=${lrTrace} />
            </aside>
        </div>

        <${ToastStack} />
    `;
};

render(h(App, null), document.getElementById('root'));
