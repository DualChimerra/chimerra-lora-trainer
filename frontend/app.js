// LoRA Trainer · Preact + htm, no build step.
import { h, render } from 'https://esm.sh/preact@10.22.0';
import { useState, useEffect, useMemo, useRef, useCallback } from 'https://esm.sh/preact@10.22.0/hooks';
import htm from 'https://esm.sh/htm@3.1.1';

import { api, openSocket } from './api.js';
import { TIPS, tipFor } from './tooltips.js';
import { validate, findWarn } from './validations.js';
import { BUILTIN_PRESETS, deepMerge } from './presets.js';

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
        // SDXL-only training noise/loss fields
        const tdel = [
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
                <${Field} label="network_dropout" tipKey="network.network_dropout">
                    <${Num} value=${n.network_dropout} step=${0.05} min=${0} max=${1} onInput=${v => set('network.network_dropout', v)} />
                </${Field}>
                <${Field} label="scale_weight_norms" tipKey="network.scale_weight_norms">
                    <${Num} value=${n.scale_weight_norms} step=${0.1} onInput=${v => set('network.scale_weight_norms', v)} />
                </${Field}>
            </div>
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
        </div>

        <div class="card">
            <div class="card-title">Продолжить с весов (опционально)</div>
            <${Field} label="network_weights" tipKey="network.network_weights">
                <${Text} value=${n.network_weights || ''} onInput=${v => set('network.network_weights', v || null)}
                    placeholder="(пустое — обучаем с нуля; или путь к .safetensors LoRA)" />
            </${Field}>
            <${Field} label="dim_from_weights" tipKey="network.dim_from_weights">
                <${Check} value=${!!n.dim_from_weights} onInput=${v => set('network.dim_from_weights', v)}
                    label="взять network_dim/alpha из весов" />
            </${Field}>
        </div>
    `;
};

// =============================================================================
// Section: OPTIMIZER / TRAINING
// =============================================================================
const SectionTraining = ({ cfg, set, val }) => {
    const opts = ['AdamW','AdamW8bit','Lion','Lion8bit','Prodigy','DAdaptation','DAdaptAdam','DAdaptLion',
                  'SGDNesterov','SGDNesterov8bit','AdaFactor','PagedAdamW8bit','PagedLion8bit'];
    const isAnima = cfg.model.arch === 'anima';
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
                    <${Select} value=${cfg.optimizer.lr_scheduler} onInput=${v => set('optimizer.lr_scheduler', v)}
                        options=${['constant','constant_with_warmup','linear','cosine','cosine_with_restarts','polynomial','adafactor','rex']} />
                </${Field}>
            </div>
            <div class="grid-3">
                <${Field} label="unet_lr" tipKey="optimizer.unet_lr">
                    <${Num} value=${cfg.optimizer.unet_lr ?? ''} step=${1e-5} onInput=${v => set('optimizer.unet_lr', v)} />
                </${Field}>
                <${Field} label="text_encoder_lr" tipKey="optimizer.text_encoder_lr">
                    <${Num} value=${cfg.optimizer.text_encoder_lr ?? ''} step=${1e-5} onInput=${v => set('optimizer.text_encoder_lr', v)} />
                </${Field}>
                <${Field} label="lr_warmup_steps" tipKey="optimizer.lr_warmup_steps">
                    <${Num} value=${cfg.optimizer.lr_warmup_steps} onInput=${v => set('optimizer.lr_warmup_steps', v)} />
                </${Field}>
                <${Field} label="lr_scheduler_num_cycles" tipKey="optimizer.lr_scheduler_num_cycles"
                    warn=${val.warnMap['optimizer.lr_scheduler_num_cycles']}>
                    <${Num} value=${cfg.optimizer.lr_scheduler_num_cycles} onInput=${v => set('optimizer.lr_scheduler_num_cycles', v)} />
                </${Field}>
                <${Field} label="lr_scheduler_power" tipKey="optimizer.lr_scheduler_power">
                    <${Num} value=${cfg.optimizer.lr_scheduler_power} step=${0.1} onInput=${v => set('optimizer.lr_scheduler_power', v)} />
                </${Field}>
                <${Field} label="max_grad_norm" tipKey="optimizer.max_grad_norm">
                    <${Num} value=${cfg.optimizer.max_grad_norm ?? 1.0} step=${0.1} min=${0}
                        onInput=${v => set('optimizer.max_grad_norm', v)} />
                </${Field}>
            </div>
            <${Field} label="optimizer_args" tipKey="optimizer.optimizer_args" columns="stack">
                <textarea placeholder=${"weight_decay=0.1\nbetas=0.9,0.999"}
                    value=${(cfg.optimizer.optimizer_args || []).join('\n')}
                    onInput=${e => set('optimizer.optimizer_args',
                        // split on real newlines or HTML-encoded newlines users may have pasted
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
                <${Check} value=${cfg.training.mem_eff_attn} onInput=${v => set('training.mem_eff_attn', v)} label="mem_eff_attn" />
                <${Check} value=${cfg.training.full_bf16} onInput=${v => set('training.full_bf16', v)} label="full_bf16" />
                <${Check} value=${cfg.training.full_fp16} onInput=${v => set('training.full_fp16', v)} label="full_fp16" />
                <${Check} value=${cfg.training.lowram} onInput=${v => set('training.lowram', v)} label="lowram" />
                <${Check} value=${cfg.training.highvram} onInput=${v => set('training.highvram', v)} label="highvram" />
                <${Check} value=${cfg.training.fused_backward_pass} onInput=${v => set('training.fused_backward_pass', v)} label="fused_backward_pass" />
            </div>
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
                        options=${['l2','l1','huber','smooth_l1']} />
                </${Field}>
                ${(cfg.training.loss_type === 'huber' || cfg.training.loss_type === 'smooth_l1') && html`
                    <${Field} label="huber_schedule" tipKey="training.huber_schedule">
                        <${Select} value=${cfg.training.huber_schedule || 'snr'} onInput=${v => set('training.huber_schedule', v)}
                            options=${['snr','exponential','constant']} />
                    </${Field}>
                    <${Field} label="huber_c" tipKey="training.huber_c">
                        <${Num} value=${cfg.training.huber_c ?? 0.1} step=${0.01} onInput=${v => set('training.huber_c', v)} />
                    </${Field}>
                `}
                <${Field} label="min_timestep" tipKey="training.min_timestep">
                    <${Num} value=${cfg.training.min_timestep ?? 0} min=${0} max=${1000}
                        onInput=${v => set('training.min_timestep', v)} />
                </${Field}>
                <${Field} label="max_timestep" tipKey="training.max_timestep">
                    <${Num} value=${cfg.training.max_timestep ?? 1000} min=${0} max=${1000}
                        onInput=${v => set('training.max_timestep', v)} />
                </${Field}>
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

        <div class="card">
            <div class="card-title">${isAnima ? 'FlowMatch (Anima)' : 'Шум и Loss'}</div>
            ${isAnima
                ? html`
                    <div class="grid-3">
                        <${Field} label="weighting_scheme" tipKey="training.weighting_scheme">
                            <${Select} value=${cfg.training.weighting_scheme} onInput=${v => set('training.weighting_scheme', v)}
                                options=${['sigma_sqrt','logit_normal','mode','cosmap','none']} />
                        </${Field}>
                        <${Field} label="timestep_sampling" tipKey="training.timestep_sampling">
                            <${Select} value=${cfg.training.timestep_sampling} onInput=${v => set('training.timestep_sampling', v)}
                                options=${['uniform','sigmoid','shift','flux_shift']} />
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
        try {
            const text = await file.text();
            const parsed = JSON.parse(text);
            // Either a TrainConfig or a Preset (with .config inside)
            const next = parsed.config && parsed.created_at ? parsed.config : parsed;
            replaceCfg(next);
            toast('ok', 'Конфиг импортирован');
        } catch (e) { toast('err', 'Не удалось разобрать JSON: ' + e.message); }
    };

    return html`
        <h2>Шаблоны (пресеты)</h2>
        <div class="subtitle">Сохраняйте и переключайте конфигурации, экспортируйте в JSON.</div>

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
                <${Btn} variant="ghost" onClick=${() => fileRef.current?.click()}>↑ Импорт JSON</${Btn}>
                <input ref=${fileRef} type="file" accept="application/json" style="display:none;"
                    onChange=${e => e.target.files[0] && onImportFile(e.target.files[0])} />
            </div>
        </div>
    `;
};

// =============================================================================
// Right stats panel
// =============================================================================
const StatPanel = ({ cfg, status, totalSteps, etaSec }) => {
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
    `;
};

// =============================================================================
// Topbar
// =============================================================================
const TopBar = ({ cfg, status, onStart, onStop, dirty, onSaveConfig }) => html`
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

const App = () => {
    const [cfg, setCfg] = useState(null);
    const [section, setSection] = useState('project');
    const [status, setStatus] = useState({ state: 'idle', step: 0, total_steps: 0, epoch: 0 });
    const [logs, setLogs] = useState([]);
    const [presets, setPresets] = useState([]);
    const [outputs, setOutputs] = useState([]);
    const [samples, setSamples] = useState([]);
    const [models, setModels] = useState([]);
    const [scanResult, setScanResult] = useState([]);
    const [dirty, setDirty] = useState(false);
    const saveTimer = useRef(null);

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
            if (ev.type === 'status') setStatus(ev.status);
            if (ev.type === 'log') setLogs(prev => {
                const next = [...prev, ev];
                return next.length > 5000 ? next.slice(-5000) : next;
            });
            // when training emits a save line, refresh outputs/samples
            if (ev.type === 'log' && /saving (checkpoint|state)|saved model|saving images/i.test(ev.line || '')) {
                if (cfg) { refreshOutputs(cfg); refreshSamples(cfg); }
            }
            // sample generation lags behind the "saving checkpoint" line by 20-60s,
            // so also kick off a delayed refresh when kohya announces generation.
            if (ev.type === 'log' && /generating sample images/i.test(ev.line || '')) {
                if (cfg) setTimeout(() => { refreshSamples(cfg); }, 45000);
            }
        });
        return close;
    }, [cfg]);

    // While training is running, poll samples/outputs periodically so the
    // gallery fills up even if we miss the log triggers above.
    useEffect(() => {
        if (status.state !== 'running' && status.state !== 'starting') return;
        if (!cfg) return;
        const id = setInterval(() => {
            refreshSamples(cfg);
            refreshOutputs(cfg);
        }, 20000);
        return () => clearInterval(id);
    }, [status.state, cfg]);

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
            toast('ok', 'Обучение запущено');
        } catch (e) { toast('err', String(e.message)); }
    };
    const onStop = async () => { try { await api.trainStop(); toast('ok', 'Запрошена остановка'); } catch (e) { toast('err', String(e.message)); } };
    const onSaveConfig = async () => {
        try { await api.putConfig(cfg); setDirty(false); toast('ok', 'Конфиг сохранён'); }
        catch (e) { toast('err', String(e.message)); }
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
            <${TopBar} cfg=${cfg} status=${status} onStart=${onStart} onStop=${onStop} dirty=${dirty} onSaveConfig=${onSaveConfig} />

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
                <${StatPanel} cfg=${cfg} status=${status} totalSteps=${totalSteps} />
            </aside>
        </div>

        <${ToastStack} />
    `;
};

render(h(App, null), document.getElementById('root'));
