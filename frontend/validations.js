// Real-time UI hints. Returns { warnings: [{path, message}], errors: [...] }
// All messages in Russian — they're shown to the operator.

export function validate(cfg) {
    const W = [];   // soft warnings (yellow)
    const E = [];   // hard errors (red)

    // paths
    if (!cfg.paths.dataset_root)    E.push({ path: 'paths.dataset_root', message: 'Не указана папка датасета' });
    if (!cfg.paths.output_root)     E.push({ path: 'paths.output_root', message: 'Не указана папка для выходных файлов' });
    if (!cfg.paths.sd_scripts_dir)  E.push({ path: 'paths.sd_scripts_dir', message: 'Не указан путь к sd-scripts' });

    // model
    if (!cfg.model.pretrained_model_name_or_path) {
        E.push({ path: 'model.pretrained_model_name_or_path', message: 'Не выбрана базовая модель' });
    }
    if (cfg.model.arch === 'anima') {
        if (!cfg.model.anima_qwen3)              E.push({ path: 'model.anima_qwen3',              message: 'Anima: укажите путь к Qwen3 text encoder' });
        if (!cfg.model.anima_t5_tokenizer_path)  E.push({ path: 'model.anima_t5_tokenizer_path',  message: 'Anima: укажите путь к T5 tokenizer' });
        if (!cfg.model.anima_vae)                E.push({ path: 'model.anima_vae',                message: 'Anima: укажите путь к Qwen-Image VAE' });
    }

    // dataset
    if (!cfg.dataset.subsets || cfg.dataset.subsets.length === 0) {
        E.push({ path: 'dataset.subsets', message: 'Нажмите «↻ Пересканировать папку» или добавьте папку вручную кнопкой «+ Добавить»' });
    } else {
        const total = cfg.dataset.subsets.reduce((s, x) => s + (x.num_images || 0) * (x.num_repeats || 0), 0);
        if (total === 0) {
            E.push({ path: 'dataset.subsets', message: 'В папках не найдено картинок (.png / .jpg / .webp)' });
        }
        cfg.dataset.subsets.forEach((s, i) => {
            if (s.num_repeats < 1)
                W.push({ path: `dataset.subsets.${i}.num_repeats`, message: 'Повторы должны быть ≥ 1' });
            if (s.num_repeats > 30)
                W.push({ path: `dataset.subsets.${i}.num_repeats`, message: `Repeats=${s.num_repeats} — много, возможен оверфит` });
        });
    }
    if (cfg.dataset.resolution % 64 !== 0) {
        W.push({ path: 'dataset.resolution', message: 'Разрешение лучше кратное 64 для bucket-режима' });
    }
    if (cfg.model.arch === 'sdxl' && cfg.dataset.resolution < 768) {
        W.push({ path: 'dataset.resolution', message: 'SDXL обучается заметно хуже на разрешении ниже 1024' });
    }

    // cache_text_encoder_outputs incompatibilities (kohya assertion at startup)
    if (cfg.dataset.cache_text_encoder_outputs) {
        const anyShuffle = (cfg.dataset.subsets || []).some(s => s.shuffle_caption);
        const trainsTE = !cfg.network.network_train_unet_only
            && (cfg.optimizer.text_encoder_lr == null || cfg.optimizer.text_encoder_lr > 0);
        if (anyShuffle) {
            E.push({
                path: 'dataset.cache_text_encoder_outputs',
                message: 'cache_text_encoder_outputs несовместим с shuffle_caption — выключите одно из двух',
            });
        }
        if (trainsTE) {
            E.push({
                path: 'dataset.cache_text_encoder_outputs',
                message: 'cache_text_encoder_outputs нельзя при обучении text encoder (text_encoder_lr / не unet-only)',
            });
        }
    }

    // network — главное правило: alpha ≤ dim для обычной LoRA
    const n = cfg.network;
    if (n.kind === 'lora' && n.network_alpha > n.network_dim) {
        W.push({
            path: 'network.network_alpha',
            message: `Alpha (${n.network_alpha}) больше Dim (${n.network_dim}) — обычно alpha ≤ dim для LoRA`
        });
    }
    if (n.kind === 'lokr' && n.factor === 0) {
        W.push({ path: 'network.factor', message: 'Factor=0 не имеет смысла; -1 = автоматически' });
    }
    if (n.kind !== 'lora' && (n.conv_dim || 0) > n.network_dim * 2) {
        W.push({ path: 'network.conv_dim', message: 'conv_dim сильно больше network_dim — возможно избыточно' });
    }
    if (n.network_train_unet_only && n.network_train_text_encoder_only) {
        E.push({ path: 'network.network_train_text_encoder_only', message: 'Нельзя одновременно train_unet_only и train_text_encoder_only' });
    }

    // optimizer
    const lr = cfg.optimizer.learning_rate;
    const opt = cfg.optimizer.optimizer_type;
    if ((opt === 'Prodigy' || opt === 'DAdaptation' || opt === 'DAdaptAdam' || opt === 'DAdaptLion')
        && lr !== 1.0 && lr !== 1) {
        W.push({
            path: 'optimizer.learning_rate',
            message: `${opt} сам подбирает LR — обычно ставят learning_rate=1.0, а unet_lr/text_encoder_lr оставляют пустыми`,
        });
    }
    if (opt === 'AdamW8bit' && (lr > 5e-4 || lr < 1e-6)) {
        W.push({ path: 'optimizer.learning_rate', message: `LR=${lr} необычный для AdamW8bit (типично 5e-5…3e-4)` });
    }
    if (opt === 'Lion' && lr > 1e-4) {
        W.push({ path: 'optimizer.learning_rate', message: 'Lion: LR обычно в 3–10× меньше, чем у AdamW' });
    }
    if (cfg.optimizer.lr_scheduler === 'cosine_with_restarts' && cfg.optimizer.lr_scheduler_num_cycles < 1) {
        W.push({ path: 'optimizer.lr_scheduler_num_cycles', message: 'cosine_with_restarts требует ≥ 1 цикл' });
    }

    // training
    if (cfg.training.train_batch_size < 1)
        E.push({ path: 'training.train_batch_size', message: 'batch_size ≥ 1' });
    if (cfg.training.max_train_epochs < 1 && cfg.training.max_train_steps < 1)
        E.push({ path: 'training.max_train_epochs', message: 'Укажите либо max_train_epochs, либо max_train_steps' });
    if (cfg.training.mixed_precision === 'no')
        W.push({ path: 'training.mixed_precision', message: 'Без mixed_precision обучение будет медленнее и потребует больше VRAM' });

    // samples
    if (cfg.samples.enable && (!cfg.samples.prompts || cfg.samples.prompts.length === 0))
        W.push({ path: 'samples.prompts', message: 'Сэмплы включены, но список промптов пуст' });

    return { warnings: W, errors: E };
}

export function findWarn(byPath, path) {
    return byPath.find(x => x.path === path)?.message;
}
