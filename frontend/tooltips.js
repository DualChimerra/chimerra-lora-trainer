// Field tooltips. Keys = "section.field" or just "field" inside a known section.
// Values: human-readable RU explanation shown on hover.

export const TIPS = {
    // paths
    'paths.dataset_root':   'Корневая папка с подпапками вида "10_concept" (число — repeats).',
    'paths.base_model_root':'Папка, где лежат базовые модели (.safetensors / .ckpt).',
    'paths.output_root':    'Куда сохранять чекпоинты LoRA и логи.',
    'paths.samples_root':   'Куда складывать сэмплы; обычно подпапка sample/ внутри output_root.',
    'paths.sd_scripts_dir': 'Локальный клон kohya-ss/sd-scripts (обычно /content/sd-scripts).',

    // project
    'project.output_name':  'Имя выходного файла (.safetensors).',
    'project.save_model_as':'Формат файла модели. По умолчанию safetensors — безопаснее и быстрее.',
    'project.log_with':     'TensorBoard — лучший вариант для Colab; wandb требует ключа.',

    // model
    'model.arch':           'Тип архитектуры. SDXL подходит для NoobAI и Illustrious. Anima — отдельная DiT-архитектура (Qwen3 TE + Qwen-Image VAE).',
    'model.preset':         'Профиль базовой модели. Меняет дефолты под NoobAI/Illustrious/Anima.',
    'model.pretrained_model_name_or_path':
        'Путь к .safetensors базовой модели (для Anima — путь к DiT-чекпоинту).',
    'model.vae':            'Внешний VAE. Оставьте пустым, если VAE уже встроен в модель.',
    'model.v_parameterization':'v-prediction. Включайте только если базовая модель v-pred.',
    'model.clip_skip':      'CLIP skip. Для SDXL/Illustrious обычно 2.',
    'model.anima_qwen3':    'Anima: путь к Qwen3-0.6B text encoder.',
    'model.anima_t5_tokenizer_path':'Anima: путь к токенизатору T5.',
    'model.anima_vae':      'Anima: путь к Qwen-Image VAE.',

    // dataset
    'dataset.resolution':   'Базовое разрешение бакета. SDXL — 1024, Anima — 1024.',
    'dataset.enable_bucket':'Кадры разного aspect ratio обучаются в "бакетах" своих размеров.',
    'dataset.min_bucket_reso':'Нижний предел стороны bucket.',
    'dataset.max_bucket_reso':'Верхний предел стороны bucket.',
    'dataset.bucket_reso_steps':'Шаг bucket-сетки в пикселях (обычно 64).',
    'dataset.bucket_no_upscale':'Не апскейлить картинки меньше базового разрешения.',
    'dataset.cache_latents':'Кешировать латенты VAE — резко ускоряет эпохи.',
    'dataset.cache_latents_to_disk':'Сохранять кеш латентов на диск (между запусками).',
    'dataset.cache_text_encoder_outputs':'Кешировать выходы text encoder (для SDXL/Anima — почти всегда нужно).',
    'dataset.cache_text_encoder_outputs_to_disk':'Сохранять кеш TE на диск.',
    'dataset.color_aug':    'Случайные сдвиги цвета. Может слегка помочь стилю.',
    'dataset.flip_aug':     'Горизонтальное отзеркаливание. Не включайте для асимметричных концептов.',
    'dataset.random_crop':  'Случайные кропы вместо центральных.',
    'dataset.max_token_length':'Максимум токенов в подписи (75 / 150 / 225). Для длинных тегов — 225.',
    'dataset.max_data_loader_n_workers':'Воркеры PyTorch DataLoader. По умолчанию kohya = 0 (single-thread!) — это сильно замедляет эпохи. На Colab оптимально 4–8.',
    'dataset.persistent_data_loader_workers':'Не пересоздавать воркеры каждую эпоху. Включайте при num_workers > 0 — экономит секунды между эпохами.',
    'dataset.caption_dropout_rate':'Вероятность полностью обнулить подпись для картинки (0.0–0.1 типично).',
    'dataset.caption_tag_dropout_rate':'Вероятность дропа отдельного тега (0.0–0.2). Регуляризация для тегированных датасетов.',
    'dataset.caption_dropout_every_n_epochs':'Дропать подписи каждые N эпох (0 = выключено).',
    'dataset.token_warmup_min':'Минимальное число токенов на старте при token-warmup.',
    'dataset.token_warmup_step':'За сколько шагов разогнаться до полной длины подписи (0 = выключено).',
    'dataset.weighted_captions':'Поддержка веса (tag:1.2) в подписях.',

    // dataset subset
    'subset.num_repeats':   'Сколько раз набор картинок повторится за эпоху.',
    'subset.caption_extension':'Расширение файлов подписей (.txt или .caption).',
    'subset.class_tokens':  'Дефолтная подпись для картинок без .txt.',
    'subset.shuffle_caption':'Перемешивать порядок тегов между картинками.',
    'subset.keep_tokens':   'Сколько первых токенов всегда оставлять впереди при shuffle.',

    // network
    'network.kind':         'Тип сети: LoRA (низкоранговая дельта), LyCORIS — LoCon/LoHA/LoKr — более выразительные семейства.',
    'network.network_dim':  'Ранг сети. Чем больше — тем больше параметров и места под "стиль".',
    'network.network_alpha':'Масштаб обновлений. Правило: alpha ≤ dim. Часто alpha = dim/2.',
    'network.conv_dim':     'Ранг для свёрток (LoCon/LoHA/LoKr).',
    'network.conv_alpha':   'Alpha для свёрток.',
    'network.preset':       'Группа слоёв, в которые встраивается сеть. full = все.',
    'network.factor':       'LoKr factor. -1 = автоматически.',
    'network.decompose_both':'LoKr: разложение обоих факторов. Больше параметров, лучше выразительность.',
    'network.use_tucker':   'Tucker-разложение свёрток. Меньше параметров.',
    'network.use_scalar':   'Обучаемый скаляр-множитель на слой.',
    'network.rank_dropout': 'Дропаут по рангу: 0.0–0.3 типично.',
    'network.module_dropout':'Дропаут по модулям.',
    'network.network_dropout':'Общий dropout по сети.',
    'network.network_train_unet_only':'Тренировать только UNet (часто для SDXL).',
    'network.network_train_text_encoder_only':'Тренировать только Text Encoder.',
    'network.scale_weight_norms':'Жёсткий клиппинг норм весов сети (0 = выключено).',

    // optimizer
    'optimizer.optimizer_type':'Оптимизатор. AdamW8bit — экономный по VRAM. Prodigy/DAdapt — авто-LR (ставьте LR=1). pytorch_optimizer.CAME — адаптивный, экономный (нужен пакет pytorch-optimizer).',
    'optimizer.optimizer_args':'Доп. аргументы оптимизатора в форме "key=value".',
    'optimizer.learning_rate':'Базовый LR. Для Prodigy/DAdaptation = 1.0.',
    'optimizer.unet_lr':    'Отдельный LR для UNet (по умолчанию = learning_rate).',
    'optimizer.text_encoder_lr':'Отдельный LR для Text Encoder. Обычно меньше unet_lr.',
    'optimizer.lr_scheduler':'cosine — самый частый выбор для LoRA. warmup_stable_decay (WSD) — разогрев → постоянный LR через всю середину → короткий резкий спад в конце (требует lr_decay_steps). piecewise_constant — ручные ступени: тяни точки на графике ниже.',
    'optimizer.lr_warmup_steps':'Сколько шагов с линейным разогревом LR от 0.',
    'optimizer.lr_decay_steps':'Только для WSD: длина финальной фазы спада. Доля 0–1 (0.2 = последние 20% шагов) или абсолютное число шагов (≥1). Меньше доля = резче обрыв.',
    'optimizer.lr_scheduler_num_cycles':'Циклы для cosine_with_restarts.',
    'optimizer.lr_scheduler_args':'Доп. аргументы шедулера (например, num_cycles=2 для cosine_with_restarts).',
    'optimizer.max_grad_norm':'Клиппинг градиентов. 1.0 — стандартно. 0 = выключить клиппинг.',

    // network — продолжение с весов
    'network.network_weights':'Путь к существующей LoRA (.safetensors) — обучение продолжится с этих весов.',
    'network.dim_from_weights':'Взять network_dim и alpha из файла весов (полезно при resume).',

    // training
    'training.train_batch_size':'Размер батча. На Colab T4 безопасно 1–2, A100 — 4+.',
    'training.gradient_accumulation_steps':'Эмуляция большого батча. Эффективный батч = batch × accum.',
    'training.max_train_epochs':'Сколько эпох обучать.',
    'training.max_train_steps':'Альтернатива эпохам: жёсткий потолок шагов. 0 = использовать эпохи.',
    'training.save_every_n_epochs':'Сохранять промежуточный чекпоинт каждые N эпох.',
    'training.save_last_n_epochs':'Хранить только последние N чекпоинтов (0 = все).',
    'training.seed':        'Сид для воспроизводимости.',
    'training.mixed_precision':'bf16 — лучший выбор для A100/L4. fp16 — для T4.',
    'training.save_precision':'Точность сохранённого чекпоинта.',
    'training.gradient_checkpointing':'Экономит VRAM ценой ~20% скорости. Обычно ВКЛ.',
    'training.xformers':    'Memory-efficient attention. Обычно используется sdpa вместо xformers.',
    'training.sdpa':        'PyTorch native scaled dot-product attention.',
    'training.min_snr_gamma':'Min-SNR weighting. 5.0 — рабочее значение для SD.',
    'training.noise_offset':'Сдвиг шума. 0.0357 (≈1/28) — типично.',
    'training.adaptive_noise_scale':'Адаптивный множитель к noise_offset.',
    'training.multires_noise_iterations':'Multi-resolution noise. 6–10 типично.',
    'training.multires_noise_discount':'Дисконт для multi-res шума (0.1–0.3).',
    'training.ip_noise_gamma':'Input perturbation γ. Сглаживает обучение.',
    'training.debiased_estimation_loss':'Debiased estimation loss (помогает с outliers).',
    'training.zero_terminal_snr':'Включить zero terminal SNR. Для v-pred моделей.',
    'training.vae_batch_size':'Размер батча для VAE при кешировании латентов. 0 = дефолт kohya. Увеличьте для ускорения cache-фазы.',
    'training.prior_loss_weight':'Вес prior-loss для регуляризационных картинок. 1.0 — стандарт.',
    'training.loss_type':'Тип loss-функции. l2 — стандарт; huber/smooth_l1 устойчивее к outliers.',
    'training.huber_schedule':'Как расписан huber c по timestep (snr — рекомендовано).',
    'training.huber_c':'Параметр c для Huber-loss (0.1 типично).',
    'training.min_timestep':'Минимальный timestep при сэмплинге (0 = весь диапазон).',
    'training.max_timestep':'Максимальный timestep (1000 = весь диапазон). Уменьшение фокусирует на низком шуме.',
    'training.weighting_scheme':'Anima: схема взвешивания timestep-ов (logit_normal управляется logit_mean/logit_std).',
    'training.timestep_sampling':'Anima: метод выборки t (flux-style). logit_normal — логит-нормальная выборка (типичная пара к weighting_scheme=logit_normal).',
    'training.discrete_flow_shift':'Anima: дискретный shift для FlowMatch (3.0 типично).',

    // samples
    'samples.enable':       'Генерировать предпросмотровые картинки прямо во время обучения.',
    'samples.every_n_epochs':'Каждые N эпох делать сэмпл-сет. 0 — отключено.',
    'samples.every_n_steps':'Каждые N шагов делать сэмпл (если >0, эпохи игнорируются).',
    'samples.sampler':      'Sampler для сэмплов. euler_a — быстрый и нейтральный.',
    'samples.prompt.text':  'Позитивный промпт. Можно использовать LoRA-теги.',
    'samples.prompt.negative':'Негативный промпт.',
    'samples.prompt.steps': 'Шагов сэмплера.',
    'samples.prompt.scale': 'CFG scale.',
    'samples.prompt.seed':  'Сид сэмпла. Одинаковый сид → детерминированное сравнение по эпохам.',
};

export function tipFor(key) {
    return TIPS[key] || null;
}
