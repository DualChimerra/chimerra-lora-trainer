# LoRA Trainer · Colab

UI-оболочка над [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) для тренировки LoRA / LyCORIS / LoKr под:

- **NoobAI** и **Illustrious** — через `sdxl_train_network.py`
- **Anima** (DiT · Qwen3-0.6B TE · Qwen-Image VAE) — через `anima_train_network.py`

Цель — productivity-first интерфейс в стиле Linear/Attio/Notion поверх неизменённого тренировочного бэкенда. Сама логика обучения **не меняется**: мы только собираем те же CLI-аргументы, которые принимает upstream sd-scripts, и запускаем их через `accelerate launch`.

## Архитектура

```
                                ┌─────────────────────────────┐
                                │  Browser (через cloudflared) │
                                │   Preact + htm, no build     │
                                └──────────────┬──────────────┘
                                               │ HTTP + WebSocket
                                ┌──────────────▼──────────────┐
                                │  FastAPI (backend/app.py)    │
                                │  • config persistence        │
                                │  • file-system browser       │
                                │  • train-process manager     │
                                └──────────────┬──────────────┘
                                               │ subprocess (accelerate launch)
                                ┌──────────────▼──────────────┐
                                │  kohya-ss/sd-scripts (UNTOUCHED) │
                                │  sdxl_train_network.py / anima_train_network.py │
                                └─────────────────────────────┘
```

## Быстрый старт

1. Откройте `colab/LoRA_Trainer.ipynb` в Google Colab (GPU runtime).
2. Запустите 4 ячейки по порядку — установка ставит kohya-ss/sd-scripts, lycoris-lora и cloudflared.
3. В выводе последней ячейки появится `https://<random>.trycloudflare.com` — это URL UI.
4. Откройте ссылку в браузере, заполните остальное в UI и нажмите **▶ Старт**.

## Что есть в UI

- **Project** — пути и имя выходного файла.
- **Model** — выбор архитектуры (SDXL/Anima) и весов. Для Anima поднимаются дополнительные поля `qwen3`, `t5_tokenizer_path`, `vae`.
- **Dataset** — автоскан подпапок вида `10_concept`, репитсы, разрешение, bucket, кеши.
- **Network** — LoRA / LoCon / LoHA / LoKr / DyLoRA / IA³ с подсветкой alpha > dim для классической LoRA.
- **Training** — оптимизатор, LR, scheduler, длительность, mixed precision, и (для Anima) FlowMatch-параметры `weighting_scheme`, `timestep_sampling`, `discrete_flow_shift` и т.д.
- **Samples** — список промптов + расписание `every_n_epochs` / `every_n_steps`.
- **Gallery** — матрица «промпт × эпоха» из сгенерированных сэмплов, клик по картинке — full-screen viewer.
- **Logs** — live-stream из stdout тренера через WebSocket.
- **Files** — список выходных `.safetensors` с размером, временем и инферренной эпохой.
- **Presets** — встроенные стартеры (NoobAI/Illustrious/LoKr/Anima) и собственные пресеты, импорт/экспорт в JSON.

## Что под капотом не меняется

Тренер запускается так же, как если бы вы вызвали его из терминала:

```bash
accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --pretrained_model_name_or_path=... --dataset_config=dataset.toml \
  --network_module=networks.lora --network_dim=32 --network_alpha=16 \
  --optimizer_type=AdamW8bit --learning_rate=1e-4 ... 
```

`backend/config_builder.py` собирает эту команду **один-в-один** под аргументы upstream. Никаких патчей в файлах sd-scripts мы не делаем.

## Безопасность файловой системы

UI может листать и читать только файлы внутри 4 «корневых» путей, заданных в Colab-ячейке (`DATASET_ROOT`, `BASE_MODEL_ROOT`, `OUTPUT_ROOT`, `SAMPLES_ROOT`). Любой запрос к файлу вне этих папок отклоняется FastAPI-эндпоинтом.

## Состояние и пресеты

- Текущий конфиг автосохраняется в `{OUTPUT_ROOT}/.lora_trainer/last_config.json` при каждом изменении (debounce 600 мс).
- Пресеты лежат в `{OUTPUT_ROOT}/.lora_trainer/presets/<name>.json`.
- Если Drive не доступен, всё падает в `/content/.lora_trainer/`.

## Локальная разработка (без Colab)

```bash
pip install -r requirements.txt
export LT_DATASET_ROOT=/tmp/dataset
export LT_BASE_MODEL_ROOT=/tmp/models
export LT_OUTPUT_ROOT=/tmp/output
export LT_SD_SCRIPTS_DIR=/tmp/sd-scripts   # клонируйте upstream сюда
uvicorn backend.app:app --reload --port 7860
```

Откройте `http://127.0.0.1:7860`.

## Поддержка Anima

`anima_train_network.py` принимает отдельные пути для:

- `--pretrained_model_name_or_path` — DiT-чекпоинт Anima
- `--qwen3` — Qwen3-0.6B text encoder
- `--t5_tokenizer_path` — T5 tokenizer
- `--vae` — Qwen-Image VAE

В UI переключение `arch = anima` автоматически раскрывает эти поля и убирает SDXL-специфичные (clip_skip, v_parameterization), а во вкладке Training показывает FlowMatch-параметры вместо стандартных noise/SNR.

## Лицензия

UI: MIT. Тренировочный бэкенд (sd-scripts) — Apache 2.0, авторский © kohya-ss.
