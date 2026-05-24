"""FastAPI server: REST + WebSocket bridge between the Preact UI and the
sd-scripts trainer subprocess running in Colab.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .schemas import TrainConfig, Preset
from .state import StateStore
from .filesystem import FileSystem
from .trainer import Trainer
from .config_builder import compute_total_steps, build_command


ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"


def _read_env_paths() -> dict:
    """Initial paths come from the Colab cell via env vars."""
    return {
        "dataset_root": os.environ.get("LT_DATASET_ROOT", ""),
        "base_model_root": os.environ.get("LT_BASE_MODEL_ROOT", ""),
        "output_root": os.environ.get("LT_OUTPUT_ROOT", ""),
        "samples_root": os.environ.get("LT_SAMPLES_ROOT", ""),
        "sd_scripts_dir": os.environ.get("LT_SD_SCRIPTS_DIR", "/content/sd-scripts"),
    }


def make_app() -> FastAPI:
    app = FastAPI(title="LoRA Trainer (Colab)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_paths = _read_env_paths()

    # state dir lives under output_root if set, else /content
    state = StateStore(
        primary_dir=os.path.join(init_paths.get("output_root", "") or "", ".lora_trainer"),
        fallback_dir="/content/.lora_trainer",
    )

    cfg = state.load_config()
    if cfg is None:
        cfg = TrainConfig()
        cfg.paths.dataset_root = init_paths["dataset_root"]
        cfg.paths.base_model_root = init_paths["base_model_root"]
        cfg.paths.output_root = init_paths["output_root"]
        cfg.paths.samples_root = init_paths["samples_root"]
        cfg.paths.sd_scripts_dir = init_paths["sd_scripts_dir"]
        if not cfg.samples.prompts:
            from .schemas import SamplePrompt
            cfg.samples.prompts = [
                SamplePrompt(text="1girl, masterpiece, best quality"),
            ]
    else:
        # always overlay paths with the live Colab session
        for k, v in init_paths.items():
            if v:
                setattr(cfg.paths, k, v)

    fs = FileSystem(roots=[
        cfg.paths.dataset_root,
        cfg.paths.base_model_root,
        cfg.paths.output_root,
        cfg.paths.samples_root,
    ])
    trainer = Trainer()

    # in-memory "current config" — single user
    current: dict = {"cfg": cfg}

    def get_cfg() -> TrainConfig:
        return current["cfg"]

    def set_cfg(new_cfg: TrainConfig) -> None:
        current["cfg"] = new_cfg
        fs.update_roots([
            new_cfg.paths.dataset_root,
            new_cfg.paths.base_model_root,
            new_cfg.paths.output_root,
            new_cfg.paths.samples_root,
        ])
        try:
            state.save_config(new_cfg)
        except Exception as e:
            print(f"[state] save failed: {e}")

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------
    @app.get("/api/config")
    def api_get_config():
        return get_cfg().model_dump()

    @app.put("/api/config")
    def api_put_config(cfg_in: TrainConfig):
        set_cfg(cfg_in)
        return {"ok": True, "total_steps": compute_total_steps(cfg_in)}

    @app.get("/api/paths")
    def api_paths():
        return get_cfg().paths.model_dump()

    @app.get("/api/fs/list")
    def api_fs_list(path: str):
        return [asdict(e) for e in fs.list(path)]

    @app.get("/api/fs/scan_dataset")
    def api_fs_scan_dataset(path: str):
        return fs.scan_dataset(path)

    @app.get("/api/fs/models")
    def api_fs_models(path: str):
        return [asdict(e) for e in fs.list_models(path)]

    @app.get("/api/fs/outputs")
    def api_fs_outputs(path: str = ""):
        path = path or get_cfg().paths.output_root
        return [asdict(e) for e in fs.list_outputs(path)]

    @app.get("/api/fs/samples")
    def api_fs_samples(path: str = ""):
        path = path or get_cfg().paths.samples_root or get_cfg().paths.output_root
        return fs.list_samples(path)

    @app.get("/api/fs/file")
    def api_fs_file(path: str):
        """Stream a file (image/model) under one of the allow-listed roots."""
        from .filesystem import _safe_resolve
        resolved = _safe_resolve(path, fs.roots)
        if resolved is None or not resolved.is_file():
            raise HTTPException(404, "not found or not allowed")
        return FileResponse(str(resolved))

    @app.get("/api/calc/total_steps")
    def api_calc_total_steps():
        return {"total_steps": compute_total_steps(get_cfg())}

    @app.post("/api/calc/preview_command")
    def api_preview_command(cfg_in: TrainConfig):
        workdir = Path("/tmp/lora_trainer_preview")
        argv, _env = build_command(cfg_in, workdir)
        return {"argv": argv}

    # presets
    @app.get("/api/presets")
    def api_presets_list():
        return [p.model_dump() for p in state.list_presets()]

    class SavePresetIn(BaseModel):
        name: str
        description: str = ""
        config: TrainConfig

    @app.post("/api/presets")
    def api_presets_save(p: SavePresetIn):
        saved = state.save_preset(p.name, p.description, p.config)
        return saved.model_dump()

    @app.delete("/api/presets/{name}")
    def api_presets_delete(name: str):
        return {"deleted": state.delete_preset(name)}

    # trainer
    @app.post("/api/train/start")
    async def api_train_start():
        cfg = get_cfg()
        if not cfg.paths.sd_scripts_dir:
            raise HTTPException(400, "sd_scripts_dir is not set")
        if not cfg.paths.output_root:
            raise HTTPException(400, "output_root is not set")
        workdir = Path(cfg.paths.output_root) / ".lora_trainer" / "runs" / time.strftime("%Y%m%d-%H%M%S")
        try:
            await trainer.start(cfg, workdir)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "status": trainer.status.model_dump()}

    @app.post("/api/train/stop")
    async def api_train_stop():
        await trainer.stop()
        return {"ok": True, "status": trainer.status.model_dump()}

    @app.get("/api/train/status")
    def api_train_status():
        return trainer.snapshot()

    @app.post("/api/train/clear_logs")
    def api_clear_logs():
        trainer.clear_logs()
        return {"ok": True}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()

        async def push(event: dict):
            try:
                await ws.send_json(event)
            except Exception:
                pass

        # send snapshot first
        snap = trainer.snapshot()
        await ws.send_json({"type": "snapshot", **snap})

        trainer.subscribe(push)
        try:
            while True:
                # keep the socket alive; we don't expect inbound messages,
                # but accept pings/anything just in case
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            trainer.unsubscribe(push)

    # ------------------------------------------------------------------
    # Static frontend
    # ------------------------------------------------------------------
    if FRONTEND_DIR.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(FRONTEND_DIR / "assets")) if (FRONTEND_DIR / "assets").exists() else StaticFiles(directory=str(FRONTEND_DIR)),
            name="assets",
        )

        @app.get("/")
        def index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))

        @app.get("/{name:path}")
        def static_passthrough(name: str):
            target = FRONTEND_DIR / name
            if target.is_file():
                return FileResponse(str(target))
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app


app = make_app()
