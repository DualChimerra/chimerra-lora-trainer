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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Body
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


# Defined at module scope so FastAPI's dependency-resolver can introspect it
# reliably (a class nested inside a factory function makes FastAPI confused
# and it falls back to treating the param as a query scalar → 422
# `loc=["query","p"]` on POST /api/presets).
class SavePresetIn(BaseModel):
    name: str
    description: str = ""
    config: TrainConfig


def _read_env_paths() -> dict:
    """Initial paths come from the Colab cell via env vars."""
    return {
        "dataset_root": os.environ.get("LT_DATASET_ROOT", ""),
        "base_model_root": os.environ.get("LT_BASE_MODEL_ROOT", ""),
        "output_root": os.environ.get("LT_OUTPUT_ROOT", ""),
        "samples_root": os.environ.get("LT_SAMPLES_ROOT", ""),
        "sd_scripts_dir": os.environ.get("LT_SD_SCRIPTS_DIR", "/content/sd-scripts"),
        "anima_studio_dir": os.environ.get("LT_ANIMA_STUDIO_DIR", "/content/AnimaLoraStudio"),
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

    # state dir: LT_STATE_DIR wins (lets the Colab cache cell pin state on
    # Drive even when output_root is redirected to local SSD), then output_root,
    # then /content.
    state_override = os.environ.get("LT_STATE_DIR", "").strip()
    state = StateStore(
        primary_dir=state_override
            or os.path.join(init_paths.get("output_root", "") or "", ".lora_trainer"),
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
        cfg.paths.anima_studio_dir = init_paths["anima_studio_dir"]
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
        # kohya hardcodes sample output to "<output_dir>/sample", so we must
        # always scan output_root; samples_root is treated as an extra location
        # (e.g. for users who copy samples elsewhere). Caller can override with
        # ?path=... to inspect a single dir.
        cfg = get_cfg()
        if path:
            return fs.list_samples(path)
        roots = [cfg.paths.output_root, cfg.paths.samples_root]
        seen = set()
        out = []
        for r in roots:
            if not r:
                continue
            for item in fs.list_samples(r):
                if item["path"] in seen:
                    continue
                seen.add(item["path"])
                out.append(item)
        return out

    @app.get("/api/fs/file")
    def api_fs_file(path: str):
        """Stream a file (image/model) under one of the allow-listed roots."""
        from .filesystem import _safe_resolve
        resolved = _safe_resolve(path, fs.roots)
        if resolved is None or not resolved.is_file():
            raise HTTPException(404, "not found or not allowed")
        # immutable-ish: filename + mtime is the cache key on the client
        return FileResponse(
            str(resolved),
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Cached webp thumbnails for the gallery. Drive-mounted PNGs can be 1-3 MB
    # each — at 30+ samples this kills page load. PIL renders a ~256px webp
    # (~20 KB), cached on local SSD; subsequent loads are instant.
    THUMB_CACHE = Path("/tmp/lora_trainer_thumbs")
    THUMB_CACHE.mkdir(parents=True, exist_ok=True)

    @app.get("/api/fs/thumb")
    def api_fs_thumb(path: str, size: int = 256):
        from .filesystem import _safe_resolve
        import hashlib
        resolved = _safe_resolve(path, fs.roots)
        if resolved is None or not resolved.is_file():
            raise HTTPException(404, "not found or not allowed")
        size = max(64, min(int(size), 1024))
        try:
            mtime = int(resolved.stat().st_mtime)
        except OSError:
            raise HTTPException(404, "stat failed")
        key = hashlib.sha1(f"{resolved}|{mtime}|{size}".encode()).hexdigest()
        cached = THUMB_CACHE / f"{key}.webp"
        if not cached.exists():
            try:
                from PIL import Image
                with Image.open(resolved) as im:
                    im = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im
                    im.thumbnail((size, size), Image.LANCZOS)
                    im.save(cached, "webp", quality=82, method=4)
            except Exception:
                # fall back to streaming the original
                return FileResponse(
                    str(resolved),
                    headers={"Cache-Control": "public, max-age=86400"},
                )
        return FileResponse(
            str(cached),
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    # Manual local→Drive sync. Triggered by a button in the UI because the
    # Colab kernel queue is permanently blocked by the UI cell, so users can't
    # just call push_to_drive() interactively. The FastAPI process is separate,
    # so this works even while training runs.
    _sync_state = {"running": False, "last_at": 0.0, "last_msg": ""}

    @app.post("/api/sync/push_to_drive")
    def api_sync_push():
        import subprocess, threading
        src = os.environ.get("LT_OUTPUT_ROOT", "").strip()
        dst = os.environ.get("LT_DRIVE_OUTPUT_ROOT", "").strip()
        if not src or not dst:
            raise HTTPException(400, "local output or drive output not configured (run the local-cache cell first)")
        if src.rstrip("/") == dst.rstrip("/"):
            return {"ok": True, "skipped": "local==drive", **_sync_state}
        if _sync_state["running"]:
            return {"ok": True, "already_running": True, **_sync_state}
        def _run():
            _sync_state["running"] = True
            try:
                r = subprocess.run(
                    ["rsync", "-a", "--quiet", src.rstrip("/") + "/", dst.rstrip("/") + "/"],
                    timeout=3600,
                )
                _sync_state["last_msg"] = "ok" if r.returncode == 0 else f"rsync exit {r.returncode}"
            except Exception as e:
                _sync_state["last_msg"] = f"error: {e}"
            finally:
                _sync_state["last_at"] = time.time()
                _sync_state["running"] = False
        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "started": True, **_sync_state}

    @app.get("/api/sync/status")
    def api_sync_status():
        return {
            "enabled": bool(os.environ.get("LT_DRIVE_OUTPUT_ROOT", "").strip()),
            "src": os.environ.get("LT_OUTPUT_ROOT", ""),
            "dst": os.environ.get("LT_DRIVE_OUTPUT_ROOT", ""),
            **_sync_state,
        }

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

    @app.post("/api/presets")
    def api_presets_save(p: SavePresetIn = Body(...)):
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

    # Wipe local artifacts: samples, trained LoRA files, .lora_trainer state dir,
    # and the thumb cache. Operates on output_root/samples_root (which on Colab
    # point at the local SSD cache), NOT the Drive mirror.
    @app.post("/api/cleanup")
    def api_cleanup():
        import shutil
        if trainer.status.state in ("running", "starting"):
            raise HTTPException(409, "training is running — stop it first")

        cfg = get_cfg()
        removed = {"samples": 0, "loras": 0, "state_dirs": 0, "thumb_cache": 0}
        errors: list[str] = []

        def _rm_dir(p: Path) -> bool:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                    return True
            except Exception as e:
                errors.append(f"{p}: {e}")
            return False

        def _rm_file(p: Path) -> bool:
            try:
                p.unlink()
                return True
            except Exception as e:
                errors.append(f"{p}: {e}")
            return False

        # samples: kohya writes to <output_root>/sample; samples_root may hold copies
        sample_dirs: list[Path] = []
        if cfg.paths.output_root:
            sample_dirs.append(Path(cfg.paths.output_root) / "sample")
        if cfg.paths.samples_root:
            sample_dirs.append(Path(cfg.paths.samples_root))
        for d in sample_dirs:
            if not d.is_dir():
                continue
            for entry in d.iterdir():
                if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    if _rm_file(entry):
                        removed["samples"] += 1

        # trained LoRA files under output_root (recursive, skip the state dir)
        if cfg.paths.output_root:
            out_root = Path(cfg.paths.output_root)
            state_subdir = out_root / ".lora_trainer"
            if out_root.is_dir():
                for f in out_root.rglob("*.safetensors"):
                    try:
                        f.relative_to(state_subdir)
                        continue  # inside state dir, handled below
                    except ValueError:
                        pass
                    if _rm_file(f):
                        removed["loras"] += 1

        # .lora_trainer dirs (state + presets). Wipe both primary and fallback.
        for d in {state.dir, state.primary, state.fallback}:
            if _rm_dir(d):
                removed["state_dirs"] += 1
        # re-create an empty state dir so subsequent saves work
        try:
            state._ensure()
        except Exception as e:
            errors.append(f"state re-init: {e}")

        # thumb cache
        if THUMB_CACHE.is_dir():
            for f in THUMB_CACHE.iterdir():
                if f.is_file() and _rm_file(f):
                    removed["thumb_cache"] += 1

        return {"ok": True, "removed": removed, "errors": errors}

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
