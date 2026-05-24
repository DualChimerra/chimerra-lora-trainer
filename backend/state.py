"""Persistent state: last-used config + saved presets.

Stored under {output_root}/.lora_trainer/ so it survives Colab restarts when
Drive is mounted, or in /content as a fallback.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, List

from .schemas import TrainConfig, Preset


class StateStore:
    def __init__(self, primary_dir: str, fallback_dir: str = "/content/.lora_trainer"):
        self.primary = Path(primary_dir) if primary_dir else Path(fallback_dir)
        self.fallback = Path(fallback_dir)
        self._ensure()

    def _ensure(self) -> Path:
        for candidate in (self.primary, self.fallback):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                (candidate / "presets").mkdir(exist_ok=True)
                self.dir = candidate
                return candidate
            except Exception:
                continue
        raise RuntimeError("Cannot create state directory")

    # --- config ----------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.dir / "last_config.json"

    def load_config(self) -> Optional[TrainConfig]:
        if not self.config_path.exists():
            return None
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            return TrainConfig.model_validate(data)
        except Exception:
            return None

    def save_config(self, cfg: TrainConfig) -> None:
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.config_path)

    # --- presets ---------------------------------------------------------
    @property
    def presets_dir(self) -> Path:
        return self.dir / "presets"

    def list_presets(self) -> List[Preset]:
        out: List[Preset] = []
        for p in sorted(self.presets_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append(Preset.model_validate(data))
            except Exception:
                continue
        return out

    def _preset_path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "-_.").strip() or "preset"
        return self.presets_dir / f"{safe}.json"

    def save_preset(self, name: str, description: str, config: TrainConfig) -> Preset:
        preset = Preset(
            name=name,
            description=description,
            created_at=time.time(),
            config=config,
        )
        path = self._preset_path(name)
        path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")
        return preset

    def delete_preset(self, name: str) -> bool:
        path = self._preset_path(name)
        if path.exists():
            path.unlink()
            return True
        return False
