"""Trainer subprocess manager.

Spawns accelerate-launched sd-scripts, streams stdout/stderr to subscribers
(WebSocket clients), parses progress lines, and lets the UI stop the run.
"""
from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Optional, List, Callable, Awaitable, Deque

from .schemas import TrainConfig, TrainStatus
from .config_builder import build_command, compute_total_steps
from .anima_lokr import is_anima_lokr


# kohya/tqdm progress lines look like:
#   "steps:   3%|▎         | 12/400 [00:34<18:30,  2.86s/it, avr_loss=0.123, lr=1.0e-4]"
PROGRESS_RX = re.compile(
    r"steps:\s+\d+%\|[^|]*\|\s+(?P<step>\d+)/(?P<total>\d+)"
    r"(?:.*?avr_loss=(?P<loss>[0-9.eE+\-]+))?"
    r"(?:.*?lr=(?P<lr>[0-9.eE+\-]+))?",
    re.IGNORECASE,
)
EPOCH_RX = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
SAVE_RX = re.compile(r"saving checkpoint:.*?epoch[_-]?(\d+)", re.IGNORECASE)
# AnimaLoraStudio plain progress (rich disabled): "epoch=3 step=120 loss=0.1234 lr=1.00e-04 speed=...".
# Their other shape (rich enabled, end='\r') is identical key=value soup, so the
# same regex covers both.
ANIMA_STUDIO_RX = re.compile(
    r"epoch\s*[=\s]\s*(?P<epoch>\d+)\s*(?:/\s*(?P<total_epochs>\d+))?"
    r".*?step\s*[=\s]\s*(?P<step>\d+)\s*(?:/\s*(?P<total>\d+))?"
    r"(?:.*?loss\s*=\s*(?P<loss>[0-9.eE+\-]+))?"
    r"(?:.*?lr\s*=\s*(?P<lr>[0-9.eE+\-]+))?",
    re.IGNORECASE,
)


class Trainer:
    def __init__(self):
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.status = TrainStatus()
        self._log_subs: List[Callable[[dict], Awaitable[None]]] = []
        # Ring buffer: maxlen handles trimming in O(1), no per-message slicing.
        self._log_buffer: Deque[dict] = deque(maxlen=5000)
        self._tail_task: Optional[asyncio.Task] = None
        self._workdir: Optional[Path] = None
        self._argv: Optional[List[str]] = None

    # --- pub/sub --------------------------------------------------------
    def subscribe(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        self._log_subs.append(fn)

    def unsubscribe(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        try:
            self._log_subs.remove(fn)
        except ValueError:
            pass

    async def _broadcast(self, event: dict) -> None:
        event = {"ts": time.time(), **event}
        self._log_buffer.append(event)  # deque(maxlen) drops the oldest itself
        for fn in list(self._log_subs):
            try:
                await fn(event)
            except Exception:
                pass

    def snapshot(self) -> dict:
        # deque has no slicing; islice from the tail offset avoids materializing
        # the whole 5000-entry buffer just to drop all but the last 300.
        n = len(self._log_buffer)
        tail = list(islice(self._log_buffer, max(0, n - 300), n))
        return {
            "status": self.status.model_dump(),
            "tail": tail,
            "argv": self._argv,
        }

    # --- lifecycle ------------------------------------------------------
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, cfg: TrainConfig, workdir: Path) -> None:
        if self.is_running():
            raise RuntimeError("Training already in progress")

        self._workdir = workdir
        argv, env = build_command(cfg, workdir)
        self._argv = argv
        cwd = (cfg.paths.anima_studio_dir if is_anima_lokr(cfg)
               else cfg.paths.sd_scripts_dir)

        self.status = TrainStatus(
            state="starting",
            total_epochs=cfg.training.max_train_epochs,
            total_steps=compute_total_steps(cfg),
            started_at=time.time(),
        )
        await self._broadcast({"type": "status", "status": self.status.model_dump()})
        await self._broadcast({"type": "log", "stream": "system", "line": "$ " + " ".join(argv)})

        # Preflight: uvloop turns a missing cwd / script / executable into a
        # bare `FileNotFoundError: [Errno 2] No such file or directory` with no
        # filename attached. For Anima+LoKr we also pre-check the four model
        # paths up front — otherwise the script burns 2-3 minutes loading the
        # transformer and VAE before tripping on text-encoder path mistakes.
        missing = self._preflight(argv, cwd, is_anima_lokr(cfg), cfg)
        if missing:
            self.status.state = "error"
            self.status.message = missing
            await self._broadcast({"type": "status", "status": self.status.model_dump()})
            await self._broadcast({"type": "log", "stream": "system", "line": f"[preflight] {missing}"})
            raise RuntimeError(missing)

        try:
            self.proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as e:
            self.status.state = "error"
            self.status.message = f"Cannot launch: {e}"
            await self._broadcast({"type": "status", "status": self.status.model_dump()})
            raise

        self.status.pid = self.proc.pid
        self.status.state = "running"
        await self._broadcast({"type": "status", "status": self.status.model_dump()})

        self._tail_task = asyncio.create_task(self._tail())

    @staticmethod
    def _preflight(argv: List[str], cwd: str, anima_lokr: bool,
                   cfg: Optional[TrainConfig] = None) -> Optional[str]:
        """Return a user-facing error message if the subprocess will fail to
        spawn or trip on a config typo seconds into execution, else None.
        Checked in order: cwd → script file → executable → model paths.
        """
        import shutil
        if not cwd:
            return ("Папка для запуска не задана. Заполни "
                    f"{'paths.anima_studio_dir' if anima_lokr else 'paths.sd_scripts_dir'} "
                    "в настройках.")
        if not os.path.isdir(cwd):
            return (f"Папка `{cwd}` не существует. "
                    + ("Похоже AnimaLoraStudio не склонирована. Запусти в Colab-ячейке:\n"
                       "  `!git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio /content/AnimaLoraStudio`"
                       if anima_lokr else
                       "Похоже sd-scripts не склонированы. Проверь paths.sd_scripts_dir."))
        # argv[1] is the training script for the anima-lokr path
        # (python interpreter + script). For kohya it's `accelerate launch <script>`.
        if anima_lokr and len(argv) >= 2 and not os.path.isfile(argv[1]):
            return (f"Тренировочный скрипт не найден: `{argv[1]}`. "
                    "Проверь что AnimaLoraStudio клонирована полностью и что "
                    "paths.anima_studio_dir указывает на корень репо.")
        # Executable resolvable (absolute path exists, or it's in PATH).
        exe = argv[0]
        if os.path.isabs(exe):
            if not os.path.isfile(exe):
                return f"Исполняемый файл не найден: `{exe}`."
        else:
            if shutil.which(exe) is None:
                hint = " (`pip install accelerate`)" if exe == "accelerate" else ""
                return f"`{exe}` не найден в PATH{hint}."

        # AnimaLoraStudio model paths: file-vs-directory mistakes here cost
        # users 2-3 minutes because the script loads the transformer (~6 GB)
        # and VAE before reaching the text encoder. Catch them now.
        if anima_lokr and cfg is not None:
            m = cfg.model
            # Files
            for label, p in [("model.pretrained_model_name_or_path",
                              m.pretrained_model_name_or_path),
                             ("model.anima_vae", m.anima_vae)]:
                if not p:
                    return f"`{label}` не задан."
                if not os.path.isfile(p):
                    return f"Файл `{p}` не найден ({label})."
            # Directories — and Qwen3 needs HF format, not a .safetensors file.
            for label, p, marker in [
                ("model.anima_qwen3", m.anima_qwen3, "tokenizer_config.json"),
                ("model.anima_t5_tokenizer_path",
                 m.anima_t5_tokenizer_path, "tokenizer_config.json"),
            ]:
                if not p:
                    return f"`{label}` не задан."
                if os.path.isfile(p):
                    return (f"`{label}` указывает на файл, а нужна **директория** "
                            f"в HuggingFace-формате (`{p}`).\n"
                            + ("Qwen3 надо скачать целиком:\n"
                               "  `from huggingface_hub import snapshot_download`\n"
                               "  `snapshot_download(\"Qwen/Qwen3-0.6B-Base\", local_dir=\"/path/to/qwen3-dir\")`\n"
                               "и в поле прописать путь к этой директории."
                               if "qwen3" in label.lower() else
                               "T5 tokenizer — это директория с tokenizer_config.json внутри."))
                if not os.path.isdir(p):
                    return f"Директория `{p}` не существует ({label})."
                if not os.path.isfile(os.path.join(p, marker)):
                    return (f"В директории `{p}` нет `{marker}` ({label}). "
                            "Похоже модель скачана не полностью.")
        return None

    async def _tail(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        # tqdm rewrites the progress bar with \r, not \n, so reading by newline
        # buffers the bar until a real log line shows up. Split on either.
        buf = bytearray()
        last_status_emit = 0.0
        last_progress_line = ""
        try:
            while True:
                chunk = await self.proc.stdout.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                while True:
                    idx = -1
                    sep = 0
                    for i, b in enumerate(buf):
                        if b == 0x0a or b == 0x0d:  # \n or \r
                            idx = i
                            sep = b
                            break
                    if idx == -1:
                        break
                    raw = bytes(buf[:idx])
                    del buf[: idx + 1]
                    line = raw.decode("utf-8", errors="replace")
                    if not line.strip():
                        continue
                    is_progress = self._parse_line(line)
                    if is_progress:
                        # tqdm rewrites — don't spam the log; just push status.
                        if line != last_progress_line:
                            last_progress_line = line
                            now = time.time()
                            if now - last_status_emit > 0.5:
                                last_status_emit = now
                                await self._broadcast({"type": "status", "status": self.status.model_dump()})
                    else:
                        await self._broadcast({"type": "log", "stream": "stdout", "line": line})
            # flush trailing partial line
            if buf:
                line = bytes(buf).decode("utf-8", errors="replace")
                if line.strip():
                    is_progress = self._parse_line(line)
                    if is_progress:
                        await self._broadcast({"type": "status", "status": self.status.model_dump()})
                    else:
                        await self._broadcast({"type": "log", "stream": "stdout", "line": line})
        except Exception as e:
            await self._broadcast({"type": "log", "stream": "system", "line": f"[tail error] {e}"})

        rc = await self.proc.wait()
        if rc == 0:
            self.status.state = "finished"
            self.status.message = "Training finished"
        elif self.status.state == "stopping":
            self.status.state = "finished"
            self.status.message = "Stopped by user"
        else:
            self.status.state = "error"
            self.status.message = f"Process exited with code {rc}"
        await self._broadcast({"type": "status", "status": self.status.model_dump()})

    def _parse_line(self, line: str) -> bool:
        """Return True if the line is a tqdm progress update (not a real log line)."""
        m = PROGRESS_RX.search(line)
        is_anima = False
        if not m:
            m = ANIMA_STUDIO_RX.search(line)
            is_anima = bool(m)
        if m:
            try:
                self.status.step = int(m.group("step"))
                # ANIMA_STUDIO_RX's `total` is optional (their plain logger may
                # omit it). Keep the prior total in that case.
                total = m.groupdict().get("total")
                if total:
                    self.status.total_steps = int(total)
                total_epochs = m.groupdict().get("total_epochs")
                if m.groupdict().get("epoch"):
                    ep = int(m.group("epoch"))
                    # AnimaLoraStudio's plain logger (we force no_progress=True)
                    # prints the 0-based loop index as `epoch={n}` with no
                    # `/total`; its rich form is `epoch n/m` and already 1-based.
                    # Bump only the plain 0-based case so the UI matches kohya.
                    if is_anima and not total_epochs:
                        ep += 1
                    self.status.epoch = ep
                if total_epochs:
                    self.status.total_epochs = int(m.group("total_epochs"))
                if m.group("loss"):
                    self.status.loss = float(m.group("loss"))
                if m.group("lr"):
                    self.status.lr = float(m.group("lr"))
                started = self.status.started_at or time.time()
                elapsed = max(1.0, time.time() - started)
                if self.status.step > 0 and self.status.total_steps > 0:
                    per_step = elapsed / self.status.step
                    remaining = max(0, self.status.total_steps - self.status.step)
                    self.status.eta_seconds = int(per_step * remaining)
            except (ValueError, TypeError):
                pass
            return True
        me = EPOCH_RX.search(line)
        if me:
            try:
                self.status.epoch = int(me.group(1))
                self.status.total_epochs = int(me.group(2))
            except ValueError:
                pass
        return False

    async def stop(self) -> None:
        if not self.is_running():
            return
        self.status.state = "stopping"
        await self._broadcast({"type": "status", "status": self.status.model_dump()})
        assert self.proc is not None
        try:
            self.proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=20)
        except asyncio.TimeoutError:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass

    def clear_logs(self) -> None:
        self._log_buffer.clear()
