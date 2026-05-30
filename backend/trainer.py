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
from pathlib import Path
from typing import Optional, List, Callable, Awaitable

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
        self._log_buffer: List[dict] = []
        self._log_buffer_max = 5000
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
        self._log_buffer.append(event)
        if len(self._log_buffer) > self._log_buffer_max:
            self._log_buffer = self._log_buffer[-self._log_buffer_max:]
        for fn in list(self._log_subs):
            try:
                await fn(event)
            except Exception:
                pass

    def snapshot(self) -> dict:
        return {
            "status": self.status.model_dump(),
            "tail": self._log_buffer[-300:],
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

        self.status = TrainStatus(
            state="starting",
            total_epochs=cfg.training.max_train_epochs,
            total_steps=compute_total_steps(cfg),
            started_at=time.time(),
        )
        await self._broadcast({"type": "status", "status": self.status.model_dump()})
        await self._broadcast({"type": "log", "stream": "system", "line": "$ " + " ".join(argv)})

        try:
            self.proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=(cfg.paths.anima_studio_dir if is_anima_lokr(cfg)
                     else cfg.paths.sd_scripts_dir),
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
        m = PROGRESS_RX.search(line) or ANIMA_STUDIO_RX.search(line)
        if m:
            try:
                self.status.step = int(m.group("step"))
                # ANIMA_STUDIO_RX's `total` is optional (their plain logger may
                # omit it). Keep the prior total in that case.
                total = m.groupdict().get("total")
                if total:
                    self.status.total_steps = int(total)
                if m.groupdict().get("epoch"):
                    self.status.epoch = int(m.group("epoch"))
                if m.groupdict().get("total_epochs"):
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
