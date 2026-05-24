"""File system browsing endpoints.

Restricted to a set of allow-listed roots configured at startup, so the UI can
only see what the Colab user authorised (dataset/model/output/samples roots).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}


@dataclass
class FsEntry:
    name: str
    path: str
    is_dir: bool
    size: int
    mtime: float


def _safe_resolve(target: str, roots: List[str]) -> Optional[Path]:
    """Resolve `target` and verify it is equal to or lives inside one of the allow-listed roots."""
    if not target:
        return None
    p = Path(target).expanduser().resolve()
    for r in roots:
        if not r:
            continue
        rp = Path(r).expanduser().resolve()
        # p == rp  (the root itself is allowed) or p is a sub-path of rp
        if p == rp or str(p).startswith(str(rp) + os.sep):
            return p
    return None


class FileSystem:
    def __init__(self, roots: List[str]):
        # de-dupe + drop empty
        self.roots = [r for r in dict.fromkeys(roots) if r]

    def update_roots(self, roots: List[str]) -> None:
        self.roots = [r for r in dict.fromkeys(roots) if r]

    def list(self, path: str) -> List[FsEntry]:
        target = _safe_resolve(path, self.roots)
        if target is None or not target.exists() or not target.is_dir():
            return []
        entries: List[FsEntry] = []
        for child in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append(FsEntry(
                name=child.name,
                path=str(child),
                is_dir=child.is_dir(),
                size=st.st_size,
                mtime=st.st_mtime,
            ))
        return entries

    def count_images(self, dir_path: str) -> int:
        target = _safe_resolve(dir_path, self.roots)
        if target is None or not target.exists() or not target.is_dir():
            return 0
        n = 0
        for child in target.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                n += 1
        return n

    def scan_dataset(self, root: str) -> List[dict]:
        """Scan dataset root for image subsets.

        Supports two layouts:
        1. Root contains sub-folders, each with images (kohya-style).
        2. Root contains images directly — treat root itself as a single subset.

        Any folder name is valid (no `N_concept` naming required).
        Repeats are NOT inferred from folder names — they come from UI settings.
        """
        target = _safe_resolve(root, self.roots)
        if target is None or not target.is_dir():
            return []
        out = []
        for child in sorted(target.iterdir()):
            if not child.is_dir():
                continue
            n = self.count_images(str(child))
            if n == 0:
                continue
            out.append({
                "image_dir": str(child),
                "num_repeats": 1,
                "concept": child.name,
                "num_images": n,
            })
        # If no subdirs with images, but root has images directly — use the root.
        if not out:
            root_imgs = self.count_images(str(target))
            if root_imgs > 0:
                out.append({
                    "image_dir": str(target),
                    "num_repeats": 1,
                    "concept": target.name,
                    "num_images": root_imgs,
                })
        return out

    def list_models(self, root: str) -> List[FsEntry]:
        target = _safe_resolve(root, self.roots)
        if target is None or not target.exists():
            return []
        out: List[FsEntry] = []
        for child in target.rglob("*"):
            if child.is_file() and child.suffix.lower() in MODEL_EXTS:
                try:
                    st = child.stat()
                except OSError:
                    continue
                out.append(FsEntry(
                    name=child.name,
                    path=str(child),
                    is_dir=False,
                    size=st.st_size,
                    mtime=st.st_mtime,
                ))
        return out

    def list_outputs(self, root: str) -> List[FsEntry]:
        """Final/intermediate .safetensors in the output dir, newest first."""
        target = _safe_resolve(root, self.roots)
        if target is None or not target.exists():
            return []
        items: List[FsEntry] = []
        for child in target.glob("*"):
            if child.is_file() and child.suffix.lower() in MODEL_EXTS:
                try:
                    st = child.stat()
                except OSError:
                    continue
                items.append(FsEntry(
                    name=child.name,
                    path=str(child),
                    is_dir=False,
                    size=st.st_size,
                    mtime=st.st_mtime,
                ))
        items.sort(key=lambda e: e.mtime, reverse=True)
        return items

    def list_samples(self, root: str) -> List[dict]:
        """Sample images grouped by inferred (epoch, prompt_idx).

        kohya names sample images like `<output_name>_<epoch>_<step>_<prompt>.png`
        — we parse what we can, fall back gracefully.
        """
        target = _safe_resolve(root, self.roots)
        if target is None or not target.exists():
            return []
        items = []
        pat_full = re.compile(r"^(.*?)_e(\d+)_s(\d+)(?:_(.+))?$")
        pat_loose = re.compile(r"e(\d+)|epoch[_-]?(\d+)|step[_-]?(\d+)", re.IGNORECASE)
        for child in sorted(target.rglob("*")):
            if not child.is_file() or child.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = child.stem
            epoch = 0
            step = 0
            prompt_tag = ""
            m = pat_full.match(stem)
            if m:
                epoch = int(m.group(2))
                step = int(m.group(3))
                prompt_tag = m.group(4) or ""
            else:
                for m2 in pat_loose.finditer(stem):
                    if m2.group(1) or m2.group(2):
                        epoch = int(m2.group(1) or m2.group(2))
                    elif m2.group(3):
                        step = int(m2.group(3))
            try:
                st = child.stat()
            except OSError:
                continue
            items.append({
                "name": child.name,
                "path": str(child),
                "rel": str(child.relative_to(target)),
                "epoch": epoch,
                "step": step,
                "prompt_tag": prompt_tag,
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
        return items
