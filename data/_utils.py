"""Small shared utilities for the data prep scripts."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def banner(msg: str) -> None:
    print(f"[{Path(__file__).parent.name}] {msg}")


def npz_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npz"


def save_dataset(name: str, **arrays) -> Path:
    p = npz_path(name)
    np.savez(p, **arrays)
    return p


def load_dataset(name: str) -> dict:
    p = npz_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"cached dataset {name!r} not found at {p}. "
            f"Run `python data/prepare_all.py` first."
        )
    return dict(np.load(p, allow_pickle=True))
