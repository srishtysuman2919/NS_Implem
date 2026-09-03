"""Stub for the Algonauts tutorial subset.

The Algonauts Project (https://algonauts.csail.mit.edu) publishes small
tutorial subsets designed for exactly this kind of RSA / encoding
comparison. URLs change per year, so we do not automate the download; if
you place the tutorial's `.mat` or `.npy` files under
``data/downloads/algonauts/`` this script will load them. Otherwise it's
a no-op -- the other datasets are sufficient for the experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from _utils import banner, DOWNLOAD_DIR


def main() -> None:
    d = DOWNLOAD_DIR / "algonauts"
    if not d.exists() or not any(d.iterdir()):
        banner("no Algonauts tutorial files found under data/downloads/algonauts/")
        banner("    (skipping -- see data/README.md for manual download)")
        return
    banner("Algonauts loader not implemented -- format changes yearly")
    banner("    files found:")
    for f in d.iterdir():
        banner(f"      {f.name}")


if __name__ == "__main__":
    main()
