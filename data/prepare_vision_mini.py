"""Prepare a mini vision fMRI dataset for the searchlight demo.

Small enough to run in seconds. Produces:
    * A ``(nx, ny, nz, n_conditions)`` volume of beta maps.
    * A binary brain mask.
    * A "true" reference RDM the hotspot region should recover.
    * Labels for the conditions (for coloured RDM plots).

If a lightweight real subset (e.g. a preprocessed BOLD5000 tutorial slice)
is available under ``data/downloads/bold5000_mini/``, the script would load
it instead -- see the stub in :func:`_try_real`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import numpy as np

from _utils import banner, save_dataset


N_COND = 40
CATS = np.array(["face", "body", "object", "scene"])


def _try_real() -> dict | None:
    """Placeholder for a real BOLD5000/NSD mini subset."""
    return None


def _synthesise(rng: np.random.Generator) -> dict:
    nx, ny, nz = 16, 16, 16
    labels = np.repeat(np.arange(4), N_COND // 4)
    latent_dim = 6
    proto = rng.standard_normal((4, latent_dim)) * 1.5
    latent = proto[labels] + 0.5 * rng.standard_normal((N_COND, latent_dim))

    # Category-informative voxels live in a small hotspot cube
    hotspot = np.zeros((nx, ny, nz), dtype=bool)
    hotspot[6:10, 6:10, 6:10] = True
    n_hot = hotspot.sum()

    W_hot = rng.standard_normal((latent_dim, n_hot)) / np.sqrt(latent_dim)
    hot_patterns = latent @ W_hot                       # (N_COND, n_hot)

    volume = 0.3 * rng.standard_normal((nx, ny, nz, N_COND))
    # inject signal
    hx, hy, hz = np.where(hotspot)
    for k in range(n_hot):
        volume[hx[k], hy[k], hz[k], :] += hot_patterns[:, k]

    # crude brain mask: sphere
    xx, yy, zz = np.indices((nx, ny, nz))
    center = np.array([nx // 2, ny // 2, nz // 2])
    mask = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 +
            (zz - center[2]) ** 2) <= 7 ** 2

    # reference RDM: what a noise-free reader of the hotspot would see
    from methods.classic_rsa import rdm

    ref_rdm = rdm(latent @ W_hot, metric="correlation")

    return dict(
        volume=volume.astype(np.float32),
        mask=mask,
        labels=labels.astype(np.int64),
        label_names=CATS,
        hotspot_center=np.array(center),
        reference_rdm=ref_rdm.astype(np.float32),
        source=np.array("synthesised (16x16x16 volume, 4-category hotspot)"),
    )


def main() -> None:
    rng = np.random.default_rng(1)
    real = _try_real()
    if real is not None:
        banner("using real BOLD5000-mini subset from data/downloads/")
        data = real
    else:
        banner("no real vision mini-dataset present; synthesising")
        data = _synthesise(rng)
        banner(f"    -> volume {data['volume'].shape}, {N_COND} conditions")

    p = save_dataset("vision_mini", **data)
    banner(f"wrote {p}")


if __name__ == "__main__":
    main()
