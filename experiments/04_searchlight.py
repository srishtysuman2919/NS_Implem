"""Experiment 4 -- searchlight RSA on original Haxby 2001 data.

Odd runs define the reference RDM and even runs define the searchlight
patterns, keeping the comparison independent across run halves.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from data._utils import load_dataset
from methods.searchlight_rsa import searchlight_similarity_map


RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ds = load_dataset("haxby")
    volume = ds["volume"]
    mask = ds["mask"]
    ref = ds["reference_rdm"]
    print(f"volume {volume.shape}, mask voxels = {int(mask.sum())}")

    r_map = searchlight_similarity_map(
        volume, ref, radius=2.5, mask=mask, metric="correlation",
        compare="spearman",
    )
    peak = np.unravel_index(np.nanargmax(r_map), r_map.shape)
    print(f"peak voxel {peak}")
    print(f"peak ρ = {np.nanmax(r_map):+.3f}")

    # Save a 3-panel montage of orthogonal slices through the peak
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = [
        ("axial (z=peak)", r_map[:, :, peak[2]]),
        ("coronal (y=peak)", r_map[:, peak[1], :]),
        ("sagittal (x=peak)", r_map[peak[0], :, :]),
    ]
    for ax, (title, img) in zip(axes, slices):
        im = ax.imshow(img.T, origin="lower", cmap="hot",
                       vmin=0, vmax=float(np.nanmax(r_map)))
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Searchlight ρ against reference RDM")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_searchlight.png", dpi=140)
    plt.close(fig)

    np.save(RESULTS_DIR / "04_searchlight_map.npy", r_map)
    print(f"\nresults written to {RESULTS_DIR}/04_searchlight.*")


if __name__ == "__main__":
    main()
