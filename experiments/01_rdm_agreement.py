"""Experiment 1 -- do the different comparison metrics agree?

On the original multi-run Haxby 2001 subject-2 dataset, we build:

1. A **classic-RSA RDM**       (1 - Pearson r on run-averaged patterns)
2. A **crossnobis RDM**        (cross-validated Mahalanobis)
3. A **PCM G-matrix**          -> squared-Euclidean RDM implied by G
4. A **linear-CKA**            score matrix over "systems" (see below)
5. A **shape-distance**        matrix over the same systems

For (4)-(5) we treat each *run* as a "system" and ask how similar the runs
are to one another under each metric. Because all runs come from the same
underlying representation, a good similarity index should rank runs of the
same condition-set as most similar to themselves.

The comparison table shows the rank-correlation between every pair of
metrics on the same data -- when they disagree, we know we are seeing the
particular assumption each metric encodes (noise bias, scale invariance,
metric-ness, ...).
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data._utils import load_dataset
from methods.classic_rsa import rdm, rdm_similarity
from methods.crossnobis import crossnobis_rdm
from methods.pcm import unbiased_G, predict_rdm_from_G
from methods.cka import linear_cka
from methods.shape_distance import angular_shape_distance


RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    from methods.rdm_utils import compare_rdms
    A = np.zeros((len(a), len(a)))
    A[np.triu_indices(len(a), k=1)] = a
    A = A + A.T
    B = np.zeros((len(b), len(b)))
    B[np.triu_indices(len(b), k=1)] = b
    B = B + B.T
    return compare_rdms(A, B, method="spearman")


def main() -> None:
    ds = load_dataset("haxby")
    patterns = ds["patterns"]
    reference_rdm = ds["reference_rdm"]
    n_runs = patterns.shape[0]

    # -----------------------------------------------------------------
    # Build the "same-data" RDMs
    # -----------------------------------------------------------------
    mean_patterns = patterns.mean(axis=0)            # (92, n_vox)

    rdm_classic = rdm(mean_patterns, metric="correlation")
    rdm_cn = crossnobis_rdm(patterns)
    G = unbiased_G(patterns)
    rdm_pcm = predict_rdm_from_G(G)

    # Normalise for plotting (crossnobis / PCM can be negative or in
    # different absolute scales -- the RDM comparison uses rank, so
    # scale doesn't matter, but the plot is easier to read.)
    rdms = {
        "classic RSA (1-r)": rdm_classic,
        "crossnobis":        rdm_cn,
        "PCM (from G)":      rdm_pcm,
        "odd-run reference": reference_rdm,
    }

    # -----------------------------------------------------------------
    # Pairwise Spearman between RDMs
    # -----------------------------------------------------------------
    names = list(rdms.keys())
    n = len(names)
    agree = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            agree[i, j] = rdm_similarity(rdms[a], rdms[b], method="spearman")
    agree_df = pd.DataFrame(agree, index=names, columns=names)

    # -----------------------------------------------------------------
    # CKA and shape distance between runs
    # -----------------------------------------------------------------
    cka_mat = np.zeros((n_runs, n_runs))
    shape_mat = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            cka_mat[i, j] = linear_cka(patterns[i], patterns[j])
            shape_mat[i, j] = angular_shape_distance(patterns[i], patterns[j])

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print("Pairwise Spearman ρ between RDMs on the same data:")
    print(agree_df.round(3).to_string())
    agree_df.to_csv(RESULTS_DIR / "01_rdm_agreement.csv")

    print()
    print(f"CKA between runs -- mean = {cka_mat[np.triu_indices(n_runs, 1)].mean():.3f}")
    print(f"Shape distance between runs -- mean = "
          f"{shape_mat[np.triu_indices(n_runs, 1)].mean():.3f} rad")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, len(rdms), figsize=(4 * len(rdms), 4))
    for ax, (name, D) in zip(axes, rdms.items()):
        im = ax.imshow(D, aspect="equal")
        ax.set_title(name, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Same data, four dissimilarity structures")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_rdms.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agree, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{agree[i, j]:.2f}", ha="center", va="center",
                    color="white" if agree[i, j] < 0.5 else "black", fontsize=9)
    ax.set_title("Spearman ρ between RDMs")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_rdm_agreement.png", dpi=140)
    plt.close(fig)

    print(f"\nresults written to {RESULTS_DIR}/01_*.png and 01_rdm_agreement.csv")


if __name__ == "__main__":
    main()
