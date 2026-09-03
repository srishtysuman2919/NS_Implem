"""Experiment 3 -- layer-wise LLM ↔ brain alignment.

Extract layer-wise hidden-state embeddings from GPT-2-small (or distilgpt2
if you want it faster) on the Pereira-style sentence stimuli from
``data/cache/pereira2018.npz``, then compute layer-wise alignment against
the brain data under THREE metrics:

    * RSA          (Spearman ρ of RDMs)
    * linear CKA
    * ridge encoding (5-fold CV, mean per-voxel Pearson r)

The community-wide finding is that **middle transformer layers align best
with high-level language cortex**. If this repo's synthesised brain data
was calibrated correctly, the plotted curves should peak somewhere in the
middle of the network.

If `transformers` / `torch` is not installed, this script skips gracefully.
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


RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(model_name: str = "distilgpt2") -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError:
        print("transformers / torch not installed -- experiment skipped")
        return

    from methods.llm_alignment import (
        get_layerwise_embeddings,
        layerwise_rsa,
        layerwise_cka,
        layerwise_ridge,
    )

    ds = load_dataset("pereira2018")
    sentences = [str(s) for s in ds["sentences"]]
    brain = ds["brain_responses"]
    brain_rdm = ds["brain_rdm"]
    n = len(sentences)
    print(f"loaded {n} sentences, brain shape {brain.shape}, model = {model_name}")

    emb, names = get_layerwise_embeddings(sentences, model_name=model_name)
    print(f"got embeddings: {emb.shape[0]} layers x {n} sentences x {emb.shape[2]} dim")

    rsa_curve = layerwise_rsa(emb, brain_rdm)
    cka_curve = layerwise_cka(emb, brain)
    ridge_curve = layerwise_ridge(emb, brain, alphas=(1.0, 10.0, 100.0, 1000.0),
                                  k=5)

    df = pd.DataFrame({
        "layer": names,
        "RSA":   rsa_curve,
        "CKA":   cka_curve,
        "ridge": ridge_curve,
    })
    print("\nLayer-wise alignment:")
    print(df.round(3).to_string(index=False))
    df.to_csv(RESULTS_DIR / "03_llm_layerwise.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in ["RSA", "CKA", "ridge"]:
        v = df[m].to_numpy()
        vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax.plot(range(len(df)), vn, marker="o", label=m)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["layer"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("normalised alignment (higher = closer to brain)")
    ax.set_title(f"Layer-wise {model_name} ↔ brain alignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_llm_layerwise.png", dpi=140)
    plt.close(fig)

    print(f"\nresults written to {RESULTS_DIR}/03_llm_layerwise.*")


if __name__ == "__main__":
    main()
