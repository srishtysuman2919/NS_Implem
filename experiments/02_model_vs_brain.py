"""Experiment 2 -- does the model ranking against IT change with the metric?

For each layer of the small vision model (`resnet18` if torchvision is
available, else a random-projection stand-in from the prep script), we
compute how similar that layer is to the IT-fMRI RDM under FIVE metrics:

    * classic RSA         (Spearman ρ of RDMs)
    * crossnobis RSA      (Spearman ρ of the crossnobis RDM with the model RDM)
    * linear CKA          (patterns vs. patterns, no RDMs)
    * angular shape dist  (patterns vs. patterns, distance-not-similarity)
    * ridge encoding      (k-fold CV Pearson r, patterns predict brain)

Then we rank the layers under each metric and produce a comparison table
and a plot. The interesting question is: do different metrics rank the
layers *differently*? Where they disagree, we tie the disagreement back
to what each paper's introduction claims its method fixes.

The model and brain measurements use the same repeated NSD images in the
same order. The cache contains subject-1 betas and pretrained ResNet-18
features extracted from those exact images.
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
from methods.cka import linear_cka
from methods.shape_distance import angular_shape_distance
from methods.ridge_encoding import ridge_cv_score


RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ds = load_dataset("nsd_model_brain")

    # Run-averaged brain patterns
    brain_patterns = ds["patterns"].mean(axis=0)
    brain_multi = ds["patterns"]

    layer_names = list(map(str, ds["layer_names"]))
    model_layers = list(ds["embeddings"])

    n = brain_patterns.shape[0]
    if any(np.asarray(layer).shape[0] != n for layer in model_layers):
        raise ValueError("NSD brain and model stimulus counts do not match")
    print(f"loaded {n} exactly matched NSD stimuli")

    # Precompute brain RDMs
    brain_rdm_classic = rdm(brain_patterns, metric="correlation")
    brain_rdm_cn = crossnobis_rdm(brain_multi)

    rows = []
    for name, X in zip(layer_names, model_layers):
        X = np.asarray(X, dtype=float)
        mrdm = rdm(X, metric="correlation")

        classic = rdm_similarity(mrdm, brain_rdm_classic, method="spearman")
        cn = rdm_similarity(mrdm, brain_rdm_cn, method="spearman")
        cka = linear_cka(X, brain_patterns)
        shape = angular_shape_distance(X, brain_patterns)
        ridge_out = ridge_cv_score(
            X, brain_patterns, alphas=[1.0, 10.0, 100.0, 1000.0], k=5,
        )
        ridge = float(np.nanmean(ridge_out["scores"]))
        rows.append(dict(
            layer=name,
            classic_RSA=classic,
            crossnobis_RSA=cn,
            linear_CKA=cka,
            angular_shape_dist=shape,      # note: distance, lower = better
            ridge_encoding_r=ridge,
        ))

    df = pd.DataFrame(rows).set_index("layer")
    print("\nMetric-by-layer table:")
    print(df.round(3).to_string())
    df.to_csv(RESULTS_DIR / "02_model_vs_brain.csv")

    # Ranking table
    def _rank(series: pd.Series, ascending: bool) -> pd.Series:
        return series.rank(ascending=ascending, method="min").astype(int)

    ranks = pd.DataFrame({
        "classic RSA":       _rank(df["classic_RSA"], ascending=False),
        "crossnobis RSA":    _rank(df["crossnobis_RSA"], ascending=False),
        "linear CKA":        _rank(df["linear_CKA"], ascending=False),
        "shape dist. (rev)": _rank(df["angular_shape_dist"], ascending=True),
        "ridge encoding":    _rank(df["ridge_encoding_r"], ascending=False),
    }, index=df.index)
    print("\nRanks (1 = best model layer under this metric):")
    print(ranks.to_string())
    ranks.to_csv(RESULTS_DIR / "02_model_vs_brain_ranks.csv")

    # Plot: parallel-coordinates style layer-vs-metric
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_to_plot = ["classic_RSA", "crossnobis_RSA", "linear_CKA",
                       "ridge_encoding_r"]
    for m in metrics_to_plot:
        # min-max normalise for shared axis
        v = df[m]
        vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax.plot(range(len(df)), vn, marker="o", label=m)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("normalised score  (higher = more similar)")
    ax.set_title("Do different metrics agree on the model-vs-brain layer ranking?")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_model_vs_brain.png", dpi=140)
    plt.close(fig)

    print(f"\nresults written to {RESULTS_DIR}/02_*.png/.csv")


if __name__ == "__main__":
    main()
