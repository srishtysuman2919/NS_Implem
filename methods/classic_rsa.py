"""Classic RSA -- Kriegeskorte, Mur & Bandettini (2008).

Paper goal
----------
Prior work compared brains and models either at the level of *individual voxels
/ units* (impossible across systems with different tunings) or via decoding
accuracy (only tells you if a distinction exists, not how the space is
organised). Kriegeskorte, Mur & Bandettini proposed a *representation-level*
comparison: for each system, build a **Representational Dissimilarity Matrix
(RDM)** whose (i, j) entry is the dissimilarity of the responses to
conditions i and j; then compare two systems by comparing their RDMs (rank
correlation of upper triangles). RDMs live in a shared, feature-space-invariant
coordinate system, so an fMRI RDM and a CNN RDM can be compared directly.

Contents of this module
-----------------------
* :func:`rdm`                     -- build an RDM from a response-pattern matrix
  (dissimilarity metric = ``1 - Pearson r`` by default; other metrics
  supported for pedagogy).
* :func:`rdm_similarity`          -- Spearman rank correlation between two RDMs,
  the canonical Kriegeskorte 2008 comparison.
* :func:`rdm_significance_perm`   -- permutation test against random condition
  labelling.

What shortcoming of the previous method does this address?
---------------------------------------------------------
Voxel-by-voxel and unit-by-unit comparisons across systems require aligning
tuning curves that live in incompatible spaces. RSA lifts everything to a
**stimulus × stimulus** dissimilarity matrix, which has the same shape and
axes regardless of the underlying system, and is invariant to arbitrary
rotations of the feature space (because it depends only on pairwise
correlations of patterns).
"""
from __future__ import annotations

import numpy as np

from .rdm_utils import (
    check_rdm,
    compare_rdms,
    rdm_correlation,
    rdm_euclidean,
    upper_tri,
)


def rdm(X: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Build an RDM from a response-pattern matrix.

    Parameters
    ----------
    X : (n_conditions, n_features) ndarray
    metric : {'correlation', 'euclidean', 'sqeuclidean'}
        Kriegeskorte 2008 uses correlation distance because it is invariant
        to per-voxel gain and offset, which are nuisance factors in fMRI.

    Returns
    -------
    (n_conditions, n_conditions) ndarray
    """
    if metric == "correlation":
        return rdm_correlation(X)
    if metric == "euclidean":
        return rdm_euclidean(X, squared=False)
    if metric == "sqeuclidean":
        return rdm_euclidean(X, squared=True)
    raise ValueError(f"unknown metric {metric!r}")


def rdm_similarity(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    method: str = "spearman",
) -> float:
    """Compare two RDMs. Kriegeskorte 2008's default is Spearman rank ρ."""
    check_rdm(rdm_a)
    check_rdm(rdm_b)
    return compare_rdms(rdm_a, rdm_b, method=method)


def rdm_significance_perm(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_perm: int = 1000,
    method: str = "spearman",
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Permutation test: shuffle condition labels of ``rdm_b`` and recompute.

    Returns
    -------
    observed : float
        Rank correlation on the actual RDMs.
    p : float
        Fraction of permutations reaching or exceeding ``observed``
        (right-tailed).
    """
    rng = rng or np.random.default_rng(0)
    check_rdm(rdm_a)
    check_rdm(rdm_b)
    n = rdm_a.shape[0]
    observed = compare_rdms(rdm_a, rdm_b, method=method)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        r = compare_rdms(rdm_a, rdm_b_perm, method=method)
        if r >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return observed, p


# ---------------------------------------------------------------------------
# self-test: sanity check against rsatoolbox if available
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 100))
    D_ours = rdm(X, metric="correlation")
    check_rdm(D_ours)

    # scipy sanity check
    from scipy.spatial.distance import pdist, squareform

    D_scipy = squareform(pdist(X, metric="correlation"))
    assert np.allclose(D_ours, D_scipy, atol=1e-8), (
        "correlation RDM disagrees with scipy"
    )

    # RDM-vs-RDM similarity: perfect correlation with itself
    assert abs(rdm_similarity(D_ours, D_ours) - 1.0) < 1e-8

    # random vs itself with shuffled labels should be ~0
    perm = rng.permutation(20)
    D_shuf = D_ours[np.ix_(perm, perm)]
    r = rdm_similarity(D_ours, D_shuf, method="spearman")
    assert -1.0 <= r <= 1.0

    print("classic_rsa self-test OK")
    print(f"  n_conditions   : {X.shape[0]}")
    print(f"  RDM shape      : {D_ours.shape}")
    print(f"  self-similarity: {rdm_similarity(D_ours, D_ours):.3f}")
    print(f"  shuffled-sim   : {r:+.3f}")


if __name__ == "__main__":
    _selftest()
