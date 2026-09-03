"""Small shared helpers used by every method module.

Nothing heavy or method-specific lives here -- just RDM plumbing that every
downstream file needs, written explicitly so the shapes and indexing are
obvious.

Conventions
-----------
Response patterns are stored as ``X`` of shape ``(n_conditions, n_features)``:
one row per stimulus/condition, one column per voxel / model unit. This
matches the convention in Kriegeskorte 2008 and Diedrichsen & Kriegeskorte 2017.

An RDM is a square, symmetric, zero-diagonal ``(n_conditions, n_conditions)``
matrix of pairwise dissimilarities.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# RDM construction
# ---------------------------------------------------------------------------
def _pearson_row(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D vectors, written out explicitly."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def rdm_correlation(X: np.ndarray) -> np.ndarray:
    """Classic RSA dissimilarity: ``1 - Pearson r`` between condition patterns.

    Parameters
    ----------
    X : (n_conditions, n_features) ndarray
        Response patterns.

    Returns
    -------
    (n_conditions, n_conditions) ndarray
        Symmetric, zero-diagonal RDM. Values in ``[0, 2]``.

    Notes
    -----
    Written as an explicit double loop first (for readability) and then
    vectorised. The vectorised path is used at runtime; the loop is kept as a
    reference implementation in ``_rdm_correlation_loop``.
    """
    # centre each row
    Xc = X - X.mean(axis=1, keepdims=True)
    # row norms
    norms = np.sqrt((Xc * Xc).sum(axis=1))
    # avoid division by zero
    norms[norms == 0] = 1.0
    Xn = Xc / norms[:, None]
    R = Xn @ Xn.T          # correlations
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    # numerical clean-up: enforce symmetry
    D = 0.5 * (D + D.T)
    return D


def _rdm_correlation_loop(X: np.ndarray) -> np.ndarray:  # reference impl
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = _pearson_row(X[i], X[j])
            D[i, j] = D[j, i] = 1.0 - r
    return D


def rdm_euclidean(X: np.ndarray, squared: bool = False) -> np.ndarray:
    """Euclidean (or squared-Euclidean) RDM."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    G = X @ X.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2.0 * G
    sq = np.maximum(sq, 0.0)
    if squared:
        return sq
    return np.sqrt(sq)


# ---------------------------------------------------------------------------
# RDM comparison
# ---------------------------------------------------------------------------
def upper_tri(mat: np.ndarray) -> np.ndarray:
    """Return the strict upper-triangular entries of a square matrix as a vector."""
    n = mat.shape[0]
    i, j = np.triu_indices(n, k=1)
    return mat[i, j]


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> float:
    """Compare two RDMs by rank or linear correlation of their upper triangles.

    Parameters
    ----------
    method : {'spearman', 'pearson', 'kendall'}
        Kriegeskorte 2008 recommends Spearman because RDMs from different
        modalities are not expected to have a linear relationship, only a
        monotonic one.
    """
    a = upper_tri(rdm_a)
    b = upper_tri(rdm_b)
    if method == "pearson":
        return _pearson_row(a, b)
    if method == "spearman":
        return _pearson_row(_rankdata(a), _rankdata(b))
    if method == "kendall":
        # Kendall's tau-a -- explicit O(n^2) implementation for clarity.
        n = len(a)
        num = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                num += s
        return num / (n * (n - 1) / 2)
    raise ValueError(f"unknown method {method!r}")


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie rank, matching scipy.stats.rankdata(method='average').

    Written explicitly so it is obvious what Spearman is doing.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    ranks[order] = np.arange(1, len(x) + 1)
    # handle ties: average the ranks of equal values
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Utility for the experiments
# ---------------------------------------------------------------------------
def check_rdm(D: np.ndarray) -> None:
    """Sanity-check that ``D`` looks like an RDM. Raises on failure."""
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"RDM must be square, got {D.shape}")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("RDM is not symmetric")
    if not np.allclose(np.diag(D), 0.0, atol=1e-8):
        raise ValueError("RDM has nonzero diagonal")
