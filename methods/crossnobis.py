"""Crossnobis / cross-validated Mahalanobis distance
-- Walther et al. (2016); Diedrichsen et al. (2021).

Paper goal
----------
Classic RSA distances (correlation, Euclidean) are **positively biased by
measurement noise**. If you measure the same condition twice under
independent noise, the empirical distance between the two estimates is
nonzero, and this bias grows with noise. That contaminates every downstream
RDM statistic and, in particular, makes it impossible to test whether two
conditions are *actually* represented identically.

Walther et al. 2016 (and the earlier Ledoit-Wolf-style precursors in
Diedrichsen 2011) introduced the **cross-validated Mahalanobis distance**,
a.k.a. **crossnobis**. It fixes the noise bias in two ways:

1. **Noise normalisation** -- divide the patterns by the (regularised)
   residual covariance ``Σ`` so distances are measured in units of noise
   standard deviation. Equivalently, whiten patterns to have identity noise.
2. **Cross-validation** -- estimate the difference vector `(u_i - u_j)` from
   independent measurement partitions A and B, and use the *inner product*

       d²_ij = ⟨u_i^A − u_j^A , u_i^B − u_j^B⟩ / n_features

   as the distance estimate. Because the noise in partition A is independent
   of the noise in partition B, the cross terms have zero expectation, so
   ``d²`` is an **unbiased** estimator of the true squared Mahalanobis
   distance. It can be negative, and that's fine: 0 is the true chance level.

Diedrichsen et al. 2021 gives the modern reference implementation and
extends this to fixed / multi-run designs.

Contents of this module
-----------------------
* :func:`estimate_noise_covariance` -- residual covariance from a
  ``(n_runs, n_conditions, n_features)`` array, shrunk by Ledoit-Wolf.
* :func:`whiten` -- multiply patterns by ``Σ^{-1/2}``.
* :func:`crossnobis_rdm` -- unbiased squared crossnobis distance matrix from
  a ``(n_runs, n_conditions, n_features)`` array.

What shortcoming of the previous method does this address?
---------------------------------------------------------
Classic RSA with correlation or Euclidean distance overestimates all
dissimilarities and, worse, overestimates them **more when noise is
larger**. That confounds SNR with representational similarity: a noisier
brain region will look like it has "more separated" representations even
if the true patterns are identical. Crossnobis returns an unbiased
estimator so that a zero result really does mean "no representational
separation", and RDMs measured under different noise regimes are directly
comparable.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Noise covariance
# ---------------------------------------------------------------------------
def estimate_noise_covariance(
    patterns: np.ndarray,
    shrinkage: float | str = "ledoit_wolf",
) -> np.ndarray:
    """Estimate the per-feature noise covariance ``Σ``.

    Parameters
    ----------
    patterns : (n_runs, n_conditions, n_features) ndarray
        Response amplitude estimates. The "noise" is the *variability across
        runs*: for each condition c and feature f, the residuals are
        ``patterns[:, c, f] - mean_over_runs``.
    shrinkage : float in [0, 1] or 'ledoit_wolf'
        Convex combination weight for ``λ * diag(Σ_emp) + (1-λ) * Σ_emp``.
        With ``'ledoit_wolf'`` we pick λ analytically (Ledoit & Wolf 2004),
        which is what modern crossnobis pipelines default to.

    Returns
    -------
    (n_features, n_features) ndarray -- symmetric positive-definite estimate.
    """
    n_runs, n_cond, n_feat = patterns.shape
    # residuals: subtract per-condition mean across runs
    mu = patterns.mean(axis=0, keepdims=True)          # (1, n_cond, n_feat)
    resid = (patterns - mu).reshape(n_runs * n_cond, n_feat)
    # empirical covariance
    n_obs = resid.shape[0]
    Sigma = resid.T @ resid / max(n_obs - 1, 1)

    # shrinkage target: diagonal of Sigma
    target = np.diag(np.diag(Sigma))

    if shrinkage == "ledoit_wolf":
        # Ledoit-Wolf shrinkage intensity, following eq. 14 in LW 2004.
        # var(s_ij) sum divided by squared distance to target.
        X = resid                                          # (n_obs, n_feat)
        Xc = X - X.mean(axis=0, keepdims=True)
        # element-wise: s_ij = (1/n) sum_k Xc[k,i] Xc[k,j]
        n = Xc.shape[0]
        # asymptotic variance of the sample covariance entries
        var_num = 0.0
        for k in range(n):
            outer = np.outer(Xc[k], Xc[k])
            var_num += ((outer - Sigma) ** 2).sum()
        var_num /= n * n
        # squared distance from target (off-diagonal energy)
        diff = Sigma - target
        denom = (diff * diff).sum()
        if denom == 0:
            lam = 0.0
        else:
            lam = max(0.0, min(1.0, var_num / denom))
    else:
        lam = float(shrinkage)

    Sigma_hat = lam * target + (1.0 - lam) * Sigma
    return Sigma_hat


def whiten(X: np.ndarray, Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return ``X @ Σ^{-1/2}`` where ``Σ^{-1/2}`` is the symmetric root.

    Written out via eigendecomposition so it is clear what is happening.
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, eps, None)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return X @ inv_sqrt


# ---------------------------------------------------------------------------
# Crossnobis RDM
# ---------------------------------------------------------------------------
def crossnobis_rdm(
    patterns: np.ndarray,
    Sigma: np.ndarray | None = None,
    shrinkage: float | str = "ledoit_wolf",
) -> np.ndarray:
    """Unbiased squared crossnobis distance matrix.

    Parameters
    ----------
    patterns : (n_runs, n_conditions, n_features) ndarray
        At least 2 runs required. Rows within a run are the condition-wise
        response patterns (e.g. GLM betas from a single scanning run).
    Sigma : (n_features, n_features) ndarray or None
        Noise covariance. If None it is estimated from `patterns` via
        :func:`estimate_noise_covariance`.
    shrinkage : passed through to :func:`estimate_noise_covariance`.

    Returns
    -------
    (n_conditions, n_conditions) ndarray -- squared crossnobis distances.
    Diagonal is zero by construction (i vs. i has ``u_i - u_i = 0``).
    Off-diagonal entries can be *negative* in low-SNR settings; that's the
    intended feature of the unbiased estimator.
    """
    n_runs, n_cond, n_feat = patterns.shape
    if n_runs < 2:
        raise ValueError("crossnobis requires at least 2 independent runs")

    # 1. noise-whiten every run's patterns
    if Sigma is None:
        Sigma = estimate_noise_covariance(patterns, shrinkage=shrinkage)
    W = np.stack([whiten(patterns[r], Sigma) for r in range(n_runs)], axis=0)

    # 2. cross-validated inner product of pairwise contrast vectors
    D = np.zeros((n_cond, n_cond), dtype=float)
    n_pairs = 0
    # leave-one-run-out: for each pair (a, b) of distinct runs
    for a in range(n_runs):
        for b in range(n_runs):
            if a == b:
                continue
            # difference vectors from each run
            for i in range(n_cond):
                for j in range(i + 1, n_cond):
                    d_a = W[a, i] - W[a, j]
                    d_b = W[b, i] - W[b, j]
                    D[i, j] += (d_a @ d_b) / n_feat
        # end for b
    # end for a
    n_pairs = n_runs * (n_runs - 1)
    D = D / n_pairs
    # symmetric, zero diagonal
    D = D + D.T
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n_runs, n_cond, n_feat = 6, 8, 40
    # ground-truth patterns
    U = rng.standard_normal((n_cond, n_feat))
    # add heteroskedastic noise
    noise_scale = 1.5
    patterns = U[None, :, :] + noise_scale * rng.standard_normal(
        (n_runs, n_cond, n_feat)
    )

    D_true = ((U[:, None] - U[None, :]) ** 2).sum(axis=-1) / n_feat
    D_cn = crossnobis_rdm(patterns)

    # sanity: shapes, symmetry, zero diagonal
    assert D_cn.shape == (n_cond, n_cond)
    assert np.allclose(D_cn, D_cn.T)
    assert np.allclose(np.diag(D_cn), 0.0)

    # noise-biased Euclidean distance for comparison
    from .rdm_utils import rdm_euclidean

    mean_patterns = patterns.mean(axis=0)
    D_biased = rdm_euclidean(mean_patterns, squared=True) / n_feat

    # crossnobis should be closer (in mean) to the true squared distance than
    # the biased Euclidean
    err_cn = np.abs(D_cn - D_true).mean()
    err_bi = np.abs(D_biased - D_true).mean()
    print("crossnobis self-test OK")
    print(f"  mean|D_true - D_crossnobis| = {err_cn:.3f}")
    print(f"  mean|D_true - D_euclidean|  = {err_bi:.3f}")
    print(f"  crossnobis is less biased?  {err_cn < err_bi}")
    assert err_cn < err_bi, (
        "crossnobis did not beat naive Euclidean in this SNR regime -- "
        "should not happen with seed 0"
    )


if __name__ == "__main__":
    _selftest()
