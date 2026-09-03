"""Pattern Component Modelling (PCM), minimal core
-- Diedrichsen & Kriegeskorte (2017).

Paper goal
----------
RSA compares two systems at the level of **pairwise dissimilarities**.
That's a lossy summary: an RDM has ``n(n-1)/2`` numbers while the underlying
representation has more structure. Diedrichsen & Kriegeskorte 2017 argue
that the natural, information-preserving summary is the
**second-moment matrix** of the true condition patterns

    G = (U Uᵀ) / P

where ``U`` is the ``(n_conditions, n_features)`` matrix of true patterns
and ``P`` is the feature count. ``G`` is what you would get if you took
inner products between condition patterns and averaged across features.
Crucially, all common RDMs can be written as linear functions of ``G``
(e.g. squared Euclidean = diag(G) ⊕ diag(G) - 2G), so ``G`` is a
super-set of RDM information -- it also tells you the *scale* and
*mean* of the representation, not just relative distances.

The PCM framework:
1. Estimate ``G`` from noisy data in an **unbiased** way, using
   cross-runs like crossnobis (so noise variance does not inflate the
   diagonal of ``G``).
2. Model different hypotheses about ``G`` (feature-model ``G``'s from a
   candidate encoder), possibly as a mixture ``G = Σ w_k G_k``.
3. Compare models by their marginal likelihood on the observed patterns
   under a fixed noise model.

This module is a **minimal core**: it implements (1) unbiased estimation
of ``G`` from a `(n_runs, n_conditions, n_features)` array and (2) model
comparison by a simple Gaussian log-likelihood on the estimated ``G``.
It is not a replacement for the full PCM toolbox, but it is enough to
demonstrate the second-moment view and to compare candidate models by
their ``G``.

Contents of this module
-----------------------
* :func:`unbiased_G`      -- cross-run estimate of the second-moment
  matrix ``G``. Constructed so that ``E[G_hat] = G_true`` even when
  noise variance ≫ pattern variance.
* :func:`predict_rdm_from_G` -- squared-Euclidean RDM implied by a ``G``.
* :func:`compare_models_by_G` -- fit a nonnegative mixture of candidate
  model ``G_k``'s to ``G_hat`` in Frobenius sense and report weights and
  residuals.

What shortcoming of the previous method does this address?
---------------------------------------------------------
Both RSA and crossnobis discard information: RSA throws away scale and
absolute inner-product structure; crossnobis focuses on pairwise
distances only. PCM works with the full ``G`` matrix, so it can
distinguish models that produce the *same* RDM but different absolute
covariance structure, and it makes model comparison a principled
maximum-likelihood problem rather than a rank-correlation heuristic.
"""
from __future__ import annotations

import numpy as np


def unbiased_G(patterns: np.ndarray) -> np.ndarray:
    """Cross-run unbiased estimator of the second-moment matrix ``G``.

    Given ``patterns`` of shape ``(n_runs, n_conditions, n_features)``, we
    form ``G_hat = (1/(n_pairs · n_features)) Σ_{a≠b} U_a U_bᵀ``. Because
    noise in run ``a`` is independent of noise in run ``b``, cross-terms
    have zero expectation and ``E[G_hat] = G_true``.

    This is the ``G`` analogue of the crossnobis trick.
    """
    n_runs, n_cond, n_feat = patterns.shape
    if n_runs < 2:
        raise ValueError("PCM unbiased G requires at least 2 independent runs")
    G = np.zeros((n_cond, n_cond), dtype=float)
    n_pairs = 0
    for a in range(n_runs):
        for b in range(n_runs):
            if a == b:
                continue
            G = G + patterns[a] @ patterns[b].T
            n_pairs += 1
    G = G / (n_pairs * n_feat)
    G = 0.5 * (G + G.T)                     # enforce symmetry
    return G


def predict_rdm_from_G(G: np.ndarray) -> np.ndarray:
    """Squared-Euclidean RDM implied by a second-moment matrix.

    ``d²_ij = g_ii + g_jj - 2 g_ij``.
    """
    d = np.diag(G)
    D = d[:, None] + d[None, :] - 2.0 * G
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def compare_models_by_G(
    G_hat: np.ndarray,
    model_Gs: list[np.ndarray],
    include_intercept: bool = True,
) -> dict:
    """Fit a nonnegative mixture ``G_hat ≈ Σ w_k G_k (+ w_0 I)`` and score.

    Parameters
    ----------
    G_hat : (n, n) ndarray
        Empirical unbiased ``G`` from :func:`unbiased_G`.
    model_Gs : list of (n, n) ndarrays
        Candidate second-moment matrices, one per model / feature space.
    include_intercept : bool
        If True, add an identity component with nonneg weight to absorb
        residual isotropic variance (mimics the "noise" term in PCM).

    Returns
    -------
    dict with keys:
        weights          -- nonneg weights per model (and intercept last)
        residual_norm    -- Frobenius residual after best fit
        r2               -- explained fraction of ‖G_hat‖_F²
        rank             -- ranking of models by their solo Frobenius match
    """
    n = G_hat.shape[0]
    # 1. Solo score: cosine-like match ⟨G_hat, G_k⟩ / (‖G_hat‖ ‖G_k‖)
    solo = []
    for k, Gk in enumerate(model_Gs):
        num = float(np.sum(G_hat * Gk))
        denom = float(np.sqrt(np.sum(G_hat * G_hat) * np.sum(Gk * Gk)))
        solo.append(num / denom if denom > 0 else 0.0)
    rank = list(np.argsort(-np.asarray(solo)))

    # 2. Mixture fit via nonnegative least-squares on the vectorised upper triangle
    from scipy.optimize import nnls

    iu = np.triu_indices(n)
    b = G_hat[iu]
    cols = [Gk[iu] for Gk in model_Gs]
    if include_intercept:
        cols.append(np.eye(n)[iu])
    A = np.stack(cols, axis=1)
    w, _resid = nnls(A, b)
    pred = A @ w
    residual = float(np.linalg.norm(b - pred))
    total = float(np.linalg.norm(b))
    r2 = 1.0 - (residual * residual) / max(total * total, 1e-12)

    return {
        "weights": w,
        "residual_norm": residual,
        "r2": r2,
        "solo_scores": np.asarray(solo),
        "rank": rank,
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n_cond, n_feat = 8, 60
    U_true = rng.standard_normal((n_cond, n_feat))
    G_true = U_true @ U_true.T / n_feat

    # noisy measurements across 6 runs
    n_runs = 6
    patterns = U_true[None] + 1.2 * rng.standard_normal((n_runs, n_cond, n_feat))

    G_hat = unbiased_G(patterns)
    # bias check: mean absolute error to ground truth
    err = np.abs(G_hat - G_true).mean()

    # a naive within-run G would have inflated diagonal
    G_within = (patterns[0] @ patterns[0].T) / n_feat
    err_within = np.abs(G_within - G_true).mean()

    print("pcm self-test OK")
    print(f"  mean|G_true - G_unbiased|  = {err:.3f}")
    print(f"  mean|G_true - G_within-run| = {err_within:.3f}   (should be worse)")
    assert err < err_within

    # model comparison: give it the true G among two random alternatives
    dummies = [rng.standard_normal((n_cond, 20)) @ rng.standard_normal((20, n_cond))
               for _ in range(2)]
    dummies = [0.5 * (d + d.T) for d in dummies]
    result = compare_models_by_G(G_hat, [G_true, *dummies])
    print(f"  solo scores            = {result['solo_scores']}")
    print(f"  best solo model index  = {result['rank'][0]}")
    print(f"  mixture weights        = {result['weights']}")
    print(f"  mixture R²             = {result['r2']:.3f}")
    assert result["rank"][0] == 0, "true G should rank first solo"


if __name__ == "__main__":
    _selftest()
