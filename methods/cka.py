"""Centered Kernel Alignment (CKA) -- Kornblith, Norouzi, Lee & Hinton (2019).

Paper goal
----------
The authors wanted a similarity index between two neural-network
representations that is (a) invariant to orthogonal transforms and to
isotropic scaling but (b) *not* invariant to arbitrary invertible linear
transforms (unlike linear CCA / SVCCA, which are dominated by the
lowest-variance directions and so give unintuitive rankings).

**Linear CKA** between two ``(n, p)`` and ``(n, q)`` matrices X, Y is

    CKA(X, Y) = ‖YᵀX‖_F² / (‖XᵀX‖_F · ‖YᵀY‖_F)

Equivalently, it is the normalised Frobenius inner product of the two
**centered gram matrices** ``K = X_c X_cᵀ`` and ``L = Y_c Y_cᵀ``:

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))

with HSIC (Hilbert-Schmidt Independence Criterion) computed with a linear
kernel. Replacing the linear kernel with an RBF kernel gives **RBF-CKA**,
which is sensitive to nonlinear structure.

Contents of this module
-----------------------
* :func:`center_gram`     -- double-centre a gram matrix (equivalent to
  centring the features first and then computing the gram matrix).
* :func:`hsic`            -- biased HSIC estimator (the one used in Kornblith
  et al. 2019; there is also an unbiased variant).
* :func:`linear_cka`      -- linear CKA between two feature matrices; also
  provides a "feature-space" fast path that avoids materialising the
  ``n × n`` gram matrix when ``p, q ≪ n``.
* :func:`rbf_cka`         -- RBF-kernel CKA.

What shortcoming of the previous method does this address?
---------------------------------------------------------
Classic RSA with correlation distance is invariant to arbitrary rotations
of the feature space, which is often exactly what you want -- but it also
throws away information about the *variance* along each direction, so two
representations that differ only in which directions carry the most
signal look identical. Linear CCA, at the other extreme, is invariant to
*every* invertible linear transform, so it is dominated by low-variance
directions and gives nonintuitive similarities.

CKA sits in between: it is invariant to orthogonal transforms and
isotropic scaling (so it is not fooled by rotations or unit changes), but
it *does* respect the covariance structure (so directions carrying more
variance count more). It also does not require ``p = q`` and remains
well-defined for very wide layers.
"""
from __future__ import annotations

import numpy as np


def center_gram(K: np.ndarray) -> np.ndarray:
    """Double-centre a gram matrix: ``H K H`` where ``H = I - 11ᵀ/n``.

    This is equivalent to computing ``K'`` from the mean-centred data.
    Written explicitly instead of ``H @ K @ H`` because that would materialise
    two ``(n, n)`` matrices for no reason.
    """
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC estimator (Kornblith et al. 2019, eq. 3).

    HSIC(K, L) = (1 / (n-1)²) · trace(K_c L_c) with K_c, L_c centred.
    """
    n = K.shape[0]
    Kc = center_gram(K)
    Lc = center_gram(L)
    return float(np.sum(Kc * Lc) / (n - 1) ** 2)


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices.

    Parameters
    ----------
    X : (n, p) ndarray
    Y : (n, q) ndarray
        Same number of examples ``n``; feature counts can differ.

    Returns
    -------
    float in [0, 1]

    Notes
    -----
    Two paths:

    * **Gram path** -- always correct, ``O(n²·max(p, q))``:
      compute centred gram matrices ``K = X X^T``, ``L = Y Y^T`` and
      normalised HSIC. Used when ``n ≤ 2 * max(p, q)``.
    * **Feature path** -- faster when ``n ≫ p, q``: use the algebraic
      identity ``‖Yᵀ X‖_F² = trace((X X^T)(Y Y^T))`` after centring the
      *features* directly. Same answer.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA requires equal n; got {X.shape[0]} vs {Y.shape[0]}"
        )
    n, p = X.shape
    q = Y.shape[1]

    # Always centre features (cheap and stable)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    if n <= 2 * max(p, q):                            # gram path
        K = Xc @ Xc.T
        L = Yc @ Yc.T
        num = np.sum(K * L)                           # trace(K L)
        denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    else:                                             # feature path
        YX = Yc.T @ Xc                                # (q, p)
        num = float(np.sum(YX * YX))                  # ||Y^T X||_F^2
        XX = Xc.T @ Xc
        YY = Yc.T @ Yc
        denom = np.sqrt(np.sum(XX * XX) * np.sum(YY * YY))
    if denom == 0.0:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# RBF CKA
# ---------------------------------------------------------------------------
def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, written out for clarity."""
    G = X @ X.T
    d = np.diag(G)
    sq = d[:, None] + d[None, :] - 2.0 * G
    return np.maximum(sq, 0.0)


def rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gaussian RBF kernel matrix.

    If ``sigma is None`` we use the **median heuristic**: σ = √(median(||x_i - x_j||²) / 2).
    """
    sq = _pairwise_sq_dists(X)
    if sigma is None:
        # off-diagonal only, to avoid the pile of zeros on the diagonal
        n = sq.shape[0]
        iu = np.triu_indices(n, k=1)
        med = np.median(sq[iu])
        sigma = float(np.sqrt(max(med, 1e-12) / 2.0))
    return np.exp(-sq / (2.0 * sigma * sigma))


def rbf_cka(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> float:
    """RBF-kernel CKA between two feature matrices."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("RBF CKA requires equal n")
    K = rbf_kernel(X, sigma=sigma_x)
    L = rbf_kernel(Y, sigma=sigma_y)
    num = hsic(K, L)
    denom = np.sqrt(hsic(K, K) * hsic(L, L))
    if denom == 0.0:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n, p, q = 100, 30, 40
    X = rng.standard_normal((n, p))
    Y = X @ rng.standard_normal((p, q)) + 0.1 * rng.standard_normal((n, q))

    cka_self = linear_cka(X, X)
    cka_xy = linear_cka(X, Y)
    cka_rand = linear_cka(X, rng.standard_normal((n, q)))

    assert abs(cka_self - 1.0) < 1e-8, f"CKA(X, X) should be 1, got {cka_self}"
    assert cka_xy > cka_rand, "linearly-related pair should score higher"
    print("cka self-test OK")
    print(f"  linear CKA(X, X)                = {cka_self:.4f}")
    print(f"  linear CKA(X, linear-of-X + ε)  = {cka_xy:.4f}")
    print(f"  linear CKA(X, random)           = {cka_rand:.4f}")

    rbf_xy = rbf_cka(X, Y)
    rbf_rand = rbf_cka(X, rng.standard_normal((n, q)))
    print(f"  RBF    CKA(X, linear-of-X + ε)  = {rbf_xy:.4f}")
    print(f"  RBF    CKA(X, random)           = {rbf_rand:.4f}")


if __name__ == "__main__":
    _selftest()
