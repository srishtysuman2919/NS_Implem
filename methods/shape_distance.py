"""Generalized shape metric / Procrustes distance
-- Williams, Kunin, Dapello, Ganguli, Kolter & Tanaka (2021).

Paper goal
----------
CKA is a *similarity score*, not a proper metric: it does not satisfy the
triangle inequality, its value is not a distance, and averaging or
clustering representations under CKA is ill-defined. Williams et al. 2021
argue that if we want a **geometry** of neural representations (so we can do
clustering, averaging, statistical hypothesis testing on distances, etc.)
we need an *actual* distance function on an appropriate quotient space.

They construct one: view a representation as a point on the Riemannian
manifold of ``n × p`` matrices modulo an equivalence group ``G`` (e.g. all
orthogonal ``p × p`` rotations, if we want rotation invariance). Two
representations are then compared by finding the ``g ∈ G`` that best
aligns them (an **orthogonal Procrustes** problem) and reporting the
resulting **arc-length distance** on the shape manifold.

Two flavours in this module
---------------------------
* :func:`orthogonal_procrustes`         -- solve for the optimal rotation.
* :func:`procrustes_distance`           -- Euclidean distance after alignment.
* :func:`angular_shape_distance`        -- Williams et al.'s angular
  (arc-cosine) distance on the shape manifold: :math:`\\arccos(\\rho)` where
  ``ρ`` is the cosine of the two representations after mean-centring,
  Frobenius-normalising, and rotation-aligning. This is a proper metric.

What shortcoming of the previous method does this address?
---------------------------------------------------------
CKA gives you a number in ``[0, 1]``, but it does not tell you *how far
apart* two representations are in any geometrically consistent sense: you
cannot cluster them, you cannot average them, and you cannot use
distance-based hypothesis testing. Shape distance provides an actual
Riemannian metric on representations modulo rotation, so all of those
standard tools become available.
"""
from __future__ import annotations

import numpy as np


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve ``argmin_R ‖X - Y R‖_F`` over orthogonal ``R`` (``R^T R = I``).

    Classical closed form (Schönemann 1966): let ``M = Y^T X``, take its
    SVD ``M = U Σ V^T``. Then ``R* = U V^T``.

    Parameters
    ----------
    X, Y : (n, p) ndarrays with the same shape.

    Returns
    -------
    R : (p, p) ndarray -- the optimal rotation to apply to ``Y``.
    residual : float  -- ``‖X - Y R‖_F``.
    """
    if X.shape != Y.shape:
        raise ValueError(
            f"orthogonal_procrustes requires equal shapes, got {X.shape} vs {Y.shape}"
        )
    M = Y.T @ X
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    residual = float(np.linalg.norm(X - Y @ R, ord="fro"))
    return R, residual


def procrustes_distance(X: np.ndarray, Y: np.ndarray, center: bool = True) -> float:
    """Frobenius distance after optimal orthogonal alignment of ``Y`` onto ``X``.

    Not scale-invariant; call :func:`angular_shape_distance` for the scale-
    invariant, angular metric from Williams et al.
    """
    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
    _, residual = orthogonal_procrustes(X, Y)
    return residual


def _pad_features(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the narrower of the two matrices so they have the same width.

    Only used by :func:`procrustes_distance`, which needs an explicit rotation
    matrix. :func:`angular_shape_distance` avoids padding entirely via the
    nuclear-norm identity documented below, so it stays tractable for very
    wide feature matrices (e.g. flattened conv layers with p >> n).
    """
    p_x, p_y = X.shape[1], Y.shape[1]
    if p_x == p_y:
        return X, Y
    p = max(p_x, p_y)
    X2 = np.zeros((X.shape[0], p))
    Y2 = np.zeros((Y.shape[0], p))
    X2[:, :p_x] = X
    Y2[:, :p_y] = Y
    return X2, Y2


def angular_shape_distance(
    X: np.ndarray,
    Y: np.ndarray,
    center: bool = True,
) -> float:
    """Angular Procrustes distance -- Williams et al. 2021, eq. 4.

    Steps:
    1. Optionally mean-centre each column of X and Y.
    2. Frobenius-normalise both.
    3. Compute the maximum inner product over all orthogonal alignments
       using the identity

           max_{R orthogonal} <X, Y R>_F
               = ||X^T Y||_*                            (nuclear norm)
               = sum of singular values of X^T Y

       (This is von Neumann's trace inequality; the SVD ``M = U Σ V^T``
       of ``M = Y^T X`` gives the closed-form rotation ``R = U V^T``, and
       the resulting inner product equals ``sum(Σ)``.)
    4. Return ``arccos(nuclear_norm / (||X||_F ||Y||_F))``. Because X and Y
       are already Frobenius-normalised, the denominator is 1.

    Range is ``[0, π/2]`` for centred inputs and this is a proper metric
    on the quotient manifold, so clustering / averaging / bootstrapping
    representations is well-defined.

    Implementation note
    -------------------
    Instead of materialising ``X^T Y`` (which is ``p × q`` and blows up for
    wide model layers), we compute its singular values from compact SVDs of
    X and Y: writing ``X = U_x Σ_x V_x^T`` and ``Y = U_y Σ_y V_y^T`` with
    ``U_x, U_y`` of shape ``(n, r)``, the singular values of ``X^T Y``
    equal those of the small ``r_x × r_y`` matrix
    ``Σ_x U_x^T U_y Σ_y``, where ``r_x, r_y ≤ n``. This makes the metric
    tractable for arbitrarily wide feature spaces.
    """
    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
    nx = np.linalg.norm(X, ord="fro")
    ny = np.linalg.norm(Y, ord="fro")
    if nx == 0.0 or ny == 0.0:
        return float(np.pi / 2)
    X = X / nx
    Y = Y / ny

    # Compact SVDs -- U's are (n, r), s's are (r,).
    Ux, sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Y, full_matrices=False)
    # Small cross-matrix M = Σ_x U_x^T U_y Σ_y, at most (n × n).
    M = (sx[:, None] * (Ux.T @ Uy)) * sy[None, :]
    # Nuclear norm of X^T Y == sum of singular values of M.
    nuclear = float(np.linalg.svd(M, compute_uv=False).sum())
    nuclear = max(-1.0, min(1.0, nuclear))
    return float(np.arccos(nuclear))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n, p = 50, 20
    X = rng.standard_normal((n, p))

    # 1. rotated copy -> distance ≈ 0
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Y = X @ Q
    d_rot = angular_shape_distance(X, Y)
    assert d_rot < 1e-6, f"expected ~0 for pure rotation, got {d_rot}"

    # 2. rotated + scaled -> also ~0 (scale-invariant after normalisation)
    d_scaled = angular_shape_distance(X, 3.7 * Y)
    assert d_scaled < 1e-6, f"expected ~0 for rotation+scale, got {d_scaled}"

    # 3. random -> some positive distance
    d_rand = angular_shape_distance(X, rng.standard_normal((n, p)))
    assert d_rand > 0.5, f"expected >0.5 for random pair, got {d_rand}"

    # 4. triangle inequality (proper metric)
    A = rng.standard_normal((n, p))
    B = rng.standard_normal((n, p))
    C = rng.standard_normal((n, p))
    d_ab = angular_shape_distance(A, B)
    d_bc = angular_shape_distance(B, C)
    d_ac = angular_shape_distance(A, C)
    assert d_ac <= d_ab + d_bc + 1e-8, "triangle inequality violated"

    print("shape_distance self-test OK")
    print(f"  angular d(X, rotated X)       = {d_rot:.6f}")
    print(f"  angular d(X, rotated+scaled)  = {d_scaled:.6f}")
    print(f"  angular d(X, random)          = {d_rand:.4f}")
    print(f"  triangle: d(A,C)={d_ac:.3f} ≤ d(A,B)+d(B,C)={d_ab + d_bc:.3f}")


if __name__ == "__main__":
    _selftest()
