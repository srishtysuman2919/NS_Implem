"""Linear ridge-regression encoding models -- Brain-Score (Schrimpf et al. 2018).

Paper goal
----------
Similarity-based methods (RSA, CKA, shape distance) compare *summaries* of
two representations. An entirely different strategy is to ask: **can you
linearly predict brain responses from model features?** If yes, and if the
prediction generalises to held-out stimuli, the model contains all the
information the brain uses (up to a linear read-out). This is the
Brain-Score protocol popularised by Schrimpf et al. 2018 / 2020 and
underlies most modern "deep-nets-as-models-of-cortex" work.

Concretely, for each voxel v:

    y_v ≈ X w_v + b_v

where ``X`` is a ``(n_stimuli, n_features)`` matrix of model activations
and ``y_v`` is the brain response of voxel v across stimuli. Fit ``w_v`` by
**ridge regression** (Tikhonov regularisation is needed because the number
of model features can be huge relative to the number of stimuli). Score by
Pearson correlation between predicted and held-out ``y_v``.

Contents of this module
-----------------------
* :func:`ridge_fit`                -- closed-form ridge weights via SVD.
* :func:`ridge_predict`
* :func:`ridge_score`              -- Pearson r between predicted and
  measured responses, per output column (voxel).
* :func:`ridge_cv_score`           -- k-fold cross-validated Pearson r,
  optionally with per-voxel ``α`` selected on an inner loop.

What shortcoming of the previous method does this address?
---------------------------------------------------------
RSA / CKA / shape distance are **structural** comparisons: they look at
the geometry of the representation, not at whether one representation can
*predict* the other in a stimulus-specific way. Two models that produce
the same RDM can still differ in which model features carry the
information -- one might be a much better mechanistic model of neural
tuning. Ridge encoding directly asks the mechanistic question: given
your features, can I linearly read out the brain response? This is
complementary to RSA, and Schrimpf et al. 2018 showed that ridge
encoding scores correlate strongly with core-object-recognition
performance across a large set of vision models, providing a
behaviour-linked, per-voxel benchmark.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Closed-form ridge via SVD (numerically stable, handles all α at once)
# ---------------------------------------------------------------------------
def ridge_fit(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``argmin_W ‖Y - X W‖² + α ‖W‖²`` in closed form.

    Uses the SVD ``X = U S V^T``. Then
    ``W = V · diag(s / (s² + α)) · U^T Y``.

    Parameters
    ----------
    X : (n_samples, n_features) ndarray
    Y : (n_samples, n_outputs) ndarray
    alpha : float
        Ridge penalty.
    center : bool
        If True, mean-centre X and Y and return an offset ``b`` so that the
        model is ``ŷ = X W + b``.

    Returns
    -------
    W : (n_features, n_outputs)
    b : (n_outputs,)
    """
    if center:
        X_mean = X.mean(axis=0, keepdims=True)
        Y_mean = Y.mean(axis=0, keepdims=True)
        Xc = X - X_mean
        Yc = Y - Y_mean
    else:
        Xc, Yc = X, Y
        X_mean = np.zeros((1, X.shape[1]))
        Y_mean = np.zeros((1, Y.shape[1]))

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # d = s / (s^2 + α)   (broadcasted over outputs)
    d = s / (s * s + alpha)
    W = Vt.T @ (d[:, None] * (U.T @ Yc))          # (n_features, n_outputs)
    b = (Y_mean - X_mean @ W).ravel()
    return W, b


def ridge_predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b[None, :]


def ridge_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between predicted and measured responses.

    This is the Brain-Score voxel-wise / neuron-wise metric.
    """
    y_pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    y_true_c = y_true - y_true.mean(axis=0, keepdims=True)
    num = (y_pred_c * y_true_c).sum(axis=0)
    denom = np.sqrt(
        (y_pred_c * y_pred_c).sum(axis=0) * (y_true_c * y_true_c).sum(axis=0)
    )
    denom = np.where(denom == 0, 1.0, denom)
    return num / denom


# ---------------------------------------------------------------------------
# k-fold cross-validated ridge score
# ---------------------------------------------------------------------------
def _kfold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def ridge_cv_score(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: float | list[float] = 1.0,
    k: int = 5,
    rng: np.random.Generator | None = None,
    select_alpha_per_output: bool = False,
) -> dict:
    """k-fold cross-validated ridge encoding score.

    Parameters
    ----------
    alphas : float or list of float
        Either a single α to use for every voxel, or a grid of α values to
        select from (inner CV on the training folds).
    select_alpha_per_output : bool
        If True and ``alphas`` is a list, choose a different α per output
        voxel by picking the one with the highest training-fold mean score
        for that voxel. Otherwise the single α with the best mean score
        across all voxels is used.

    Returns
    -------
    dict with keys:
        'scores'        : (n_outputs,) mean held-out Pearson r
        'per_fold'      : (k, n_outputs) array of per-fold scores
        'alpha_used'    : scalar or (n_outputs,) array
    """
    rng = rng or np.random.default_rng(0)
    if np.isscalar(alphas):
        alphas = [float(alphas)]
    else:
        alphas = list(alphas)
    n, n_out = Y.shape[0], Y.shape[1]
    folds = _kfold_indices(n, k, rng)

    per_fold = np.zeros((k, n_out))
    if len(alphas) == 1:
        alpha_used = np.full(n_out, alphas[0])
    else:
        alpha_used = np.zeros(n_out)

    # For α selection we do a plain "outer CV, single α" for the common case,
    # or inner-CV α selection per voxel for the fancy case.
    for fi, test_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Xtr, Ytr = X[train_mask], Y[train_mask]
        Xte, Yte = X[test_idx], Y[test_idx]

        if len(alphas) == 1:
            W, b = ridge_fit(Xtr, Ytr, alpha=alphas[0])
            r = ridge_score(ridge_predict(Xte, W, b), Yte)
            per_fold[fi] = r
        else:
            # inner CV on training fold -- 3 splits is plenty for a study repo
            inner_folds = _kfold_indices(len(Xtr), 3, rng)
            r_by_alpha = np.zeros((len(alphas), n_out))
            for a_i, a in enumerate(alphas):
                inner_scores = np.zeros((3, n_out))
                for gi, val_idx in enumerate(inner_folds):
                    m = np.ones(len(Xtr), dtype=bool)
                    m[val_idx] = False
                    W, b = ridge_fit(Xtr[m], Ytr[m], alpha=a)
                    inner_scores[gi] = ridge_score(
                        ridge_predict(Xtr[val_idx], W, b), Ytr[val_idx]
                    )
                r_by_alpha[a_i] = inner_scores.mean(axis=0)
            if select_alpha_per_output:
                best = r_by_alpha.argmax(axis=0)
                # fit one W per unique α, evaluate per voxel
                r_test = np.zeros(n_out)
                for a_i, a in enumerate(alphas):
                    cols = np.where(best == a_i)[0]
                    if len(cols) == 0:
                        continue
                    W, b = ridge_fit(Xtr, Ytr[:, cols], alpha=a)
                    r_test[cols] = ridge_score(
                        ridge_predict(Xte, W, b), Yte[:, cols]
                    )
                    alpha_used[cols] = a
                per_fold[fi] = r_test
            else:
                a_best = alphas[int(r_by_alpha.mean(axis=1).argmax())]
                W, b = ridge_fit(Xtr, Ytr, alpha=a_best)
                per_fold[fi] = ridge_score(ridge_predict(Xte, W, b), Yte)
                alpha_used[:] = a_best

    return {
        "scores": per_fold.mean(axis=0),
        "per_fold": per_fold,
        "alpha_used": alpha_used,
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n, p, q = 200, 50, 10
    X = rng.standard_normal((n, p))
    W_true = rng.standard_normal((p, q))
    Y = X @ W_true + 0.3 * rng.standard_normal((n, q))

    W, b = ridge_fit(X, Y, alpha=1.0)
    r = ridge_score(ridge_predict(X, W, b), Y)
    print("ridge_encoding self-test OK")
    print(f"  in-sample mean r    = {r.mean():.3f}")

    out = ridge_cv_score(X, Y, alphas=[0.1, 1.0, 10.0], k=5,
                        select_alpha_per_output=False)
    print(f"  5-fold CV mean r    = {out['scores'].mean():.3f}")
    print(f"  α used              = {out['alpha_used'][0]}")
    assert out["scores"].mean() > 0.7, "CV score too low for a linear task"


if __name__ == "__main__":
    _selftest()
