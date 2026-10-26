"""
Shared utilities for the region-specific brain (mis)alignment POC.

Only numpy/pandas are required for everything in this file, so the gate
analysis (03_gate_analysis.py) can run on a CPU-only box without torch.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants: fROI naming (Fedorenko-lab conventions, Shain et al. 2020/2022)
# --------------------------------------------------------------------------- #
# 6 LH language fROIs (sentences > nonwords) and 10 bilateral MD fROIs
# (nonwords > sentences / hard > easy spatial WM).  Names in the released
# CSVs may carry prefixes such as "LANG", "MD", "LH"/"RH"; we match loosely.
LANG_FROI_KEYS = ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp", "AngG"]
MD_FROI_KEYS = ["PostPar", "MidPar", "AntPar", "PrecG", "SFG", "MFGorb",
                "MFG", "IFGop", "ACC", "preSMA", "Insula"]


def infer_network(froi_name: str) -> str:
    """Map an fROI label to 'LANG' or 'MD'.

    Priority: explicit prefix in the name -> keyword lookup. MFG appears in
    both networks, so the explicit prefix is required to disambiguate it; if
    absent we raise so the user supplies --froi_map.
    """
    n = froi_name.upper()
    if n.startswith("LANG") or "_LANG" in n or "LANG_" in n:
        return "LANG"
    if n.startswith("MD") or "_MD" in n or "MD_" in n:
        return "MD"
    # keyword fallback (MFG is ambiguous)
    if any(k.upper() in n for k in ["IFGORB", "ANTTEMP", "POSTTEMP", "ANGG"]):
        return "LANG"
    if any(k.upper() in n for k in ["POSTPAR", "MIDPAR", "ANTPAR", "PRECG", "SFG",
                                     "MFGORB", "IFGOP", "ACC", "PRESMA", "INSULA"]):
        return "MD"
    raise ValueError(
        f"Cannot infer network for fROI '{froi_name}'. Pass an explicit "
        f"fROI->network JSON via --froi_map."
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def pearson_cols(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-wise Pearson r between two (T, C) arrays -> (C,)"""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + eps
    return num / den


def r2_cols(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0) + eps
    return 1.0 - ss_res / ss_tot


class RidgeCV:
    """Multi-output ridge with per-target alpha chosen by inner CV.

    Pure numpy (SVD-based) so it runs anywhere. X: (T, d), Y: (T, C).
    """

    def __init__(self, alphas: Sequence[float] = (1, 10, 100, 1e3, 1e4, 1e5),
                 n_inner: int = 5, fit_intercept: bool = True):
        self.alphas = np.asarray(alphas, dtype=float)
        self.n_inner = n_inner
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None

    @staticmethod
    def _solve_all_alphas(X, Y, alphas):
        # returns dict alpha -> W (d, C) using one SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        UtY = U.T @ Y
        out = {}
        for a in alphas:
            d = s / (s ** 2 + a)
            out[a] = Vt.T @ (d[:, None] * UtY)
        return out

    def fit(self, X: np.ndarray, Y: np.ndarray, groups: Optional[np.ndarray] = None):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        xm = X.mean(0) if self.fit_intercept else np.zeros(X.shape[1])
        ym = Y.mean(0) if self.fit_intercept else np.zeros(Y.shape[1])
        Xc, Yc = X - xm, Y - ym

        # inner CV to pick alpha per target
        T = X.shape[0]
        if groups is None:
            folds = np.array_split(np.arange(T), self.n_inner)
        else:
            ug = np.unique(groups)
            gfolds = np.array_split(ug, min(self.n_inner, len(ug)))
            folds = [np.where(np.isin(groups, g))[0] for g in gfolds]
        scores = np.zeros((len(self.alphas), Y.shape[1]))
        for te in folds:
            tr = np.setdiff1d(np.arange(T), te)
            Ws = self._solve_all_alphas(Xc[tr], Yc[tr], self.alphas)
            for i, a in enumerate(self.alphas):
                pred = Xc[te] @ Ws[a]
                scores[i] += pearson_cols(Yc[te], pred)
        best = scores.argmax(0)
        self.alpha_ = self.alphas[best]
        Ws = self._solve_all_alphas(Xc, Yc, self.alphas)
        W = np.zeros((X.shape[1], Y.shape[1]))
        for c in range(Y.shape[1]):
            W[:, c] = Ws[self.alpha_[c]][:, c]
        self.coef_ = W
        self.intercept_ = ym - xm @ W
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def story_cv_predict(X: np.ndarray, Y: np.ndarray, story_ids: np.ndarray,
                     alphas=(1, 10, 100, 1e3, 1e4, 1e5)) -> np.ndarray:
    """Leave-one-story-out cross-validated predictions (T, C)."""
    Yhat = np.zeros_like(Y, dtype=float)
    for s in np.unique(story_ids):
        te = story_ids == s
        tr = ~te
        m = RidgeCV(alphas=alphas).fit(X[tr], Y[tr], groups=story_ids[tr])
        Yhat[te] = m.predict(X[te])
    return Yhat


def make_lagged(X: np.ndarray, story_ids: np.ndarray, lags: Sequence[int] = (1, 2, 3, 4)) -> np.ndarray:
    """FIR-style lagged concatenation within each story (no leakage across stories).

    Row t gets [X[t-l] for l in lags]; rows without history are zero-padded.
    lags are in TRs. With TR=2s, lags 1..4 cover the 2-8 s HRF peak window.
    """
    T, d = X.shape
    out = np.zeros((T, d * len(lags)), dtype=X.dtype)
    for s in np.unique(story_ids):
        idx = np.where(story_ids == s)[0]
        Xs = X[idx]
        for j, l in enumerate(lags):
            shifted = np.zeros_like(Xs)
            if l == 0:
                shifted = Xs.copy()
            elif l < len(idx):
                shifted[l:] = Xs[:-l]
            out[idx, j * d:(j + 1) * d] = shifted
    return out


def spearman_brown(r: np.ndarray) -> np.ndarray:
    return 2 * r / (1 + r)


# --------------------------------------------------------------------------- #
# Dataset container (produced by 01_prepare_data.py, consumed downstream)
# --------------------------------------------------------------------------- #
@dataclass
class PreparedDataset:
    """TR-level aligned dataset.

    text_windows : list[str]  text of the words presented in the `window_trs`
                   TRs ending `delay_trs` before the target TR (for the live
                   brain-prediction heads during fine-tuning).
    story_ids    : (T,) int
    tr_index     : (T,) int   TR index within story
    Y            : (T, C) float  fROI responses (subject-mean or one subject)
    froi_names   : list[str] length C
    network      : list[str] length C  ('LANG' | 'MD')
    noise_ceiling: (C,) split-half, Spearman-Brown corrected (NaN if 1 subject)
    """
    text_windows: List[str]
    story_ids: np.ndarray
    tr_index: np.ndarray
    Y: np.ndarray
    froi_names: List[str]
    network: List[str]
    noise_ceiling: np.ndarray
    tr_seconds: float
    window_trs: int
    delay_trs: int
    meta: Dict = field(default_factory=dict)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "Y.npy"), self.Y)
        np.save(os.path.join(path, "story_ids.npy"), self.story_ids)
        np.save(os.path.join(path, "tr_index.npy"), self.tr_index)
        np.save(os.path.join(path, "noise_ceiling.npy"), self.noise_ceiling)
        with open(os.path.join(path, "text_windows.json"), "w") as f:
            json.dump(self.text_windows, f)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "froi_names": self.froi_names,
                "network": self.network,
                "tr_seconds": self.tr_seconds,
                "window_trs": self.window_trs,
                "delay_trs": self.delay_trs,
                "meta": self.meta,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PreparedDataset":
        with open(os.path.join(path, "meta.json")) as f:
            m = json.load(f)
        with open(os.path.join(path, "text_windows.json")) as f:
            tw = json.load(f)
        return cls(
            text_windows=tw,
            story_ids=np.load(os.path.join(path, "story_ids.npy")),
            tr_index=np.load(os.path.join(path, "tr_index.npy")),
            Y=np.load(os.path.join(path, "Y.npy")),
            froi_names=m["froi_names"],
            network=m["network"],
            noise_ceiling=np.load(os.path.join(path, "noise_ceiling.npy")),
            tr_seconds=m["tr_seconds"],
            window_trs=m["window_trs"],
            delay_trs=m["delay_trs"],
            meta=m.get("meta", {}),
        )

    def idx(self, net: str) -> np.ndarray:
        return np.array([i for i, n in enumerate(self.network) if n == net])


# --------------------------------------------------------------------------- #
# Gradient reversal (torch) -- imported lazily so numpy-only scripts work
# --------------------------------------------------------------------------- #
def get_grl():
    import torch

    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    def grad_reverse(x, lam: float = 1.0):
        return _GradReverse.apply(x, lam)

    return grad_reverse


def neg_pearson_loss(pred, target, squared: bool = False, eps: float = 1e-8):
    """Torch loss: mean over channels of -(r) or -(r^2) (anchor paper uses r^2
    for the misaligned model, r for the brain-tuned model)."""
    import torch
    p = pred - pred.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    r = (p * t).sum(0) / (torch.sqrt((p * p).sum(0) * (t * t).sum(0)) + eps)
    if squared:
        return -(r ** 2).mean()
    return -r.mean()
