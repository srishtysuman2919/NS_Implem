#!/usr/bin/env python
"""
Step 3 -- GO / NO-GO gate: is MD-network alignment separable from LANG-network
alignment in the model's representations?

This directly operationalises the objection "how do you misalign one region
while preserving the other?". If everything the LM can predict about MD is
*also* what it predicts about LANG, an adversarial MD head will necessarily
drag LANG alignment down and the region-specific experiment collapses into
the whole-brain result of Merlin & Toneva (2026).

Analyses (all leave-one-story-out, ridge, pure numpy):
  A. Baseline encoding: per-fROI r for LANG and MD, raw and noise-ceiling
     normalised.  (Is MD even predictable during passive listening?)
  B. Predicted-timecourse overlap: corr(mean yhat_LANG, mean yhat_MD) vs.
     corr(mean y_LANG, mean y_MD).  High overlap = shared signal.
  C. Variance partitioning for MD targets:
        R2[features], R2[LANG-BOLD], R2[features + LANG-BOLD]
     -> shared = R2_f + R2_L - R2_fL ; unique_features = R2_fL - R2_L
  D. Residual predictability: regress actual LANG BOLD out of MD BOLD, then
     predict the MD residual from features.  THIS IS THE KEY NUMBER.
     r_resid ~ 0  => MD-predictable info is entirely shared with LANG => NO-GO
     r_resid >> 0 => there is an MD-specific component to target        => GO
  E. Weight-subspace overlap: principal angles between the ridge weight
     subspaces for LANG vs MD (cos of first angles near 1 = same directions).
  F. Permutation null for D (story-wise circular shift of MD residual).

Usage
-----
  python 03_gate_analysis.py --prepared data/prepared --features features/gpt2_base/layer12.npy \
         --lags 1 2 3 4 --out results/gate_gpt2_L12.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (PreparedDataset, RidgeCV, make_lagged, pearson_cols, r2_cols,  # noqa: E402
                    story_cv_predict, zscore)


def principal_angles(A: np.ndarray, B: np.ndarray, k: int = 5):
    """cosines of principal angles between column spaces of A (d,a) and B (d,b)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s[:k], 0, 1)


def residualize(Y: np.ndarray, Z: np.ndarray, story_ids: np.ndarray) -> np.ndarray:
    """Return Y - proj_Z(Y) with the projection fit by story-CV (no leakage)."""
    Zl = make_lagged(Z, story_ids, lags=(0, 1, 2))  # allow small temporal slop
    Yhat = story_cv_predict(Zl, Y, story_ids, alphas=(0.1, 1, 10, 100))
    return Y - Yhat


def run(prepared: str, features: str, lags, out: str, n_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    ds = PreparedDataset.load(prepared)
    F = np.load(features)
    assert F.shape[0] == ds.Y.shape[0], f"features T={F.shape[0]} != Y T={ds.Y.shape[0]}"
    F = zscore(F)
    X = make_lagged(F, ds.story_ids, lags=lags) if lags else F
    Y = zscore(ds.Y)
    iL, iM = ds.idx("LANG"), ds.idx("MD")
    YL, YM = Y[:, iL], Y[:, iM]
    res = {"n_samples": int(Y.shape[0]), "n_lang": len(iL), "n_md": len(iM),
           "features": features, "lags": list(lags)}

    # ---- A. baseline encoding ----------------------------------------------
    Yhat = story_cv_predict(X, Y, ds.story_ids)
    r = pearson_cols(Y, Yhat)
    nc = ds.noise_ceiling
    res["A_encoding"] = {
        "r_lang_per_froi": dict(zip([ds.froi_names[i] for i in iL], np.round(r[iL], 4).tolist())),
        "r_md_per_froi": dict(zip([ds.froi_names[i] for i in iM], np.round(r[iM], 4).tolist())),
        "r_lang_mean": float(r[iL].mean()), "r_md_mean": float(r[iM].mean()),
        "nc_lang_mean": float(np.nanmean(nc[iL])), "nc_md_mean": float(np.nanmean(nc[iM])),
        "r_over_nc_lang": float(np.nanmean(r[iL] / np.where(nc[iL] > 0.05, nc[iL], np.nan))),
        "r_over_nc_md": float(np.nanmean(r[iM] / np.where(nc[iM] > 0.05, nc[iM], np.nan))),
    }

    # ---- B. predicted timecourse overlap -----------------------------------
    res["B_overlap"] = {
        "corr_actual_meanLANG_meanMD": float(np.corrcoef(YL.mean(1), YM.mean(1))[0, 1]),
        "corr_pred_meanLANG_meanMD": float(np.corrcoef(Yhat[:, iL].mean(1), Yhat[:, iM].mean(1))[0, 1]),
    }

    # ---- C. variance partitioning for MD -----------------------------------
    YL_lag = make_lagged(YL, ds.story_ids, lags=(0, 1, 2))
    r2_f = r2_cols(YM, story_cv_predict(X, YM, ds.story_ids))
    r2_L = r2_cols(YM, story_cv_predict(YL_lag, YM, ds.story_ids, alphas=(0.1, 1, 10, 100)))
    r2_fL = r2_cols(YM, story_cv_predict(np.hstack([X, YL_lag]), YM, ds.story_ids))
    shared = r2_f + r2_L - r2_fL
    unique_f = r2_fL - r2_L
    res["C_variance_partition_MD"] = {
        "R2_features_mean": float(r2_f.mean()), "R2_langBOLD_mean": float(r2_L.mean()),
        "R2_both_mean": float(r2_fL.mean()),
        "shared_mean": float(shared.mean()), "unique_features_mean": float(unique_f.mean()),
        "unique_fraction_of_R2_features": float(np.clip(unique_f.mean() / max(r2_f.mean(), 1e-8), 0, 1)),
    }

    # ---- D. residual predictability (KEY) ----------------------------------
    YM_res = residualize(YM, YL, ds.story_ids)
    r_res = pearson_cols(YM_res, story_cv_predict(X, YM_res, ds.story_ids))
    res["D_residual_MD"] = {
        "r_md_residual_per_froi": dict(zip([ds.froi_names[i] for i in iM], np.round(r_res, 4).tolist())),
        "r_md_residual_mean": float(r_res.mean()),
        "r_md_raw_mean": float(r[iM].mean()),
        "retained_fraction": float(r_res.mean() / max(r[iM].mean(), 1e-8)),
    }
    # symmetric check: LANG residualised on MD
    YL_res = residualize(YL, YM, ds.story_ids)
    r_resL = pearson_cols(YL_res, story_cv_predict(X, YL_res, ds.story_ids))
    res["D_residual_LANG"] = {"r_lang_residual_mean": float(r_resL.mean()),
                              "retained_fraction": float(r_resL.mean() / max(r[iL].mean(), 1e-8))}

    # ---- E. weight subspace overlap ----------------------------------------
    mL = RidgeCV().fit(X, YL, groups=ds.story_ids)
    mM = RidgeCV().fit(X, YM, groups=ds.story_ids)
    cosines = principal_angles(mL.coef_, mM.coef_, k=min(5, len(iL), len(iM)))
    res["E_weight_subspace"] = {"cos_principal_angles": np.round(cosines, 4).tolist(),
                                "mean_cos": float(cosines.mean())}

    # ---- F. permutation null for D ----------------------------------------
    null = []
    for _ in range(n_perm):
        Yp = np.zeros_like(YM_res)
        for s in np.unique(ds.story_ids):
            ix = np.where(ds.story_ids == s)[0]
            k = rng.integers(10, max(11, len(ix) - 10))
            Yp[ix] = np.roll(YM_res[ix], k, axis=0)
        null.append(pearson_cols(Yp, story_cv_predict(X, Yp, ds.story_ids)).mean())
    null = np.array(null)
    res["F_permutation"] = {"null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
                            "observed": float(r_res.mean()),
                            "p_value": float((np.sum(null >= r_res.mean()) + 1) / (n_perm + 1))}

    # ---- verdict -----------------------------------------------------------
    verdict = []
    if res["A_encoding"]["r_md_mean"] < 0.05:
        verdict.append("MD barely predictable from LM features during passive listening -> weak signal to misalign.")
    if res["F_permutation"]["p_value"] > 0.05:
        verdict.append("MD residual (after removing LANG) is NOT significantly predictable -> MD info is shared with LANG -> NO-GO for clean region-specific misalignment; consider reframing as 'shared vs unique alignment' study.")
    elif res["D_residual_MD"]["retained_fraction"] < 0.3:
        verdict.append("MD has a small but significant unique component (<30% of raw) -> region-specific misalignment is possible but effects will be subtle; use orthogonalised loss (train on MD residual) and many seeds.")
    else:
        verdict.append("MD has a substantial unique component -> GO: region-specific misalignment is well-posed.")
    res["verdict"] = verdict

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--features", required=True, help="path to layer{L}.npy from step 2")
    ap.add_argument("--lags", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--out", default="results/gate.json")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.prepared, a.features, tuple(a.lags), a.out, a.n_perm, a.seed)
