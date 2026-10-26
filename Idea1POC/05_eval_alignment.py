#!/usr/bin/env python
"""
Step 5 -- Did the intervention do what it was supposed to do?

For each condition's features (extracted with 02_extract_features.py using
--adapter), re-fit an *independent* ridge encoder (as the anchor paper does;
never trust the training heads) and report per-network alignment on the
held-out story, relative to the base model and the preserve_control.

Success criterion for Idea 1 (misalign_md_preserve_lang):
    r_MD  drops significantly vs preserve_control
    r_LANG does NOT drop (within CI of preserve_control)
    LM perplexity unchanged (checked in 04/06)

Usage
-----
  python 05_eval_alignment.py --prepared data/prepared --heldout_story 0 \
      --features base=features/gpt2_base/layer12.npy \
                 ctrl=features/gpt2_ctrl_s0/layer12.npy \
                 misMD=features/gpt2_misMD_s0/layer12.npy \
                 misLANG=features/gpt2_misLANG_s0/layer12.npy \
      --out results/align_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PreparedDataset, RidgeCV, make_lagged, pearson_cols, zscore  # noqa: E402


def bootstrap_ci(r_a, r_b, n=2000, seed=0):
    """bootstrap CI over channels of mean(r_a - r_b)"""
    rng = np.random.default_rng(seed)
    d = r_a - r_b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def evaluate(ds, F, heldout, lags):
    F = zscore(F)
    X = make_lagged(F, ds.story_ids, lags=lags) if lags else F
    Y = zscore(ds.Y)
    tr, te = ds.story_ids != heldout, ds.story_ids == heldout
    m = RidgeCV().fit(X[tr], Y[tr], groups=ds.story_ids[tr])
    r = pearson_cols(Y[te], m.predict(X[te]))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--heldout_story", type=int, required=True)
    ap.add_argument("--features", nargs="+", required=True, help="name=path/layerL.npy ...")
    ap.add_argument("--lags", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--control", default="ctrl", help="name of the preserve_control entry")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = PreparedDataset.load(a.prepared)
    iL, iM = ds.idx("LANG"), ds.idx("MD")
    R = {}
    for item in a.features:
        name, path = item.split("=", 1)
        R[name] = evaluate(ds, np.load(path), a.heldout_story, tuple(a.lags))
    out = {"heldout_story": a.heldout_story, "per_condition": {}}
    for name, r in R.items():
        out["per_condition"][name] = {
            "r_lang_mean": float(r[iL].mean()), "r_md_mean": float(r[iM].mean()),
            "r_lang_per_froi": dict(zip([ds.froi_names[i] for i in iL], np.round(r[iL], 4).tolist())),
            "r_md_per_froi": dict(zip([ds.froi_names[i] for i in iM], np.round(r[iM], 4).tolist())),
        }
    if a.control in R:
        rc = R[a.control]
        out["vs_control"] = {}
        for name, r in R.items():
            if name == a.control:
                continue
            out["vs_control"][name] = {
                "delta_lang": float((r[iL] - rc[iL]).mean()), "ci_lang": bootstrap_ci(r[iL], rc[iL]),
                "delta_md": float((r[iM] - rc[iM]).mean()), "ci_md": bootstrap_ci(r[iM], rc[iM]),
            }
        # selectivity index for Idea 1: how much more MD dropped than LANG
        if "misMD" in out["vs_control"]:
            v = out["vs_control"]["misMD"]
            out["selectivity_misMD"] = float(v["delta_lang"] - v["delta_md"])  # >0 means MD fell more than LANG
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
