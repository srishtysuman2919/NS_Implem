#!/usr/bin/env python
"""
Create a small synthetic dataset in the *CDR-style* layout that
01_prepare_data.py expects, plus matching fake "LM features", so the whole
numpy pipeline (01 -> 03 -> 05) can be dry-run without downloading 6 GB or
having torch installed.

Two regimes are generated so you can see what GO vs NO-GO looks like:
  --regime shared  : MD signal is a noisy copy of LANG signal  (expect NO-GO)
  --regime unique  : MD has its own feature-driven component    (expect GO)

Usage
-----
  python make_synthetic.py --out data/synth_unique --regime unique
  python 01_prepare_data.py --data_dir data/synth_unique/osf --out data/synth_unique/prepared
  python 03_gate_analysis.py --prepared data/synth_unique/prepared \
         --features data/synth_unique/features/layer12.npy --out results/gate_synth_unique.json --n_perm 50
"""
import argparse
import os

import numpy as np
import pandas as pd

LANG = ["LANG_IFGorb", "LANG_IFG", "LANG_MFG", "LANG_AntTemp", "LANG_PostTemp", "LANG_AngG"]
MD = [f"MD_{h}{r}" for h in ("LH", "RH") for r in
      ("PostPar", "MidPar", "AntPar", "PrecG", "SFG", "MFG", "MFGorb", "IFGop", "ACC", "Insula")]
WORDS = "the of and to in a is that it was for on with as he she they said then while because river mill old".split()


def main(out, regime, n_stories=6, n_subj=12, trs_per_story=150, d=64, seed=0):
    rng = np.random.default_rng(seed)
    tr = 2.0
    os.makedirs(f"{out}/osf", exist_ok=True)
    os.makedirs(f"{out}/features", exist_ok=True)
    stim_rows, resp_rows, feats = [], [], []
    # ONE fixed feature->BOLD mapping for the whole "brain" (constant across stories)
    wL = rng.normal(size=(d, len(LANG))) / np.sqrt(d)
    wM = rng.normal(size=(d, len(MD))) / np.sqrt(d)
    for s in range(n_stories):
        # words at ~2.5 words/sec
        n_words = int(trs_per_story * tr * 2.5)
        times = np.sort(rng.uniform(0, trs_per_story * tr, n_words))
        for t in times:
            stim_rows.append({"word": rng.choice(WORDS), "time": round(float(t), 3), "docid": f"story{s}"})
        # latent "LM feature" per TR window and the signals it drives
        F = rng.normal(size=(trs_per_story, d))
        sigL = F @ wL
        if regime == "shared":
            sigM = sigL.mean(1, keepdims=True) * np.ones((1, len(MD))) + 0.3 * rng.normal(size=(trs_per_story, len(MD)))
        else:
            sigM = F @ wM + 0.5 * sigL.mean(1, keepdims=True)
        # hemodynamic delay: shift by 2 TRs
        sigL = np.roll(sigL, 2, 0)
        sigM = np.roll(sigM, 2, 0)
        feats.append(F)
        for subj in range(n_subj):
            noiseL = rng.normal(size=sigL.shape) * 1.0
            noiseM = rng.normal(size=sigM.shape) * 1.5
            for k in range(trs_per_story):
                for j, f in enumerate(LANG):
                    resp_rows.append({"subject": f"s{subj:02d}", "docid": f"story{s}", "fROI": f,
                                      "time": k * tr, "BOLD": sigL[k, j] + noiseL[k, j]})
                for j, f in enumerate(MD):
                    resp_rows.append({"subject": f"s{subj:02d}", "docid": f"story{s}", "fROI": f,
                                      "time": k * tr, "BOLD": sigM[k, j] + noiseM[k, j]})
    pd.DataFrame(stim_rows).to_csv(f"{out}/osf/stimulus.csv", index=False)
    pd.DataFrame(resp_rows).to_csv(f"{out}/osf/response.csv", index=False)
    # features aligned to the rows 01_prepare_data will keep: it drops TRs
    # with no text in the window; we re-derive that filter here identically.
    stim = pd.DataFrame(stim_rows)
    keep = []
    for s in range(n_stories):
        w = stim[stim.docid == f"story{s}"]
        for k in range(trs_per_story):
            t_end = k * tr - 2 * tr
            t_start = t_end - 5 * tr
            if ((w.time > t_start) & (w.time <= t_end)).sum() >= 1:
                keep.append(feats[s][k] + 0.3 * rng.normal(size=d))  # LM features = latent + noise
    np.save(f"{out}/features/layer12.npy", np.vstack(keep))
    print(f"wrote {out}: {len(stim_rows)} words, {len(resp_rows)} BOLD rows, features {np.vstack(keep).shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--regime", choices=["shared", "unique"], default="unique")
    a = ap.parse_args()
    main(a.out, a.regime)
