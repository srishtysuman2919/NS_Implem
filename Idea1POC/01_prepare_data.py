#!/usr/bin/env python
"""
Step 1 -- Build a TR-aligned dataset from the Shain et al. (2020/2022) Natural
Stories fMRI release (fROI-averaged BOLD, LANG + MD networks).

The OSF archives (osf.io/eyp8q, osf.io/ah429) are distributed in the CDR
("continuous-time deconvolutional regression") format: one or more *stimulus*
tables (one row per word: word, onset time, story id, ...) and one or more
*response* tables (one row per TR per fROI per subject: BOLD, time, fROI, ...).
Column names differ slightly between releases, so this script auto-detects
them and prints what it found. Run with --inspect first.

Usage
-----
  python 01_prepare_data.py --inspect  --data_dir data/osf_eyp8q
  python 01_prepare_data.py --data_dir data/osf_eyp8q --out data/prepared \
         --subject_mode mean --window_trs 5 --delay_trs 2

Outputs a PreparedDataset directory (see common.py) with
  Y (T, C)  fROI responses, text_windows (T,) strings, story ids, noise ceiling.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PreparedDataset, infer_network, pearson_cols, spearman_brown  # noqa: E402


# --------------------------------------------------------------------------- #
# Column auto-detection
# --------------------------------------------------------------------------- #
CANDS = {
    "word": ["word", "token", "Word"],
    "time": ["time", "onset", "Time", "onset_time"],
    "story": ["docid", "story", "item", "doc", "storyid", "Story"],
    "subject": ["subject", "participant", "subj", "Subject", "UID"],
    "froi": ["fROI", "froi", "roi", "ROI", "region"],
    "bold": ["BOLD", "bold", "y", "response", "signal"],
    "network": ["network", "Network", "net"],
    "tr": ["tr", "TR", "trnum", "sampleid"],
    "sentid": ["sentid", "sentence_id", "sentpos"],
}


def pick(cols: List[str], key: str) -> Optional[str]:
    for c in CANDS[key]:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in CANDS[key]:
        if c.lower() in low:
            return low[c.lower()]
    return None


def read_table(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    sep = "\t" if path.endswith((".tsv", ".txt")) else None
    if path.endswith(".csv"):
        sep = ","
    # CDR files are often whitespace-delimited .csv/.txt
    try:
        df = pd.read_csv(path, sep=sep, engine="python", nrows=nrows)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=r"\s+", engine="python", nrows=nrows)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", nrows=nrows)
    return df


def classify_tables(data_dir: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    files = []
    for ext in ("*.csv", "*.tsv", "*.txt"):
        files += glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
    stim, resp, schema = [], [], {}
    for f in sorted(files):
        try:
            head = read_table(f, nrows=50)
        except Exception as e:  # pragma: no cover
            print(f"  skip {f}: {e}")
            continue
        cols = list(head.columns)
        schema[f] = cols
        has_word = pick(cols, "word") is not None
        has_bold = pick(cols, "bold") is not None and pick(cols, "froi") is not None
        if has_bold:
            resp.append(f)
        elif has_word and pick(cols, "time") is not None:
            stim.append(f)
    combined_stim = [f for f in stim if os.path.basename(f).startswith("LANG_MD_")]
    combined_resp = [f for f in resp if os.path.basename(f).startswith("LANG_MD_")]
    if combined_stim and combined_resp:
        stim, resp = combined_stim, combined_resp
    return stim, resp, schema


def inspect(data_dir: str):
    stim, resp, schema = classify_tables(data_dir)
    print("\n=== STIMULUS (word-level) tables ===")
    for f in stim:
        print(f"  {f}\n     cols: {schema[f]}")
    print("\n=== RESPONSE (BOLD per fROI) tables ===")
    for f in resp:
        print(f"  {f}\n     cols: {schema[f]}")
        df = read_table(f, nrows=200000)
        fc = pick(list(df.columns), "froi")
        sc = pick(list(df.columns), "subject")
        tc = pick(list(df.columns), "time")
        print(f"     fROIs ({df[fc].nunique()}): {sorted(df[fc].unique().tolist())[:40]}")
        if sc:
            print(f"     subjects (first 200k rows): {df[sc].nunique()}")
        if tc:
            t = np.sort(df[tc].unique())
            if len(t) > 2:
                print(f"     median TR (s): {np.median(np.diff(t)):.3f}")
    other = [f for f in schema if f not in stim and f not in resp]
    if other:
        print("\n=== OTHER tables (ignored) ===")
        for f in other:
            print(f"  {f}: {schema[f][:12]}")


# --------------------------------------------------------------------------- #
# Core preparation
# --------------------------------------------------------------------------- #
def load_stimulus(stim_files: List[str]) -> pd.DataFrame:
    dfs = [read_table(f) for f in stim_files]
    df = pd.concat(dfs, ignore_index=True)
    cols = list(df.columns)
    w, t, s = pick(cols, "word"), pick(cols, "time"), pick(cols, "story")
    if s is None:
        raise ValueError(f"No story/doc id column in stimulus table; cols={cols}")
    out = pd.DataFrame({"word": df[w].astype(str), "time": df[t].astype(float), "story": df[s]})
    # CDR stimulus tables may repeat per subject; words are identical, so dedupe
    out = out.drop_duplicates(subset=["story", "time", "word"]).sort_values(["story", "time"])
    return out.reset_index(drop=True)


def load_response(resp_files: List[str], froi_map: Optional[Dict[str, str]]) -> pd.DataFrame:
    dfs = [read_table(f) for f in resp_files]
    df = pd.concat(dfs, ignore_index=True)
    cols = list(df.columns)
    fc, bc, tc, sc, stc, nc = (pick(cols, k) for k in ["froi", "bold", "time", "subject", "story", "network"])
    if stc is None:
        raise ValueError(f"Response table lacks a story/doc id column; cols={cols}")
    out = pd.DataFrame({
        "froi": df[fc].astype(str),
        "bold": df[bc].astype(float),
        "time": df[tc].astype(float),
        "subject": df[sc].astype(str) if sc else "S0",
        "story": df[stc],
    })
    if nc is not None:
        net = df[nc].astype(str).str.upper().str.replace("LANGUAGE", "LANG")
        net = net.replace({"0": "MD", "1": "LANG", "0.0": "MD", "1.0": "LANG"})
        out["network"] = net.values
        # make fROI names unique across networks (MFG exists in both)
        out["froi"] = out["network"] + "_" + out["froi"]
    else:
        if froi_map:
            out["network"] = out["froi"].map(froi_map)
            if out["network"].isna().any():
                missing = out.loc[out["network"].isna(), "froi"].unique()
                raise ValueError(f"froi_map missing entries for: {missing}")
        else:
            out["network"] = out["froi"].map(infer_network)
    return out


def build(data_dir: str, out_dir: str, subject_mode: str, window_trs: int, delay_trs: int,
          froi_map_path: Optional[str], subject: Optional[str], min_words: int):
    stim_files, resp_files, _ = classify_tables(data_dir)
    if not stim_files or not resp_files:
        raise SystemExit("Could not find both stimulus and response tables. Run --inspect.")
    froi_map = json.load(open(froi_map_path)) if froi_map_path else None
    stim = load_stimulus(stim_files)
    resp = load_response(resp_files, froi_map)

    # harmonise story ids as strings
    stim["story"] = stim["story"].astype(str)
    resp["story"] = resp["story"].astype(str)
    common_stories = sorted(set(stim.story) & set(resp.story))
    print(f"stories with both stimulus+response: {len(common_stories)} -> {common_stories}")
    stim = stim[stim.story.isin(common_stories)]
    resp = resp[resp.story.isin(common_stories)]

    # TR duration
    tr_s = float(np.median(np.diff(np.sort(resp.time.unique()))))
    print(f"estimated TR = {tr_s:.3f} s")

    froi_names = sorted(resp.froi.unique())
    network = [resp.loc[resp.froi == f, "network"].iloc[0] for f in froi_names]
    print(f"{len(froi_names)} fROIs: LANG={sum(n == 'LANG' for n in network)}, MD={sum(n == 'MD' for n in network)}")

    # response tensor: subject x story -> (nTR, C)
    subjects = sorted(resp.subject.unique())
    if subject_mode == "individual":
        if subject is None:
            raise SystemExit("--subject_mode individual requires --subject <id>")
        subjects_used = [subject]
    else:
        subjects_used = subjects
    print(f"subjects available: {len(subjects)}; using: {len(subjects_used)}")

    Y_rows, text_rows, story_rows, tr_rows = [], [], [], []
    half_a, half_b = [], []  # for split-half noise ceiling
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(subjects_used))
    A = set(np.array(subjects_used)[perm[: len(perm) // 2]])

    for story in common_stories:
        r = resp[resp.story == story]
        times = np.sort(r.time.unique())
        words = stim[stim.story == story]
        # pivot per subject: (nTR, C)
        mats = {}
        for sid in subjects_used:
            rs = r[r.subject == sid]
            if rs.empty:
                continue
            piv = rs.pivot_table(index="time", columns="froi", values="bold", aggfunc="mean")
            piv = piv.reindex(index=times, columns=froi_names)
            mats[sid] = piv.values
        if not mats:
            continue
        stack = np.stack(list(mats.values()))  # (S, nTR, C)
        # z-score each subject/channel within story before averaging
        stack = (stack - np.nanmean(stack, axis=1, keepdims=True)) / (np.nanstd(stack, axis=1, keepdims=True) + 1e-8)
        Ymean = np.nanmean(stack, axis=0)
        ids = list(mats.keys())
        if len(ids) > 1:
            a = [i for i, s in enumerate(ids) if s in A]
            b = [i for i, s in enumerate(ids) if s not in A]
            if a and b:
                half_a.append(np.nanmean(stack[a], 0))
                half_b.append(np.nanmean(stack[b], 0))

        # text windows: words with onset in (t_end - window, t_end], t_end = t - delay
        for k, t in enumerate(times):
            t_end = t - delay_trs * tr_s
            t_start = t_end - window_trs * tr_s
            w = words[(words.time > t_start) & (words.time <= t_end)]
            txt = " ".join(w.word.tolist())
            if len(w) < min_words:
                continue  # skip onset TRs with no text
            Y_rows.append(Ymean[k])
            text_rows.append(txt)
            story_rows.append(common_stories.index(story))
            tr_rows.append(k)

    Y = np.vstack(Y_rows)
    if half_a:
        ha, hb = np.vstack(half_a), np.vstack(half_b)
        ok = ~(np.isnan(ha).any(1) | np.isnan(hb).any(1))
        nc = spearman_brown(pearson_cols(ha[ok], hb[ok]))
    else:
        nc = np.full(Y.shape[1], np.nan)
    print("split-half noise ceiling (Spearman-Brown) per network:")
    for net in ("LANG", "MD"):
        ix = [i for i, n in enumerate(network) if n == net]
        print(f"   {net}: mean={np.nanmean(nc[ix]):.3f}  per-fROI={np.round(nc[ix], 3).tolist()}")

    ds = PreparedDataset(
        text_windows=text_rows, story_ids=np.array(story_rows), tr_index=np.array(tr_rows),
        Y=Y, froi_names=froi_names, network=network, noise_ceiling=nc, tr_seconds=tr_s,
        window_trs=window_trs, delay_trs=delay_trs,
        meta={"stories": common_stories, "n_subjects_used": len(subjects_used), "subject_mode": subject_mode},
    )
    ds.save(out_dir)
    print(f"saved {Y.shape[0]} TR samples x {Y.shape[1]} fROIs -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--subject_mode", choices=["mean", "individual"], default="mean")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--window_trs", type=int, default=5, help="text window length in TRs (anchor paper: 5)")
    ap.add_argument("--delay_trs", type=int, default=2, help="hemodynamic delay in TRs (2 TRs = 4 s)")
    ap.add_argument("--froi_map", default=None, help="JSON {froi_name: 'LANG'|'MD'} if auto-inference fails")
    ap.add_argument("--min_words", type=int, default=1)
    a = ap.parse_args()
    if a.inspect:
        inspect(a.data_dir)
    else:
        build(a.data_dir, a.out, a.subject_mode, a.window_trs, a.delay_trs, a.froi_map, a.subject, a.min_words)
