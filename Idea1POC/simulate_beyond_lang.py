#!/usr/bin/env python
"""Simulation with known ground truth for "does GPT-2 predict MD beyond LANG?".

null:   MD's text-driven signal is entirely inherited from LANG's true signal.
unique: MD additionally has its own feature-driven component.

A trustworthy statistic stays near zero under null and rises under unique. Three
statistics are compared, all scored on held-out stories:
  residual   - current Stage A method: GPT-2 -> (MD minus cross-half LANG prediction), r
  varpart    - joint [GPT-2 + cross-half LANG] vs LANG-only, delta held-out R^2 (and delta r)
  pred_lang  - GPT-2 -> MD versus GPT-2-predicted LANG -> MD, delta r
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

FEATURE_LAGS = (1, 2, 3, 4)
LANG_LAGS = (0, 1, 2)


def smooth(rng, n, k, phi):
    return lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal((n, k)), axis=0)


def zs(a):
    a = a - a.mean(axis=0)
    sd = a.std(axis=0)
    return a / np.where(sd > 1e-12, sd, 1.0)


def lag(a, lags):
    blocks = []
    for k in lags:
        block = np.roll(a, k, axis=0)
        block[:k] = 0.0
        blocks.append(block)
    return np.hstack(blocks)


def blend(parts):
    """Sum of z-scored parts weighted so each contributes its variance share."""
    return zs(sum(np.sqrt(share) * zs(part) for share, part in parts if share > 0))


class Grams:
    """Per-story cross-products, so ridge fits on any subset of stories are cheap sums."""

    def __init__(self, xs, ys):
        self.xx = np.stack([x.T @ x for x in xs])
        self.xy = np.stack([x.T @ y for x, y in zip(xs, ys)])
        self.yy = np.array([float((y ** 2).sum()) for y in ys])

    def fit(self, stories, penalty):
        gram = self.xx[stories].sum(axis=0)
        gram[np.diag_indices_from(gram)] += penalty
        return np.linalg.solve(gram, self.xy[stories].sum(axis=0))

    def tuned_fit(self, stories, penalties, folds=3):
        """Pick the penalty with the lowest held-out squared error over story folds, then refit."""
        groups = np.array_split(np.asarray(stories), folds)
        errors = []
        for penalty in penalties:
            error = 0.0
            for held in groups:
                W = self.fit(np.setdiff1d(stories, held), penalty)
                xx, xy = self.xx[held].sum(axis=0), self.xy[held].sum(axis=0)
                error += self.yy[held].sum() - 2 * np.sum(W * xy) + np.sum(W * (xx @ W))
            errors.append(error)
        return self.fit(stories, penalties[int(np.argmin(errors))])


def make_story(rng, cfg, world, unique_share, lang_rel):
    T, P, V = cfg.trs, cfg.lang_parcels, cfg.md_vertices
    X = zs(lag(zs(smooth(rng, T, cfg.features, 0.8)), FEATURE_LAGS))
    true_L = blend([(cfg.lang_text_share, X @ world["W_L"]),
                    (1 - cfg.lang_text_share, smooth(rng, T, P, 0.8))])
    true_M = blend([(cfg.md_inherited_share, lag(true_L, LANG_LAGS) @ world["A"]),
                    (unique_share, X @ world["W_M"]),
                    (1 - cfg.md_inherited_share - unique_share, smooth(rng, T, V, 0.8))])
    halves_L, halves_M = [], []
    for _ in range(2):
        # same-subject noise (motion, physiology) hits both networks within a half
        shared = smooth(rng, T, 1, 0.5)
        noise_L = blend([(cfg.shared_noise, shared @ np.ones((1, P))), (1 - cfg.shared_noise, smooth(rng, T, P, 0.3))])
        noise_M = blend([(cfg.shared_noise, shared @ np.ones((1, V))), (1 - cfg.shared_noise, smooth(rng, T, V, 0.3))])
        halves_L.append(blend([(lang_rel, true_L), (1 - lang_rel, noise_L)]))
        halves_M.append(blend([(cfg.md_half_rel, true_M), (1 - cfg.md_half_rel, noise_M)]))
    return {"X": X, "L": halves_L, "M": halves_M,
            "L_full": zs(halves_L[0] + halves_L[1]), "M_full": zs(halves_M[0] + halves_M[1])}


def score_r(records):
    """Mean over vertices of r between concatenated, per-story-standardized actual and predicted."""
    actual = np.vstack([zs(a) for a, _ in records])
    predicted = np.vstack([zs(p) for _, p in records])
    return float(np.mean((actual * predicted).mean(axis=0)))


def score_r2(records):
    actual = np.vstack([a for a, _ in records])
    predicted = np.vstack([p for _, p in records])
    centered = actual - actual.mean(axis=0)
    return float(np.mean(1 - ((actual - predicted) ** 2).sum(axis=0) / (centered ** 2).sum(axis=0)))


def run_replicate(rng, cfg, unique_share, lang_rel):
    n_feat = cfg.features * len(FEATURE_LAGS)
    world = {"W_L": rng.standard_normal((n_feat, cfg.lang_parcels)),
             "W_M": rng.standard_normal((n_feat, cfg.md_vertices)),
             "A": rng.standard_normal((cfg.lang_parcels * len(LANG_LAGS), cfg.md_vertices))}
    stories = [make_story(rng, cfg, world, unique_share, lang_rel) for _ in range(cfg.stories)]
    X = [s["X"] for s in stories]
    L_lag = [[lag(s["L"][d], LANG_LAGS) for s in stories] for d in (0, 1)]
    x_to_md = Grams(X, [s["M_full"] for s in stories])
    x_to_lang = Grams(X, [s["L_full"] for s in stories])
    lang_to_md = [Grams(L_lag[d], [s["M"][1 - d] for s in stories]) for d in (0, 1)]
    joint = [Grams([np.hstack([x, l]) for x, l in zip(X, L_lag[d])], [s["M"][1 - d] for s in stories]) for d in (0, 1)]
    n_lang = L_lag[0][0].shape[1]
    # 192 lagged feature columns need far heavier shrinkage than ~78 LANG columns (checked on a pilot story split)
    joint_penalties = [np.r_[np.full(n_feat, a_x), np.full(n_lang, a_l)]
                       for a_x in (1e2, 1e3, 1e4, 1e5, 1e6) for a_l in (1.0, 1e2, 1e3, 1e4, 1e5)]
    lang_penalties = [1.0, 1e2, 1e3, 1e4, 1e5]

    records = {name: [] for name in ("lang_raw", "md_raw", "residual", "joint", "lang_only", "pred_lang")}
    everyone = np.arange(len(stories))
    for test in everyone:
        train = everyone[everyone != test]

        def residual(story, fit_on):
            parts = [stories[story]["M"][1 - d] - L_lag[d][story] @ lang_to_md[d].fit(fit_on, cfg.residual_alpha)
                     for d in (0, 1)]
            return zs(parts[0] + parts[1])

        res_train = {s: residual(s, train[train != s]) for s in train}
        gram = x_to_md.xx[train].sum(axis=0)
        gram[np.diag_indices_from(gram)] += cfg.encoding_alpha
        W_res = np.linalg.solve(gram, sum(X[s].T @ res_train[s] for s in train))
        records["residual"].append((residual(test, train), X[test] @ W_res))

        records["md_raw"].append((stories[test]["M_full"], X[test] @ x_to_md.fit(train, cfg.encoding_alpha)))
        records["lang_raw"].append((stories[test]["L_full"], X[test] @ x_to_lang.fit(train, cfg.encoding_alpha)))

        # each direction is scored on the MD half whose noise is independent of its LANG predictor
        for d in (0, 1):
            target = stories[test]["M"][1 - d]
            records["joint"].append((target, np.hstack([X[test], L_lag[d][test]]) @ joint[d].tuned_fit(train, joint_penalties)))
            records["lang_only"].append((target, L_lag[d][test] @ lang_to_md[d].tuned_fit(train, lang_penalties)))

        # predicted LANG has no measurement noise; training-story predictions are cross-fitted
        pred_L = {s: lag(X[s] @ x_to_lang.fit(train[train != s], cfg.encoding_alpha), LANG_LAGS) for s in train}
        pred_L[test] = lag(X[test] @ x_to_lang.fit(train, cfg.encoding_alpha), LANG_LAGS)
        pred_gram = Grams([pred_L[s] for s in train], [stories[s]["M_full"] for s in train])
        W_pred = pred_gram.fit(np.arange(len(train)), cfg.residual_alpha)
        records["pred_lang"].append((stories[test]["M_full"], pred_L[test] @ W_pred))

    md_raw = score_r(records["md_raw"])
    return {"lang_raw_r": score_r(records["lang_raw"]), "md_raw_r": md_raw,
            "residual_r": score_r(records["residual"]),
            "varpart_dR2": score_r2(records["joint"]) - score_r2(records["lang_only"]),
            "varpart_dr": score_r(records["joint"]) - score_r(records["lang_only"]),
            "pred_lang_dr": md_raw - score_r(records["pred_lang"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stories", type=int, default=15)
    parser.add_argument("--trs", type=int, default=450)
    parser.add_argument("--features", type=int, default=48, help="latent feature dims before lags 1-4")
    parser.add_argument("--lang-parcels", type=int, default=26)
    parser.add_argument("--md-vertices", type=int, default=40)
    parser.add_argument("--lang-text-share", type=float, default=0.15, help="share of true LANG signal driven by features")
    parser.add_argument("--md-inherited-share", type=float, default=0.15, help="share of true MD signal inherited from LANG")
    parser.add_argument("--unique-share", type=float, default=0.02, help="feature-driven MD share in the unique scenario")
    parser.add_argument("--md-half-rel", type=float, default=0.26, help="signal variance share of an MD half-group mean")
    parser.add_argument("--lang-half-rel", type=float, nargs="+", default=[0.2, 0.44, 0.8])
    parser.add_argument("--shared-noise", type=float, default=0.3)
    parser.add_argument("--encoding-alpha", type=float, default=100.0)
    parser.add_argument("--residual-alpha", type=float, default=1.0)
    parser.add_argument("--out", default="results/simulate_beyond_lang.json")
    cfg = parser.parse_args()

    stats = ("lang_raw_r", "md_raw_r", "residual_r", "varpart_dR2", "varpart_dr", "pred_lang_dr")
    rng = np.random.default_rng(cfg.seed)
    results = []
    started = time.time()
    for lang_rel in cfg.lang_half_rel:
        cell = {}
        for scenario, share in (("null", 0.0), ("unique", cfg.unique_share)):
            reps = [run_replicate(rng, cfg, share, lang_rel) for _ in range(cfg.reps)]
            cell[scenario] = {k: [r[k] for r in reps] for k in stats}
        for k in stats:
            null, uniq = np.array(cell["null"][k]), np.array(cell["unique"][k])
            results.append({"lang_half_rel": lang_rel, "statistic": k,
                            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                            "unique_mean": float(uniq.mean()), "unique_sd": float(uniq.std(ddof=1)),
                            "null_bias_t": float(null.mean() / (null.std(ddof=1) / np.sqrt(len(null)))),
                            "separation": float((uniq.mean() - null.mean()) / null.std(ddof=1))})
        print(f"lang_half_rel={lang_rel} done ({time.time() - started:.0f}s)", flush=True)

    print(f"\n{'LANG rel':>8} {'statistic':>13} {'null mean':>10} {'null sd':>8} {'bias t':>7} "
          f"{'unique mean':>11} {'separation':>10}")
    for row in results:
        print(f"{row['lang_half_rel']:8.2f} {row['statistic']:>13} {row['null_mean']:10.4f} {row['null_sd']:8.4f} "
              f"{row['null_bias_t']:7.1f} {row['unique_mean']:11.4f} {row['separation']:10.1f}")
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text(json.dumps({"config": vars(cfg), "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
