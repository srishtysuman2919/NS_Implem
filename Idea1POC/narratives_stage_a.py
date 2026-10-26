"""Vertex-wise Narratives Stage A; local inputs only, no synthetic fallback.

The real-data CLI requires a prepared manifest, verified features and a frozen
analysis specification. Tests call the array API with explicitly synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve


BASELINES = (
    "voiced_fraction", "word_rate", "word_length", "unigram_surprisal",
    "phoneme_or_syllable_rate", "pitch", "energy", "word_frequency", "sentence_boundary",
)


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)


def correlations(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Correlation requires matching time-by-target arrays")
    return np.mean(standardize(actual) * standardize(predicted), axis=0)


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Expected finite p values in [0,1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(
        (values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1])
    return adjusted


def pairwise_isc(subjects):
    values = np.asarray(subjects, dtype=float)
    if values.ndim != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("ISC needs finite subjects-by-time-by-vertex data and >=2 subjects")
    centered = values - values.mean(axis=1, keepdims=True)
    scales = centered.std(axis=1, keepdims=True)
    valid = (scales[:, 0] > 1e-6).all(axis=0)
    normalized = centered / np.where(scales > 1e-6, scales, 1.0)
    count = len(values)
    rho = ((normalized.sum(axis=0) ** 2).mean(axis=0) - count) / (count * (count - 1))
    return np.where(valid, rho, -1.0)


def effective_time(subjects):
    series = np.asarray(subjects).mean(axis=(0, 2))
    if series.std() <= 1e-8:
        return float(len(series))
    lag_one = float(correlations(series[:-1, None], series[1:, None])[0])
    return float(np.clip(len(series) * (1 - lag_one) / max(1 + lag_one, 1e-6), 1, len(series)))


@dataclass
class Story:
    name: str
    family: str
    subjects: np.ndarray
    gpt2: np.ndarray
    baseline: np.ndarray
    subject_ids: tuple[str, ...]
    parcel_ids: np.ndarray


def validate_arrays(stories, networks):
    if len({story.family for story in stories}) < 3:
        raise ValueError("At least three independent story families are needed for cross-fitting")
    if len({story.name for story in stories}) != len(stories):
        raise ValueError("Duplicate story names")
    width = len(networks)
    for story in stories:
        values = story.subjects
        if values.ndim != 3 or values.shape[2] != width or len(values) < 4:
            raise ValueError(f"{story.name}: require >=4 subjects and a shared vertex axis")
        if len(set(story.subject_ids)) != len(values) or len(story.subject_ids) != len(values):
            raise ValueError(f"{story.name}: duplicate/missing participant identity")
        if values.shape[1] <= 20 or story.parcel_ids.shape != (width,):
            raise ValueError(f"{story.name}: invalid time/parcel axes")
        for matrix in (values, story.gpt2, story.baseline):
            if not np.isfinite(matrix).all():
                raise ValueError(f"{story.name}: nonfinite data")
        for matrix in (story.gpt2, story.baseline):
            if matrix.ndim != 2 or len(matrix) != values.shape[1]:
                raise ValueError(f"{story.name}: feature time axis mismatch")
        if np.any(values.std(axis=1) <= 1e-6):
            raise ValueError(f"{story.name}: constant target")
    if not {"LANG", "MD"}.issubset(set(networks)):
        raise ValueError("Both LANG and MD vertices are required")
    if len({story.gpt2.shape[1] for story in stories}) != 1 or len({story.baseline.shape[1] for story in stories}) != 1:
        raise ValueError("Inconsistent feature widths")


def _stream(seed, *labels):
    text = ":".join(str(item) for item in (seed, *labels))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _shift_add(total, values, shift):
    """total += np.roll(values, shift, axis=0) without allocating."""
    if shift:
        total[shift:] += values[:-shift]
        total[:shift] += values[-shift:]
    else:
        total += values


def selection_shifts(story, n_null, seed):
    count, length = story.subjects.shape[:2]
    rng = np.random.default_rng(_stream(seed, "selection", story.name))
    return rng.integers(10, length - 9, size=(n_null, count))


def story_isc_nulls(story, shifts, workers=1):
    """Observed pairwise ISC and its subject-wise circular-shift null for one story."""
    values = np.asarray(story.subjects, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    del values
    scales = centered.std(axis=1, keepdims=True)
    valid = (scales[:, 0] > 1e-6).all(axis=0)
    normalized = centered / np.where(scales > 1e-6, scales, 1.0)
    del centered
    count, length, width = normalized.shape
    denominator = count * (count - 1)
    observed = (np.square(normalized.sum(axis=0, dtype=np.float64)).mean(axis=0) - count) / denominator
    nulls = np.empty((len(shifts), width), dtype=np.float32)

    def run(rows):
        total = np.empty((length, width), dtype=np.float32)
        for row in rows:
            total.fill(0)
            for subject, shift in zip(normalized, shifts[row]):
                _shift_add(total, subject, int(shift))
            nulls[row] = (np.einsum("tv,tv->v", total, total, dtype=np.float64) / length - count) / denominator

    chunks = [chunk for chunk in np.array_split(np.arange(len(shifts)), max(1, workers)) if len(chunk)]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(run, chunks))
    observed[~valid] = -1.0
    nulls[:, ~valid] = -1.0
    return observed, nulls


class SelectionCache:
    """Per-story ISC nulls computed once; each fold averages only its training stories."""

    def __init__(self, stories, n_null, seed, workers=1, log=None):
        if n_null < 1 or not stories:
            raise ValueError("Training stories and null draws required")
        self.observed, self.nulls, self.weights = {}, {}, {}
        for story in stories:
            started = time.time()
            self.weights[story.name] = effective_time(story.subjects)
            self.observed[story.name], self.nulls[story.name] = story_isc_nulls(
                story, selection_shifts(story, n_null, seed), workers)
            if log:
                log(f"selection null {story.name}: {story.subjects.shape} in {time.time() - started:.0f}s")

    def select(self, names, networks, floor=0.015, q_threshold=0.05):
        weights = np.asarray([self.weights[name] for name in names])
        observed = np.average([self.observed[name] for name in names], axis=0, weights=weights)
        null = np.zeros_like(self.nulls[names[0]], dtype=np.float64)
        for name, weight in zip(names, weights):
            null += weight * self.nulls[name]
        null /= weights.sum()
        p_values = ((null >= observed).sum(axis=0) + 1) / (len(null) + 1)
        q_values = np.ones(len(networks))
        for network in ("LANG", "MD", "EAC"):
            indices = np.flatnonzero(networks == network)
            if len(indices):
                q_values[indices] = bh_adjust(p_values[indices])
        chosen = (observed >= floor) & (q_values < q_threshold)
        return chosen, observed, q_values, weights


def select_vertices(training, networks, n_null, seed, floor=0.015, q_threshold=0.05, workers=1):
    cache = SelectionCache(training, n_null, seed, workers)
    return cache.select([story.name for story in training], networks, floor, q_threshold)


def ridge_predict(training_x, training_y, test_x, alpha=100.0):
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and positive")
    training_x = np.asarray(training_x, dtype=float)
    training_y = np.asarray(training_y, dtype=float)
    center = training_x.mean(axis=0)
    scale = training_x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = (training_x - center) / scale
    test = (np.asarray(test_x) - center) / scale
    target_center = training_y.mean(axis=0)
    centered_y = training_y - target_center
    if design.shape[1] <= len(design):
        gram = design.T @ design + alpha * np.eye(design.shape[1])
        coefficients = cho_solve(cho_factor(gram), design.T @ centered_y)
        return test @ coefficients + target_center
    gram = design @ design.T + alpha * np.eye(len(design))
    return (test @ design.T) @ cho_solve(cho_factor(gram), centered_y) + target_center


def lag_rows(values, lags):
    result = np.zeros((len(values), values.shape[1] * len(lags)))
    for column, lag in enumerate(lags):
        if lag < 0 or lag >= len(values):
            raise ValueError("Invalid lag")
        result[lag:, column * values.shape[1]:(column + 1) * values.shape[1]] = values[:len(values) - lag]
    return result


def subject_halves(story, split, seed):
    assignments = [hashlib.sha256(f"{seed}:{split}:{identity}".encode()).digest()[0] % 2
                   for identity in story.subject_ids]
    return tuple(np.flatnonzero(np.asarray(assignments) == half) for half in (0, 1))


def valid_split_ids(stories, count, seed):
    selected = []
    for candidate in range(10000):
        if all(min(map(len, subject_halves(story, candidate, seed))) >= 2 for story in stories):
            selected.append(candidate)
            if len(selected) == count:
                return selected
    raise ValueError("Cannot construct identity-stable subject halves with >=2 participants per cohort")


def residual_targets(stories, md_indices, networks, n_splits, alpha, seed, split_ids=None):
    results = [np.zeros((story.subjects.shape[1], len(md_indices))) for story in stories]
    lang = np.flatnonzero(networks == "LANG")
    parcels = np.unique(stories[0].parcel_ids[lang])
    if n_splits < 1:
        raise ValueError("At least one subject split required")
    for split in (valid_split_ids(stories, n_splits, seed) if split_ids is None else split_ids):
        halves = [subject_halves(story, split, seed) for story in stories]
        for direction in (0, 1):
            predictors, targets = [], []
            for story, halves_story in zip(stories, halves):
                lang_mean = story.subjects[halves_story[direction]].mean(axis=0)
                parcel_mean = np.column_stack([
                    lang_mean[:, (story.parcel_ids == parcel) & (networks == "LANG")].mean(axis=1)
                    for parcel in parcels])
                predictors.append(lag_rows(parcel_mean, (0, 1, 2)))
                targets.append(story.subjects[halves_story[1 - direction]].mean(axis=0)[:, md_indices])
            for test_index, story in enumerate(stories):
                training = [index for index, other in enumerate(stories) if other.family != story.family]
                predicted = ridge_predict(np.concatenate([predictors[index][2:] for index in training]),
                                          np.concatenate([targets[index][2:] for index in training]),
                                          predictors[test_index], alpha)
                results[test_index] += (targets[test_index] - predicted) / (2 * n_splits)
    return [standardize(result[2:]) for result in results]


class ResidualModel:
    """Exact cross-fitted LANG->MD ridge from per-story sufficient statistics.

    Equivalent to residual_targets: the ridge is linear in the MD targets, and its
    train-only scaling depends only on summed predictor moments.
    """

    def __init__(self, stories, networks, md_columns, split_ids, seed, alpha, log=None):
        lang = np.asarray(networks) == "LANG"
        parcels = np.unique(stories[0].parcel_ids[lang])
        self.alpha = alpha
        self.position = {int(column): index for index, column in enumerate(md_columns)}
        self.stats, self.mean_target, self.design = {}, {}, {}
        draws = 2 * len(split_ids)
        for story in stories:
            started = time.time()
            parcel_series = np.stack([
                np.asarray(story.subjects[:, :, (story.parcel_ids == parcel) & lang], dtype=np.float64).mean(axis=2)
                for parcel in parcels], axis=2)
            md = np.asarray(story.subjects[:, :, md_columns], dtype=np.float32)
            mean_target = np.zeros((md.shape[1], len(md_columns)))
            stats, designs = [], []
            for split in split_ids:
                halves = subject_halves(story, split, seed)
                for direction in (0, 1):
                    design = lag_rows(parcel_series[halves[direction]].mean(axis=0), (0, 1, 2))
                    target = md[halves[1 - direction]].mean(axis=0, dtype=np.float64)
                    mean_target += target / draws
                    rows, values = design[2:], target[2:]
                    stats.append(dict(n=len(rows), zsum=rows.sum(axis=0), ztz=rows.T @ rows,
                                      cross=rows.T @ values, ysum=values.sum(axis=0)))
                    designs.append(design)
            self.stats[story.name], self.design[story.name] = stats, designs
            self.mean_target[story.name] = mean_target
            if log:
                log(f"residual statistics {story.name} in {time.time() - started:.0f}s")

    def targets(self, name, training, columns):
        positions = np.asarray([self.position[int(column)] for column in columns])
        predicted = np.zeros((self.mean_target[name].shape[0], len(positions)))
        draws = len(self.stats[name])
        for draw in range(draws):
            parts = [self.stats[other][draw] for other in training]
            count = sum(part["n"] for part in parts)
            center = sum(part["zsum"] for part in parts) / count
            centered_gram = sum(part["ztz"] for part in parts) - count * np.outer(center, center)
            scale = np.sqrt(np.clip(np.diag(centered_gram) / count, 0, None))
            scale = np.where(scale > 1e-8, scale, 1.0)
            target_center = sum(part["ysum"][positions] for part in parts) / count
            cross = (sum(part["cross"][:, positions] for part in parts)
                     - count * np.outer(center, target_center)) / scale[:, None]
            gram = centered_gram / np.outer(scale, scale) + self.alpha * np.eye(len(scale))
            coefficients = cho_solve(cho_factor(gram), cross)
            predicted += ((self.design[name][draw] - center) / scale) @ coefficients + target_center
        residual = self.mean_target[name][:, positions] - predicted / draws
        return standardize(residual[2:])


class FoldEncoder:
    """Ridge readout as a fixed linear smoother: predictions = kernel @ centered training targets.

    Circularly shifting a training story's design rows leaves the train-only scaling
    and Gram matrix unchanged, so refitted null predictions reuse one kernel.
    """

    def __init__(self, train_designs, test_designs, alpha):
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("Ridge alpha must be finite and positive")
        training = np.concatenate(train_designs).astype(np.float64)
        center = training.mean(axis=0)
        scale = training.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        design = (training - center) / scale
        del training
        test = (np.concatenate(test_designs) - center) / scale
        if design.shape[1] <= len(design):
            gram = design.T @ design + alpha * np.eye(design.shape[1])
            self.kernel = test @ cho_solve(cho_factor(gram), design.T)
        else:
            gram = design @ design.T + alpha * np.eye(len(design))
            self.kernel = cho_solve(cho_factor(gram), design @ test.T).T
        self.train_lengths = [len(matrix) for matrix in train_designs]
        self.test_lengths = [len(matrix) for matrix in test_designs]

    def shifted_kernel(self, train_shifts, test_shifts, kernel=None):
        kernel = self.kernel if kernel is None else kernel
        columns, start = [], 0
        for length, shift in zip(self.train_lengths, train_shifts):
            columns.append(start + (np.arange(length) - shift) % length)
            start += length
        rows, start = [], 0
        for length, shift in zip(self.test_lengths, test_shifts):
            rows.append(start + (np.arange(length) - shift) % length)
            start += length
        return kernel[np.concatenate(rows)][:, np.concatenate(columns)]


def _standardize32(values):
    centered = values - values.mean(axis=0)
    scale = centered.std(axis=0)
    return centered / np.where(scale > 1e-8, scale, 1.0)


def prepare_fold(stories, held_family, networks, settings, rng=None):
    train = [story for story in stories if story.family != held_family]
    test = [story for story in stories if story.family == held_family]
    mask, rho, q_values, weights = select_vertices(
        train, networks, settings["selection_nulls"], settings["seed"], settings["rho_floor"], settings["selection_q"])
    indices = {network: np.flatnonzero(mask & (networks == network)) for network in ("LANG", "MD")}
    if any(len(selected) == 0 for selected in indices.values()):
        return {"family": held_family, "status": "insufficient_reliable_vertices",
                "counts": {network: len(selected) for network, selected in indices.items()}}
    md = indices["MD"]
    splits = valid_split_ids(stories, settings["subject_splits"], settings["seed"])
    train_residual = residual_targets(train, md, networks, settings["subject_splits"], settings["residual_alpha"], settings["seed"], splits)
    test_residual = []
    for story in test:
        test_residual.append(residual_targets(train + [story], md, networks, settings["subject_splits"],
                                             settings["residual_alpha"], settings["seed"], splits)[-1])
    responses = {}
    for network, selected in indices.items():
        responses[network] = ([standardize(story.subjects.mean(axis=0)[2:, selected]) for story in train],
                      [standardize(story.subjects.mean(axis=0)[2:, selected]) for story in test])
    responses["MD_residual"] = (train_residual, test_residual)
    return {"family": held_family, "status": "ready", "train": train, "test": test,
            "indices": indices, "responses": responses, "rho": rho, "q": q_values,
            "training_weights": weights}


def fold_scores(fold, feature, alpha, rng=None):
    train, test = fold["train"], fold["test"]
    designs = [getattr(story, feature)[2:] for story in train + test]
    if rng is not None:
        designs = [np.roll(matrix, int(rng.integers(10, len(matrix) - 9)), axis=0) for matrix in designs]
    training_x = np.concatenate(designs[:len(train)])
    test_x = np.concatenate(designs[len(train):])
    output = {}
    for network, (training_y, test_y) in fold["responses"].items():
        predictions = ridge_predict(training_x, np.concatenate(training_y), test_x, alpha)
        start = 0
        records = []
        for story, response in zip(test, test_y):
            stop = start + len(response)
            records.append({"story": story.name, "family": story.family,
                            "actual": standardize(response), "predicted": standardize(predictions[start:stop]),
                            "indices": fold["indices"]["MD" if network == "MD_residual" else network]})
            start = stop
        output[network] = records
    return output


def score_records(records, row_draws=None):
    vertices = sorted({int(index) for record in records for index in record["indices"]})
    scores = []
    for vertex in vertices:
        actual, predicted = [], []
        for position, record in enumerate(records):
            match = np.flatnonzero(record["indices"] == vertex)
            if len(match):
                rows = slice(None) if row_draws is None else row_draws[position]
                actual.append(record["actual"][rows, match[0]])
                predicted.append(record["predicted"][rows, match[0]])
        scores.append(correlations(np.concatenate(actual)[:, None], np.concatenate(predicted)[:, None])[0])
    return float(np.mean(scores))


def bootstrap_delta_reference(gpt_records, baseline_records, draws, rng, block=10):
    families = sorted({record["family"] for record in gpt_records})
    estimates = []
    for _ in range(draws):
        family_deltas = []
        for family in rng.choice(families, size=len(families), replace=True):
            deltas = []
            for gpt, baseline in zip(gpt_records, baseline_records):
                if gpt["family"] != family:
                    continue
                length = len(gpt["actual"])
                starts = rng.integers(0, length, size=int(np.ceil(length / block)))
                rows = ((starts[:, None] + np.arange(block)) % length).ravel()[:length]
                deltas.append(float(np.mean(correlations(gpt["actual"][rows], gpt["predicted"][rows])
                                            - correlations(baseline["actual"][rows], baseline["predicted"][rows]))))
            family_deltas.append(float(np.mean(deltas)))
        estimates.append(float(np.mean(family_deltas)))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _count_correlations(counts, actual, predicted):
    """Correlations on row-resampled data from row counts; matches correlations()."""
    total = counts.sum(axis=1, keepdims=True)
    sum_a, sum_b = counts @ actual, counts @ predicted
    mean_a, mean_b = sum_a / total, sum_b / total
    cov = counts @ (actual * predicted) / total - mean_a * mean_b
    scale_a = np.sqrt(np.clip(counts @ (actual * actual) / total - mean_a ** 2, 0, None))
    scale_b = np.sqrt(np.clip(counts @ (predicted * predicted) / total - mean_b ** 2, 0, None))
    return cov / (np.where(scale_a > 1e-8, scale_a, 1.0) * np.where(scale_b > 1e-8, scale_b, 1.0))


def bootstrap_delta(gpt_records, baseline_records, draws, rng, block=10, chunk=500):
    """Same random draws and estimand as bootstrap_delta_reference, batched by record."""
    families = sorted({record["family"] for record in gpt_records})
    members = {family: [index for index, record in enumerate(gpt_records) if record["family"] == family]
               for family in families}
    occurrences = [[] for _ in gpt_records]
    for draw in range(draws):
        for slot, family in enumerate(rng.choice(families, size=len(families), replace=True)):
            for index in members[family]:
                length = len(gpt_records[index]["actual"])
                starts = rng.integers(0, length, size=int(np.ceil(length / block)))
                rows = ((starts[:, None] + np.arange(block)) % length).ravel()[:length]
                occurrences[index].append((draw, slot, np.bincount(rows, minlength=length).astype(np.int16)))
    sums = np.zeros((draws, len(families)))
    counts_per_slot = np.zeros((draws, len(families)))
    for index, items in enumerate(occurrences):
        gpt, baseline = gpt_records[index], baseline_records[index]
        for start in range(0, len(items), chunk):
            batch = items[start:start + chunk]
            counts = np.stack([item[2] for item in batch]).astype(np.float64)
            delta = (_count_correlations(counts, gpt["actual"], gpt["predicted"])
                     - _count_correlations(counts, baseline["actual"], baseline["predicted"])).mean(axis=1)
            for (draw, slot, _), value in zip(batch, delta):
                sums[draw, slot] += value
                counts_per_slot[draw, slot] += 1
    estimates = (sums / counts_per_slot).mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def family_score(records):
    families = sorted({record["family"] for record in records})
    return float(np.mean([np.mean([np.mean(correlations(record["actual"], record["predicted"]))
                                   for record in records if record["family"] == family]) for family in families]))


def encoding_shifts(stories, n_perm, seed):
    """One within-story circular shift of the analysis rows per draw and story, shared by all folds."""
    rng = np.random.default_rng(_stream(seed, "encoding"))
    ordered = sorted(stories, key=lambda story: story.name)
    table = np.stack([rng.integers(10, len(story.gpt2) - 2 - 9, size=n_perm) for story in ordered], axis=1)
    return {story.name: table[:, index] for index, story in enumerate(ordered)}


class VertexScorer:
    """Mean over covered vertices of the correlation of per-story-standardized, concatenated series.

    Each block has mean 0 and population variance 1, so the concatenated correlation is the
    summed cross-product divided by the summed length (matches score_records).
    """

    def __init__(self, records):
        self.vertices = np.unique(np.concatenate([record["indices"] for record in records]))
        self.lookup = [np.searchsorted(self.vertices, record["indices"]) for record in records]
        self.lengths = np.zeros(len(self.vertices))
        for record, positions in zip(records, self.lookup):
            self.lengths[positions] += len(record["actual"])
        self.actual = [record["actual"] for record in records]

    def score(self, predictions):
        total = np.zeros(len(self.vertices))
        for actual, predicted, positions in zip(self.actual, predictions, self.lookup):
            total[positions] += np.einsum("tv,tv->v", actual, _standardize32(predicted))
        return float(np.mean(total / self.lengths))


def analyze(stories, networks, settings, reference, workers=1, log=None):
    networks = np.asarray(networks)
    validate_arrays(stories, networks)
    log = log or (lambda message: None)
    families = sorted({story.family for story in stories})
    selection = SelectionCache(stories, settings["selection_nulls"], settings["seed"], workers, log)
    folds = []
    for family in families:
        train = [story for story in stories if story.family != family]
        test = [story for story in stories if story.family == family]
        mask, _, _, _ = selection.select([story.name for story in train], networks,
                                         settings["rho_floor"], settings["selection_q"])
        indices = {network: np.flatnonzero(mask & (networks == network)) for network in ("LANG", "MD")}
        folds.append({"family": family, "train": train, "test": test, "indices": indices,
                      "status": "ready" if all(len(value) for value in indices.values())
                      else "insufficient_reliable_vertices"})
    audit = [{"family": fold["family"], "status": fold["status"],
              "selected_counts": {key: int(len(value)) for key, value in fold["indices"].items()},
              "training_stories": [story.name for story in fold["train"]],
              "test_stories": [story.name for story in fold["test"]],
              "selected_vertex_columns": {key: value.tolist() for key, value in fold["indices"].items()}}
             for fold in folds]
    for item in audit:
        log(f"fold {item['family']}: {item['status']} {item['selected_counts']}")
    if any(fold["status"] != "ready" for fold in folds):
        return {"status": "blocked_selection", "folds": audit, "verdict": "NOT_EVALUABLE"}

    md_union = np.unique(np.concatenate([fold["indices"]["MD"] for fold in folds]))
    splits = valid_split_ids(stories, settings["subject_splits"], settings["seed"])
    residuals = ResidualModel(stories, networks, md_union, splits, settings["seed"], settings["residual_alpha"], log)
    group = {story.name: np.asarray(story.subjects, dtype=np.float32).mean(axis=0)[2:] for story in stories}

    observed = {network: [] for network in ("LANG", "MD", "MD_residual")}
    baseline = {network: [] for network in observed}
    encoders, targets = [], []
    for fold in folds:
        started = time.time()
        train, test, indices = fold["train"], fold["test"], fold["indices"]
        responses = {network: ([_standardize32(group[story.name][:, indices[network]]) for story in train],
                               [_standardize32(group[story.name][:, indices[network]]) for story in test])
                     for network in ("LANG", "MD")}
        responses["MD_residual"] = (
            [residuals.targets(story.name, [other.name for other in train if other.family != story.family], indices["MD"])
             for story in train],
            [residuals.targets(story.name, [other.name for other in train], indices["MD"]) for story in test])
        fold_targets = {}
        for network, (training_y, test_y) in responses.items():
            stacked = np.concatenate(training_y)
            fold_targets[network] = (stacked - stacked.mean(axis=0), [_standardize32(value) for value in test_y])
        targets.append(fold_targets)
        for feature, destination in (("gpt2", observed), ("baseline", baseline)):
            encoder = FoldEncoder([getattr(story, feature)[2:] for story in train],
                                  [getattr(story, feature)[2:] for story in test], settings["encoding_alpha"])
            for network, (centered_y, test_y) in fold_targets.items():
                predictions = encoder.kernel @ centered_y
                start = 0
                for story, actual in zip(test, test_y):
                    stop = start + len(actual)
                    destination[network].append({"story": story.name, "family": story.family, "actual": actual,
                                                 "predicted": _standardize32(predictions[start:stop]),
                                                 "indices": indices["MD" if network == "MD_residual" else network]})
                    start = stop
            if feature == "gpt2":
                encoders.append(encoder)
        log(f"fold {fold['family']} encoded in {time.time() - started:.0f}s")

    scorers = {network: VertexScorer(records) for network, records in observed.items()}
    observed_scores = {network: scorers[network].score([record["predicted"] for record in records])
                       for network, records in observed.items()}
    log(f"observed r: {observed_scores}")
    shifts = encoding_shifts(stories, settings["encoding_nulls"], settings["seed"])
    null_statistics = {network: [] for network in observed}
    started = time.time()
    for draw in range(settings["encoding_nulls"]):
        predictions = {network: [] for network in observed}
        for fold, encoder, fold_targets in zip(folds, encoders, targets):
            kernel = encoder.shifted_kernel([shifts[story.name][draw] for story in fold["train"]],
                                            [shifts[story.name][draw] for story in fold["test"]])
            for network, (centered_y, test_y) in fold_targets.items():
                stacked, start = kernel @ centered_y, 0
                for actual in test_y:
                    predictions[network].append(stacked[start:start + len(actual)])
                    start += len(actual)
        for network in observed:
            null_statistics[network].append(scorers[network].score(predictions[network]))
        if (draw + 1) % 50 == 0:
            log(f"encoding null {draw + 1}/{settings['encoding_nulls']} ({time.time() - started:.0f}s)")
    p_values = {network: (int(np.sum(np.asarray(values) >= observed_scores[network])) + 1)
                / (settings["encoding_nulls"] + 1) for network, values in null_statistics.items()}
    rng = np.random.default_rng(_stream(settings["seed"], "bootstrap"))
    gate1_q = bh_adjust([p_values["LANG"], p_values["MD"]])
    network_results = {}
    for network, q_value in zip(("LANG", "MD"), gate1_q):
        network_results[network] = {"r": observed_scores[network], "p": p_values[network], "q": float(q_value),
            "minimum_r": settings["minimum_r"], "pass": bool(q_value < settings["gate_q"] and observed_scores[network] >= settings["minimum_r"]),
            "shain_reference_only": reference.get("shain_observed_r_LANG" if network == "LANG" else "shain_observed_r_MD_raw"),
            "narratives_expected_r": None}
    residual_r = observed_scores["MD_residual"]
    gate2_pass = p_values["MD_residual"] < settings["gate_q"] and residual_r >= settings["minimum_r"]
    interval = bootstrap_delta(observed["MD_residual"], baseline["MD_residual"], settings["bootstrap_draws"], rng)
    baseline_r = family_score(baseline["MD_residual"])
    gate3_gpt_r = family_score(observed["MD_residual"])
    gate3_pass = interval[0] > 0
    all_pass = all(record["pass"] for record in network_results.values()) and gate2_pass and gate3_pass
    return {"status": "completed_primary_gates", "folds": audit,
            "gate_1": {"networks": network_results, "pass": all(record["pass"] for record in network_results.values())},
            "gate_2": {"r": residual_r, "p": p_values["MD_residual"], "q": p_values["MD_residual"],
                       "family": "one primary residual-MD test", "pass": bool(gate2_pass),
                       "band": "GO" if gate2_pass and residual_r >= 0.06 else "SUBTLE_GO" if gate2_pass else "NO_GO"},
            "gate_3": {"gpt2_r": gate3_gpt_r, "baseline_r": baseline_r, "delta_r": gate3_gpt_r - baseline_r,
                       "ci95": interval, "pass": bool(gate3_pass),
                       "estimand": "Equal-family mean of within-story mean selected-vertex paired correlation differences",
                       "interval_scope": "Conditional on fitted models and outer-fold vertex supports; not full-pipeline uncertainty"},
            "null_statistics": null_statistics, "settings": settings,
            "verdict": "PRIMARY_GATES_PASS_PENDING_DIAGNOSTICS" if all_pass else "NO_GO",
            "limitations": ["Group-atlas vertices, not individually localized MD.",
                "Cross-subject residualization does not correct noisy-LANG attenuation.",
                "Circular stationarity is an assumption, not an exact finite-sample guarantee.",
                "Gate 4, ceiling normalization, secondary layers and sensitivity analyses are not implemented."]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_design(archive, record, lags):
    indices = record["tr_indices"]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Primary BOLD rows must be consecutive original TRs")
    history_indices = archive["history_tr_indices"].tolist()
    metadata = json.loads(str(archive["metadata"].item()))
    declared = metadata.get("lag_history", {})
    required_history = sorted({source - lag for source in indices for lag in lags} - set(indices))
    if (declared.get("lags_built") is not False or declared.get("required_lags_tr") != list(lags)
            or declared.get("tr_indices") != history_indices or history_indices != required_history
            or declared.get("edge_policy") != "drop_rows_without_real_in_story_history"
            or declared.get("unavailable_lags") != {}):
        raise ValueError("Inconsistent or unsupported lag-history metadata")
    if len(set(history_indices)) != len(history_indices) or set(history_indices) & set(indices):
        raise ValueError("Duplicate feature history rows")
    all_indices = indices + history_indices
    lookup = {source: index for index, source in enumerate(all_indices)}
    if any(source - lag not in lookup for source in indices for lag in lags):
        raise ValueError("Missing actual within-story feature lag history")
    designs = []
    for key in ("gpt2", "baseline"):
        combined = np.concatenate([archive[key], archive[key + "_history"]])
        if len(combined) != len(all_indices) or not np.isfinite(combined).all():
            raise ValueError("Invalid feature history shape or values")
        designs.append(np.hstack([combined[[lookup[source - lag] for source in indices]] for lag in lags]))
    return designs


def load_dataset(manifest_path, gate_config):
    manifest_path = Path(manifest_path)
    root = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("provenance") != "real_narratives" or not manifest.get("primary_eligible"):
        raise ValueError("Manifest is not an eligible real Narratives vertex dataset")
    if manifest.get("boundary_rings") != 1 or manifest.get("feature_metadata_strict") is not True:
        raise ValueError("Primary dataset requires the one-ring mask")
    targets = manifest["targets"]
    identities = [(target["hemisphere"], target["vertex_id"]) for target in targets]
    if len(set(identities)) != len(identities) or {identity[0] for identity in identities} != {"L", "R"}:
        raise ValueError("Duplicated or missing bilateral vertex identities")
    for hemisphere, vertex in identities:
        if type(vertex) is not int or not 0 <= vertex < 40962 or hemisphere not in ("L", "R"):
            raise ValueError("Invalid fsaverage6 vertex identity")
    networks = np.asarray([target["network"] for target in targets])
    for network in ("LANG", "MD"):
        if {target["hemisphere"] for target in targets if target["network"] == network} != {"L", "R"}:
            raise ValueError(f"Both hemispheres required for {network} targets")
    if any("parcel" not in target for target in targets):
        raise ValueError("Manifest must preserve parcel identity for fixed LANG covariates")
    parcels = np.asarray([f"{target['hemisphere']}:{target['parcel']}" for target in targets])
    output_hashes = manifest.get("output_hashes", {})
    stories = []
    for record in manifest["stories"]:
        paths = []
        for key in ("bold", "features"):
            relative = record[key]
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or output_hashes.get(relative) != sha256(path):
                raise ValueError(f"Missing or unverified {key}: {record['name']}")
            paths.append(path)
        with np.load(paths[1], allow_pickle=False) as features:
            from narratives_prepare import FAMILIES, validate_features
            validate_features(paths[1], len(record["tr_indices"]), record)
            metadata = json.loads(str(features["metadata"].item()))
            expected = {"model": "gpt2", "layer": gate_config["model"]["primary_layer"],
                        "context_window_tr": 5, "tr_seconds": 1.5, "provenance": "real_narratives", "causal_windows": True}
            if any(metadata.get(key) != value for key, value in expected.items()) or metadata.get("tr_indices") != record["tr_indices"]:
                raise ValueError(f"Feature semantics or timing mismatch: {record['name']}")
            if not metadata.get("model_weights_sha256") or not metadata.get("baseline_provenance") or not metadata.get("source_hashes"):
                raise ValueError("Missing feature provenance")
            names = features["baseline_names"].astype(str).tolist()
            if set(names) != set(BASELINES) or len(names) != len(BASELINES):
                raise ValueError("Extended comparator must contain exactly the nine declared baseline columns")
            gpt2 = features["gpt2"]
            if gpt2.ndim != 2 or gpt2.shape[1] != 768:
                raise ValueError("GPT-2 features must have 768 columns")
            lags = gate_config["model"]["lags_tr"]
            if lags != [1, 2, 3, 4] or gate_config["residualization"]["lang_lags_tr"] != [0, 1, 2]:
                raise ValueError("Unsupported lag specification")
            gpt_design, baseline_design = feature_design(features, record, lags)
            column_order = [lag * len(names) + names.index(name) for lag in range(len(lags)) for name in BASELINES]
            baseline_design = baseline_design[:, column_order]
            subjects = np.load(paths[0], mmap_mode="r", allow_pickle=False)
            trim = gate_config["dataset"]["trim_tr"]
            event = record["event"]
            expected_indices = list(range(event["onset"] + trim["start"], event["onset"] + event["duration"] - trim["end"]))
            if record["tr_indices"] != expected_indices or record.get("tr_seconds") != 1.5:
                raise ValueError("Prepared segment disagrees with configured trim or TR")
            if record["family"] != FAMILIES.get(record["task"], record["task"]):
                raise ValueError("Related stimuli must remain in the same task family")
            stories.append(Story(record["name"], record["family"], subjects,
                                 gpt_design, baseline_design, tuple(record["subject_ids"]), parcels))
    tasks = {record["task"] for record in manifest["stories"]}
    if tasks != set(gate_config["dataset"]["primary_tasks"]):
        raise ValueError("Primary run requires the complete declared primary task pool")
    validate_arrays(stories, networks)
    return stories, networks, manifest


def verify_freeze(config_path, config):
    timestamp = config.get("frozen_at")
    commit = config.get("git_commit_before_first_run")
    if not isinstance(timestamp, str) or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("A valid freeze timestamp and full Git commit are required before real analysis")
    try:
        frozen = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid freeze timestamp") from error
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Freeze timestamp must include a timezone and cannot be in the future")
    config_path = Path(config_path).resolve()

    def git(*arguments):
        completed = subprocess.run(["git", "-C", str(config_path.parent), *arguments],
                                   capture_output=True, check=False)
        if completed.returncode:
            raise ValueError("Freeze commit/config/engine cannot be verified in Git")
        return completed.stdout

    root = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    resolved = git("rev-parse", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise ValueError("Freeze commit did not resolve exactly")
    relative = config_path.relative_to(root).as_posix()
    committed = json.loads(git("show", f"{commit}:{relative}").decode("utf-8"))
    canonical_current = {key: value for key, value in config.items() if key != "git_commit_before_first_run"}
    canonical_committed = {key: value for key, value in committed.items() if key != "git_commit_before_first_run"}
    if canonical_current != canonical_committed:
        raise ValueError("Current specification differs from the frozen commit")
    engine_hashes = {}
    for name in ("narratives_stage_a.py", "narratives_prepare.py", "narratives_features.py"):
        path = Path(__file__).with_name(name).resolve()
        saved = git("show", f"{commit}:{path.relative_to(root).as_posix()}")
        if saved.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Engine differs from frozen commit: {name}")
        engine_hashes[name] = sha256(path)
    return engine_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stories, networks, manifest = load_dataset(args.dataset, config)
    if args.check_inputs:
        print(json.dumps({"status": "inputs_validated_not_run", "stories": len(stories), "vertices": len(networks)}))
        return
    execution = config.get("execution")
    if not execution or execution.get("engine") != "narratives_stage_a_v1":
        raise ValueError("Config lacks reviewed Stage A execution settings")
    engine_hashes = verify_freeze(args.config, config)
    settings = execution["settings"]
    if settings["encoding_nulls"] < 999 or settings["selection_nulls"] < 200 or settings["subject_splits"] < 20 or settings["bootstrap_draws"] < 10000:
        raise ValueError("Insufficient primary inference draws")
    gate1 = config["gates"]["gate_1_baseline_sanity"]
    if settings["minimum_r"] != gate1["minimum_r"] or settings["gate_q"] != gate1["q_threshold"]:
        raise ValueError("Execution settings disagree with Gate 1")
    if args.output is None or args.output.exists():
        raise ValueError("Supply a new --output path; previous results are never overwritten")
    def log(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    result = analyze(stories, networks, settings, gate1["reference_only"], workers=args.workers, log=log)
    result.update(provenance="real_narratives", config_sha256=sha256(args.config), dataset_sha256=sha256(args.dataset),
                  preparation_config_sha256=manifest["config_sha256"], engine_sha256=engine_hashes,
                  frozen_commit=config["git_commit_before_first_run"], target_identities=manifest["targets"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"], result["verdict"])


if __name__ == "__main__":
    main()
