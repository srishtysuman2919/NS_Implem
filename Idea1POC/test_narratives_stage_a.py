"""Synthetic software checks, never Narratives scientific results."""
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
from narratives_stage_a import (
    Story, analyze, bh_adjust, correlations, lag_rows, pairwise_isc,
    prepare_fold, residual_targets, ridge_predict, select_vertices, subject_halves, feature_design, verify_freeze,
    FoldEncoder, ResidualModel, SelectionCache, VertexScorer, bootstrap_delta, bootstrap_delta_reference,
    score_records, selection_shifts, standardize, story_isc_nulls, valid_split_ids,
)


def fixture():
    rng = np.random.default_rng(123)
    stories = []
    for index in range(4):
        features = rng.normal(size=(72, 3))
        signal = features @ np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [0.4, 0, -0.4, 0]])
        subjects = signal[None] + rng.normal(scale=0.15, size=(8, 72, 4))
        stories.append(Story(str(index), str(index), subjects, features, rng.normal(size=(72, 2)),
                             tuple(f"sub-{subject}" for subject in range(8)), np.asarray([1, 2, 3, 4])))
    return stories, np.asarray(["LANG", "LANG", "MD", "MD"])


SETTINGS = dict(seed=12, selection_nulls=39, encoding_nulls=39, subject_splits=2,
                bootstrap_draws=29, rho_floor=0.015, selection_q=0.05,
                minimum_r=0.04, gate_q=0.05, encoding_alpha=1.0, residual_alpha=1.0)


class StageATests(unittest.TestCase):
    def test_pairwise_matches_brute_force(self):
        subjects = fixture()[0][0].subjects
        brute = np.mean([correlations(subjects[first], subjects[second])
                         for first in range(8) for second in range(first + 1, 8)], axis=0)
        np.testing.assert_allclose(pairwise_isc(subjects), brute, atol=1e-12)

    def test_bh(self):
        np.testing.assert_allclose(bh_adjust([0.01, 0.04]), [0.02, 0.04])
        with self.assertRaises(ValueError):
            bh_adjust([float("nan")])

    def test_no_wraparound(self):
        result = lag_rows(np.arange(5)[:, None] + 1, [1, 2])
        np.testing.assert_array_equal(result[:3], [[0, 0], [1, 0], [2, 1]])

    def test_training_only_scaler(self):
        features = np.arange(10)[:, None].astype(float)
        target = 2 * features + 4
        first = ridge_predict(features, target, np.asarray([[5.0]]), 1)
        second = ridge_predict(features, target, np.asarray([[5.0], [1e9]]), 1)
        np.testing.assert_allclose(first[0], second[0])

    def test_selection_cannot_see_heldout(self):
        stories, networks = fixture()
        train = stories[1:]
        first = select_vertices(train, networks, 39, 1)
        stories[0].subjects[:] = np.random.default_rng(42).normal(size=stories[0].subjects.shape)
        second = select_vertices(train, networks, 39, 1)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_families_never_split(self):
        stories, networks = fixture()
        stories[1].family = "0"
        fold = prepare_fold(stories, "0", networks, SETTINGS, np.random.default_rng(12))
        self.assertEqual(fold["status"], "ready")
        self.assertEqual([story.name for story in fold["test"]], ["0", "1"])
        self.assertTrue(all(story.family != "0" for story in fold["train"]))

    def test_subject_halves_disjoint_stable(self):
        story = fixture()[0][0]
        first, second = subject_halves(story, 0, 42)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(set(first) | set(second), set(range(8)))

    def test_crossfit_residual_shape(self):
        stories, networks = fixture()
        residuals = residual_targets(stories, np.asarray([2, 3]), networks, 2, 1, 12)
        self.assertEqual(len(residuals), 4)
        self.assertTrue(all(value.shape == (70, 2) and np.isfinite(value).all() for value in residuals))

    def test_halves_stable_across_different_cohorts(self):
        story = fixture()[0][0]
        original = subject_halves(story, 0, 42)
        assignment = {story.subject_ids[index]: half for half, indices in enumerate(original) for index in indices}
        story.subject_ids = story.subject_ids[1:]
        reduced = subject_halves(story, 0, 42)
        self.assertTrue(all(assignment[story.subject_ids[index]] == half
                            for half, indices in enumerate(reduced) for index in indices))

    def test_feature_history_original_tr_not_row_position(self):
        archive = {"gpt2": np.asarray([[30], [40]]), "baseline": np.asarray([[3], [4]]),
                   "gpt2_history": np.asarray([[10], [20]]), "baseline_history": np.asarray([[1], [2]]),
                   "history_tr_indices": np.asarray([1, 2])}
        archive["metadata"] = np.array(json.dumps({"lag_history": {
            "lags_built": False, "required_lags_tr": [1, 2], "tr_indices": [1, 2],
            "edge_policy": "drop_rows_without_real_in_story_history", "unavailable_lags": {}}}))
        gpt, baseline = feature_design(archive, {"tr_indices": [3, 4]}, [1, 2])
        np.testing.assert_array_equal(gpt, [[20, 10], [30, 20]])
        np.testing.assert_array_equal(baseline, [[2, 1], [3, 2]])
        with self.assertRaises(ValueError):
            feature_design(archive, {"tr_indices": [3, 5]}, [1, 2])

    def test_lag_metadata_cannot_contradict_arrays(self):
        declared = dict(lags_built=False, required_lags_tr=[1], tr_indices=[1],
                        edge_policy="drop_rows_without_real_in_story_history", unavailable_lags={})
        archive = dict(gpt2=np.ones((1, 2)), baseline=np.ones((1, 2)),
                       gpt2_history=np.ones((1, 2)), baseline_history=np.ones((1, 2)),
                       history_tr_indices=np.asarray([1]))
        for key, value in (("lags_built", True), ("required_lags_tr", [0]), ("tr_indices", [999]),
                           ("edge_policy", "circular"), ("unavailable_lags", {"2": [1]})):
            archive["metadata"] = np.array(json.dumps({"lag_history": dict(declared, **{key: value})}))
            with self.subTest(field=key), self.assertRaises(ValueError):
                feature_design(archive, {"tr_indices": [2]}, [1])

    def test_freeze_rejects_invalid_fields_before_git(self):
        for timestamp, commit in (("", ""), (" ", "a" * 40), ("garbage", "a" * 40),
                                  ("2999-01-01T00:00:00+00:00", "a" * 40),
                                  ("2020-01-01T00:00:00", "a" * 40)):
            with self.subTest(timestamp=timestamp), patch("narratives_stage_a.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    verify_freeze(Path(__file__).with_name("narratives_gate_config.json"),
                                  dict(frozen_at=timestamp, git_commit_before_first_run=commit))
                run.assert_not_called()

    def test_freeze_rejects_nonexistent_commit(self):
        with patch("narratives_stage_a.subprocess.run", return_value=subprocess.CompletedProcess([], 1, b"", b"missing")):
            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                verify_freeze(Path(__file__).with_name("narratives_gate_config.json"),
                              dict(frozen_at="2020-01-01T00:00:00+00:00", git_commit_before_first_run="a" * 40))

    def test_end_to_end_signal(self):
        stories, networks = fixture()
        result = analyze(stories, networks, SETTINGS, {"shain_observed_r_LANG": 0.134})
        self.assertEqual(result["status"], "completed_primary_gates")
        self.assertTrue(result["gate_1"]["pass"])
        self.assertGreater(result["gate_1"]["networks"]["LANG"]["r"], 0.9)
        self.assertIsNone(result["gate_1"]["networks"]["MD"]["narratives_expected_r"])
        self.assertEqual(len(result["null_statistics"]["MD"]), 39)

    def test_null_does_not_get_signal_verdict(self):
        stories, networks = fixture()
        rng = np.random.default_rng(52)
        for story in stories:
            story.gpt2 = rng.normal(size=story.gpt2.shape)
        result = analyze(stories, networks, SETTINGS, {})
        self.assertFalse(result["gate_1"]["pass"])
        self.assertEqual(result["verdict"], "NO_GO")


def mixed_fixture():
    """Unequal cohorts, families and lengths with noncontiguous selected columns."""
    rng = np.random.default_rng(7)
    networks = np.asarray(["LANG", "MD", "LANG", "EAC", "MD", "MD", "LANG", "MD"])
    parcels = np.asarray(["L:1", "L:5", "L:2", "L:9", "L:5", "L:6", "L:1", "L:6"])
    stories = []
    for index, (family, length, count) in enumerate([("a", 60, 6), ("a", 48, 5), ("b", 55, 7), ("c", 50, 6), ("d", 64, 8)]):
        features = rng.normal(size=(length, 5))
        signal = features @ rng.normal(size=(5, len(networks)))
        subjects = (signal[None] + rng.normal(scale=1.5, size=(count, length, len(networks)))).astype(np.float32)
        identities = tuple(f"sub-{(index * 3 + offset) % 11:02d}" for offset in range(count))
        stories.append(Story(f"s{index}", family, subjects, features, rng.normal(size=(length, 3)), identities, parcels))
    return stories, networks


class FastPathEquivalenceTests(unittest.TestCase):
    def test_isc_null_rows_match_explicit_rolls(self):
        story = mixed_fixture()[0][0]
        shifts = selection_shifts(story, 5, 3)
        observed, nulls = story_isc_nulls(story, shifts, workers=2)
        np.testing.assert_allclose(observed, pairwise_isc(story.subjects), atol=1e-5)
        for row in range(5):
            rolled = np.stack([np.roll(subject, int(shift), axis=0) for subject, shift in zip(story.subjects, shifts[row])])
            np.testing.assert_allclose(nulls[row], pairwise_isc(rolled), atol=1e-5)

    def test_selection_cache_matches_manual_average(self):
        stories, networks = mixed_fixture()
        cache = SelectionCache(stories, 9, 3)
        names = ["s0", "s2", "s4"]
        chosen, observed, q_values, weights = cache.select(names, networks, floor=-1, q_threshold=1.1)
        expected = np.average([cache.observed[name] for name in names], axis=0, weights=weights)
        np.testing.assert_allclose(observed, expected)
        null = np.average([cache.nulls[name] for name in names], axis=0, weights=weights)
        p_values = ((null >= expected).sum(axis=0) + 1) / 10
        for network in ("LANG", "MD", "EAC"):
            columns = networks == network
            np.testing.assert_allclose(q_values[columns], bh_adjust(p_values[columns]))
        self.assertTrue(chosen.all())

    def test_residual_model_matches_reference(self):
        stories, networks = mixed_fixture()
        md = np.asarray([1, 4, 5, 7])
        splits = valid_split_ids(stories, 2, 9)
        model = ResidualModel(stories, networks, md, splits, 9, 2.0)
        selected = np.asarray([7, 1])
        train = [story for story in stories if story.family != "c"]
        reference = residual_targets(train, selected, networks, 2, 2.0, 9, splits)
        for story, expected in zip(train, reference):
            actual = model.targets(story.name, [other.name for other in train if other.family != story.family], selected)
            np.testing.assert_allclose(actual, expected, atol=1e-4)
        test = [story for story in stories if story.family == "c"][0]
        expected = residual_targets(train + [test], selected, networks, 2, 2.0, 9, splits)[-1]
        np.testing.assert_allclose(model.targets(test.name, [story.name for story in train], selected), expected, atol=1e-4)

    def test_fold_encoder_matches_refitted_ridge_on_shifted_designs(self):
        rng = np.random.default_rng(4)
        for width in (3, 70):
            train = [rng.normal(size=(length, width)) for length in (30, 25)]
            test = [rng.normal(size=(20, width))]
            targets = rng.normal(size=(55, 4))
            encoder = FoldEncoder(train, test, 3.0)
            np.testing.assert_allclose(encoder.kernel @ (targets - targets.mean(axis=0)) + targets.mean(axis=0),
                                       ridge_predict(np.concatenate(train), targets, test[0], 3.0), atol=1e-8)
            shifts, test_shift = [11, 13], 12
            expected = ridge_predict(np.concatenate([np.roll(matrix, shift, axis=0) for matrix, shift in zip(train, shifts)]),
                                     targets, np.roll(test[0], test_shift, axis=0), 3.0)
            kernel = encoder.shifted_kernel(shifts, [test_shift])
            np.testing.assert_allclose(kernel @ (targets - targets.mean(axis=0)) + targets.mean(axis=0), expected, atol=1e-8)

    def test_vertex_scorer_matches_reference(self):
        rng = np.random.default_rng(5)
        records = [{"family": "x", "actual": standardize(rng.normal(size=(30, 3))), "predicted": rng.normal(size=(30, 3)),
                    "indices": np.asarray([2, 7, 9])},
                   {"family": "y", "actual": standardize(rng.normal(size=(25, 2))), "predicted": rng.normal(size=(25, 2)),
                    "indices": np.asarray([7, 4])}]
        reference = [dict(record, predicted=standardize(record["predicted"])) for record in records]
        self.assertAlmostEqual(VertexScorer(records).score([record["predicted"] for record in records]),
                               score_records(reference), places=10)

    def test_bootstrap_matches_reference(self):
        rng = np.random.default_rng(6)
        def records():
            return [{"family": family, "actual": standardize(rng.normal(size=(length, 3))),
                     "predicted": standardize(rng.normal(size=(length, 3)))} for family, length in (("a", 40), ("a", 33), ("b", 37))]
        gpt, base = records(), records()
        np.testing.assert_allclose(bootstrap_delta(gpt, base, 40, np.random.default_rng(8)),
                                   bootstrap_delta_reference(gpt, base, 40, np.random.default_rng(8)), atol=1e-10)

    def test_mixed_cohorts_end_to_end(self):
        stories, networks = mixed_fixture()
        settings = dict(SETTINGS, selection_nulls=19, encoding_nulls=9, bootstrap_draws=19, rho_floor=-1, selection_q=1.1)
        result = analyze(stories, networks, settings, {})
        self.assertEqual(result["status"], "completed_primary_gates")
        self.assertEqual(len(result["folds"]), 4)
        self.assertTrue(all(len(values) == 9 for values in result["null_statistics"].values()))


if __name__ == "__main__":
    unittest.main()
