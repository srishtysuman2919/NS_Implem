"""Synthetic preparation-to-Stage-A smoke tests, never scientific results.

The real_narratives metadata value exercises the strict validation contract only.
All signals, atlas memberships, model/source hashes and certifications are test
fixtures, not real Narratives data, pretrained weights or scientific evidence.
Every generated input and output lives in a temporary directory.
"""

import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from narratives_features import REQUIRED_BASELINE, history_indices
from narratives_prepare import (
    Inputs, N_VERTICES, build_plan, cache_path, prepare, preprocess_segment,
    select_scans, sha256, validate_features,
)
from narratives_stage_a import BASELINES, analyze, load_dataset
from test_narratives_prepare import feature_fixture, record, save_gifti


TASKS = ("pieman", "tunnel", "lucy", "merlin")
SUBJECTS = tuple(f"sub-{index:03d}" for index in range(1, 5))
FIXTURE_NOTICE = (
    "Synthetic test fixture only: all source/model hashes, revision, alignment "
    "and baseline certifications are synthetic; real_narratives exercises "
    "validation and makes no scientific or pretrained-model claim."
)
SETTINGS = dict(
    seed=12, selection_nulls=39, encoding_nulls=2, subject_splits=1,
    bootstrap_draws=5, rho_floor=0.015, selection_q=0.05,
    minimum_r=0.04, gate_q=0.05, encoding_alpha=1.0, residual_alpha=1.0,
)


class NarrativesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="narratives-integration-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "prepared"
        cls.config_path = cls.root / "prepare.json"
        cls.manifest_path = cls.output / "dataset.json"
        cls.gate_path = cls.root / "gate-fixture.json"
        cls.event = dict(onset=2, duration=40)
        cls.indices = list(range(6, 38))
        cls.history = list(history_indices(cls.indices, 2, 40))
        cls.expected_bold = {}
        cls.feature_arrays = {}
        cls.make_fixture()

    @classmethod
    def write_json(cls, path, value):
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def make_fixture(cls):
        vertex_ids = np.arange(N_VERTICES)
        height = 1 - 2 * (vertex_ids + 0.5) / N_VERTICES
        angle = vertex_ids * np.pi * (3 - np.sqrt(5))
        radius = np.sqrt(1 - height ** 2)
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), height)).astype(np.float32)
        faces = ConvexHull(points).simplices.astype(np.int32)
        cls.centers = (500, 20000, 40000)
        cls.labels = np.zeros(N_VERTICES, dtype=np.int32)
        for parcel, center in enumerate(cls.centers, start=1):
            neighbors = np.unique(faces[np.any(faces == center, axis=1)])
            if np.any(cls.labels[neighbors]):
                raise AssertionError("Synthetic network patches must be disjoint")
            cls.labels[neighbors] = parcel
        cls.mask_vertices = np.flatnonzero(cls.labels)
        config = dict(
            schema_version=1, provenance="real_narratives", space="fsaverage6",
            test_fixture=FIXTURE_NOTICE, tasks=list(TASKS), cache_root="cache",
            trim_start=4, trim_end=4,
            metadata={name: f"{name}.json" for name in ("task_meta", "scan_exclude", "event_meta")},
            atlases={}, geometry={}, network_labels={}, membership_evidence="membership.json",
            feature_files={}, input_sha256={},
        )
        task_meta = {task: {subject: record(task, subject, runs=(1,)) for subject in SUBJECTS}
                     for task in TASKS}
        cls.write_json(cls.root / "task_meta.json", task_meta)
        cls.write_json(cls.root / "scan_exclude.json", {})
        cls.write_json(cls.root / "event_meta.json", {task: {task: cls.event} for task in TASKS})
        cls.write_json(cls.root / "membership.json", dict(source=FIXTURE_NOTICE))
        for hemi in ("L", "R"):
            config["atlases"][hemi] = f"{hemi}.label.gii"
            config["geometry"][hemi] = f"{hemi}.surf.gii"
            config["network_labels"][hemi] = dict(LANG=[1], MD=[2], EAC=[3])
            save_gifti(cls.root / f"{hemi}.label.gii", [cls.labels], [1002], hemi)
            save_gifti(cls.root / f"{hemi}.surf.gii", [points, faces], [1008, 1009], hemi)
        rng = np.random.default_rng(824)
        for task in TASKS:
            story = dict(name=task, event=cls.event, tr_indices=cls.indices)
            arrays = feature_fixture(story)
            metadata = json.loads(arrays["metadata"].item())
            metadata["test_fixture"] = FIXTURE_NOTICE
            metadata["alignment_provenance"]["source"] = FIXTURE_NOTICE
            metadata["baseline_provenance"]["unigram_reference"]["corpus"] = FIXTURE_NOTICE
            for detail in metadata["baseline_provenance"]["features"].values():
                detail.update(source=FIXTURE_NOTICE, method="synthetic fixture", units="test-only")
            metadata["source_hashes"] = {"synthetic-test-source.txt": next(iter(metadata["source_hashes"].values()))}
            metadata["lag_history"] = dict(
                lags_built=False, required_lags_tr=[1, 2, 3, 4], tr_indices=cls.history,
                edge_policy="drop_rows_without_real_in_story_history", unavailable_lags={},
            )
            full_gpt = np.zeros((40, 768), dtype=np.float32)
            full_gpt[:, :3] = rng.normal(size=(40, 3))
            full_baseline = rng.normal(size=(40, 9)).astype(np.float32)
            retained_rows = np.asarray(cls.indices) - cls.event["onset"]
            history_rows = np.asarray(cls.history) - cls.event["onset"]
            arrays.update(
                gpt2=full_gpt[retained_rows], baseline=full_baseline[retained_rows],
                gpt2_history=full_gpt[history_rows], baseline_history=full_baseline[history_rows],
                history_tr_indices=np.asarray(cls.history, dtype=np.int64),
                metadata=np.array(json.dumps(metadata)),
            )
            cls.feature_arrays[task] = arrays
            feature_path = cls.root / f"{task}-features.npz"
            np.savez_compressed(feature_path, **arrays)
            validate_features(feature_path, len(cls.indices), story)
            config["feature_files"][task] = feature_path.name
            shared = rng.normal(size=(44, len(cls.mask_vertices))).astype(np.float32)
            for hemi in ("L", "R"):
                expected = []
                for subject_index, subject in enumerate(SUBJECTS):
                    scan = select_scans({task: {subject: task_meta[task][subject]}}, {})[0][0]
                    data = np.zeros((44, N_VERTICES), dtype=np.float32)
                    data[:, cls.mask_vertices] = shared + rng.normal(scale=0.03, size=shared.shape)
                    path = cache_path(cls.root / "cache", scan, hemi)
                    if hemi == "L":
                        save_gifti(path, list(data), [2001] * len(data), hemi)
                    else:
                        save_gifti(path, [data.T.copy()], [2001], hemi)
                    expected.append(data[np.ix_(cls.indices, cls.mask_vertices)].copy())
                cls.expected_bold[task, hemi] = np.stack(expected)
        config["input_sha256"] = {
            path.relative_to(cls.root).as_posix(): sha256(path)
            for path in cls.root.rglob("*") if path.is_file()
        }
        cls.write_json(cls.config_path, config)
        inputs = Inputs(cls.config_path)
        cls.plan = build_plan(inputs)
        if cls.plan["blockers"]:
            raise AssertionError(cls.plan["blockers"])
        cls.manifest = prepare(inputs, cls.plan, cls.output)
        cls.gate_config = dict(
            test_fixture=FIXTURE_NOTICE,
            dataset=dict(primary_tasks=list(TASKS), trim_tr=dict(start=4, end=4)),
            model=dict(primary_layer=9, lags_tr=[1, 2, 3, 4]),
            residualization=dict(lang_lags_tr=[0, 1, 2]),
        )
        cls.write_json(cls.gate_path, cls.gate_config)

    def run_check_inputs(self, config=None):
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return subprocess.run(
            [sys.executable, "-B", str(Path(__file__).with_name("narratives_stage_a.py")),
             "--dataset", str(self.manifest_path), "--config", str(config or self.gate_path), "--check-inputs"],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=90,
            check=False,
        )

    @contextlib.contextmanager
    def modified_output(self, relative, metadata_change=None):
        path = self.output / relative
        original = path.read_bytes()
        original_manifest = self.manifest_path.read_bytes()
        try:
            if metadata_change is None:
                values = np.load(path, allow_pickle=False)
                values[0, 0, 0] += 0.5
                np.save(path, values)
            else:
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                metadata = json.loads(arrays["metadata"].item())
                metadata.update(metadata_change)
                arrays["metadata"] = np.array(json.dumps(metadata))
                np.savez_compressed(path, **arrays)
                manifest = copy.deepcopy(self.manifest)
                manifest["output_hashes"][relative] = sha256(path)
                self.write_json(self.manifest_path, manifest)
            yield
        finally:
            path.write_bytes(original)
            self.manifest_path.write_bytes(original_manifest)

    def test_prepared_strict_archives_and_loader_alignment(self):
        self.assertEqual(self.plan["blockers"], [])
        self.assertEqual(self.manifest["status"], "prepared")
        self.assertTrue(self.manifest["primary_eligible"])
        self.assertTrue(self.manifest["feature_metadata_strict"])
        self.assertEqual(self.manifest["boundary_rings"], 1)
        self.assertLess(len(self.mask_vertices), 40)
        self.assertEqual(self.manifest["config_sha256"], sha256(self.config_path))
        pins = json.loads(self.config_path.read_text(encoding="utf-8"))["input_sha256"]
        self.assertEqual(self.manifest["input_hashes"],
                         {(self.root / name).resolve().as_posix(): digest for name, digest in pins.items()})
        for relative, digest in self.manifest["output_hashes"].items():
            self.assertEqual(sha256(self.output / relative), digest)
        targets = self.manifest["targets"]
        for hemi in ("L", "R"):
            for center, network in zip(self.centers, ("LANG", "MD", "EAC")):
                self.assertIn((hemi, center, network),
                              [(target["hemisphere"], target["vertex_id"], target["network"]) for target in targets])
        self.assertLess(len(targets), 2 * len(self.mask_vertices))
        stories, networks, manifest = load_dataset(self.manifest_path, self.gate_config)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual({story.family for story in stories}, set(TASKS))
        self.assertEqual(set(networks), {"LANG", "MD", "EAC"})
        for story in stories:
            self.assertEqual(story.subject_ids, SUBJECTS)
            self.assertEqual(story.subjects.shape, (4, 32, len(targets)))
            self.assertEqual(story.subjects.dtype, np.float32)
            self.assertEqual(story.gpt2.shape, (32, 3072))
            self.assertEqual(story.baseline.shape, (32, 36))
            np.testing.assert_array_equal(story.parcel_ids,
                [f"{target['hemisphere']}:{target['parcel']}" for target in targets])
            np.testing.assert_allclose(story.subjects.mean(axis=1), 0, atol=1e-6)
            np.testing.assert_allclose(story.subjects.std(axis=1), 1, atol=1e-6)
            for column, target in enumerate(targets):
                self.assertEqual(target["parcel"], int(self.labels[target["vertex_id"]]))
                position = np.flatnonzero(self.mask_vertices == target["vertex_id"])[0]
                for subject_index in range(4):
                    raw = self.expected_bold[story.name, target["hemisphere"]][subject_index, :, position, None]
                    expected = preprocess_segment(raw, list(range(32)), [0])[:, 0]
                    np.testing.assert_allclose(story.subjects[subject_index, :, column], expected, atol=1e-6)
            arrays = self.feature_arrays[story.name]
            self.assertEqual(set(arrays["baseline_names"]), set(REQUIRED_BASELINE))
            self.assertEqual(arrays["baseline"].shape, (32, 9))
            self.assertEqual(arrays["gpt2_history"].shape, (4, 768))
            self.assertEqual(arrays["baseline_history"].shape, (4, 9))
            np.testing.assert_array_equal(arrays["history_tr_indices"], [2, 3, 4, 5])
            all_indices = self.indices + self.history
            lookup = {source: index for index, source in enumerate(all_indices)}
            for lag_index, lag in enumerate((1, 2, 3, 4)):
                rows = [lookup[source - lag] for source in self.indices]
                expected_gpt = np.concatenate((arrays["gpt2"], arrays["gpt2_history"]))[rows]
                expected_baseline = np.concatenate((arrays["baseline"], arrays["baseline_history"]))[rows]
                order = [arrays["baseline_names"].tolist().index(name) for name in BASELINES]
                np.testing.assert_array_equal(story.gpt2[:, lag_index * 768:(lag_index + 1) * 768], expected_gpt)
                np.testing.assert_array_equal(story.baseline[:, lag_index * 9:(lag_index + 1) * 9], expected_baseline[:, order])

    def test_analyze_completes_smoke_not_scientific_inference(self):
        stories, networks, _ = load_dataset(self.manifest_path, self.gate_config)
        result = analyze(stories, networks, dict(SETTINGS), {})
        self.assertEqual(result["status"], "completed_primary_gates")
        self.assertEqual(len(result["folds"]), 4)
        for fold in result["folds"]:
            self.assertEqual(fold["status"], "ready")
            self.assertEqual(fold["test_stories"], [fold["family"]])
            self.assertEqual(set(fold["training_stories"]), set(TASKS) - {fold["family"]})
            self.assertTrue(all(count > 0 for count in fold["selected_counts"].values()))
        for null in result["null_statistics"].values():
            self.assertEqual(len(null), SETTINGS["encoding_nulls"])
            self.assertTrue(np.isfinite(null).all())
        self.assertTrue(np.isfinite(result["gate_3"]["ci95"]).all())
        json.dumps(result, allow_nan=False)

    def test_cli_check_inputs_accepts_prepared_fixture_without_results(self):
        before = {path.relative_to(self.root) for path in self.root.rglob("*") if path.is_file()}
        completed = self.run_check_inputs()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), dict(
            status="inputs_validated_not_run", stories=4, vertices=len(self.manifest["targets"])))
        self.assertEqual(completed.stderr, "")
        self.assertEqual(before, {path.relative_to(self.root) for path in self.root.rglob("*") if path.is_file()})

    def test_each_primary_network_requires_bilateral_support(self):
        original = self.manifest_path.read_bytes()
        try:
            for network in ("LANG", "MD"):
                for hemisphere in ("L", "R"):
                    altered = copy.deepcopy(self.manifest)
                    altered["targets"] = [target for target in altered["targets"]
                                          if (target["network"], target["hemisphere"]) != (network, hemisphere)]
                    self.write_json(self.manifest_path, altered)
                    with self.subTest(network=network, hemisphere=hemisphere):
                        with self.assertRaisesRegex(ValueError, "Both hemispheres required"):
                            load_dataset(self.manifest_path, self.gate_config)
        finally:
            self.manifest_path.write_bytes(original)

    def test_cli_rejects_wrong_story_even_with_updated_archive_hash(self):
        with self.modified_output("features/pieman.npz", dict(story="synthetic-wrong-story")):
            completed = self.run_check_inputs()
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Strict feature metadata mismatch: story", completed.stderr)
            self.assertNotIn("inputs_validated_not_run", completed.stdout)

    def test_cli_rejects_tampered_bold_hash(self):
        with self.modified_output("bold/pieman.npy"):
            completed = self.run_check_inputs()
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Missing or unverified bold: pieman", completed.stderr)
            self.assertNotIn("inputs_validated_not_run", completed.stdout)

    def test_cli_rejects_unsupported_layer(self):
        with self.modified_output("features/pieman.npz", dict(layer=8)):
            completed = self.run_check_inputs()
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Strict feature metadata mismatch: layer", completed.stderr)
        config = copy.deepcopy(self.gate_config)
        config["model"]["primary_layer"] = 8
        path = self.root / "unsupported-layer.json"
        self.write_json(path, config)
        self.addCleanup(path.unlink)
        completed = self.run_check_inputs(path)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Feature semantics or timing mismatch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
