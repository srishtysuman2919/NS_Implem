"""Synthetic real-format tests; no real-data completion is asserted."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull

from narratives_prepare import (
    Inputs, N_VERTICES, PreparationError, build_plan, cache_path, load_surface,
    load_triangles, main, network_targets, prepare, preprocess_segment,
    resolve_segments, segment_indices, select_scans, sha256, validate_features,
)


def record(task="pieman", subject="sub-001", runs=(1, 2), condition="n/a"):
    paths = []
    for run in reversed(runs):
        suffix = "" if run is None else f"_run-{run}"
        for hemi in ("R", "L"):
            paths.append(f"/original/{subject}_task-{task}{suffix}_space-fsaverage6_hemi-{hemi}.func.gii")
    return {"bold": {"fsaverage6": {"preproc": paths}}, "condition": condition}


class SelectionTests(unittest.TestCase):
    def test_excluded_first_is_not_replaced_by_repeat(self):
        meta = {"pieman": {"sub-001": record()}}
        selected, rejected = select_scans(meta, {"pieman": {"sub-001": ["sub-001_task-pieman_run-1"]}})
        self.assertEqual(selected, [])
        self.assertEqual(rejected[0]["run"], 1)

    def test_exclusions_are_subject_specific(self):
        meta = {"pieman": {"sub-001": record(), "sub-002": record(subject="sub-002")}}
        selected, _ = select_scans(meta, {"pieman": {"sub-002": ["sub-001_task-pieman_run-1"]}})
        self.assertEqual([scan["subject"] for scan in selected], ["sub-001", "sub-002"])

    def test_task_exclusion_covers_numbered_runs(self):
        selected, _ = select_scans({"pieman": {"sub-001": record()}},
                                   {"pieman": {"sub-001": ["sub-001_task-pieman"]}})
        self.assertEqual(selected, [])

    def test_schema_runs_are_distinct_stories(self):
        selected, _ = select_scans({"schema": {"sub-001": record(task="schema", runs=(2, 1, 8))}}, {})
        self.assertEqual([scan["run"] for scan in selected], [1, 2, 8])

    def test_independent_segment_trimming(self):
        self.assertEqual(segment_indices({"onset": 20, "duration": 10}, 2, 3), list(range(22, 27)))
        self.assertEqual(segment_indices({"onset": 646, "duration": 8}, 2, 3), [648, 649, 650])
        with self.assertRaises(PreparationError):
            segment_indices({"onset": 1.5, "duration": 8}, 0, 0)

    def test_later_repeat_exclusion_does_not_remove_first(self):
        scans, _ = select_scans({"pieman": {"sub-001": record()}},
                                {"pieman": {"sub-001": ["sub-001_task-pieman_run-2"]}})
        self.assertEqual(scans[0]["run"], 1)

    def test_no_run_and_numbered_run_is_ambiguous(self):
        with self.assertRaisesRegex(PreparationError, "Ambiguous exposure"):
            select_scans({"pieman": {"sub-001": record(runs=(None, 1))}}, {})

    def test_missing_first_hemisphere_never_uses_repeat(self):
        entry = record()
        entry["bold"]["fsaverage6"]["preproc"] = [path for path in entry["bold"]["fsaverage6"]["preproc"]
                                                        if "run-1_space-fsaverage6_hemi-L" not in path]
        with self.assertRaisesRegex(PreparationError, "Both hemispheres"):
            select_scans({"pieman": {"sub-001": entry}}, {})

    def test_repository_metadata_first_run_and_schema(self):
        root = Path(__file__).resolve().parents[1] / "data" / "narratives_meta"
        meta = json.loads((root / "task_meta.json").read_text())
        exclusions = json.loads((root / "scan_exclude.json").read_text())
        selected, rejected = select_scans(meta, exclusions)
        self.assertFalse(any(scan["task"] == "pieman" and scan["subject"] == "sub-001" for scan in selected))
        self.assertTrue(any(scan["task"] == "pieman" and scan["subject"] == "sub-001" and scan["run"] == 1 for scan in rejected))
        schema = [scan for scan in selected if scan["task"] == "schema" and scan["subject"] == "sub-049"]
        self.assertEqual([scan["run"] for scan in schema], [1, 2, 3, 4])


class SegmentTests(unittest.TestCase):
    def scan(self, task, condition="n/a", run=None):
        return select_scans({task: {"sub-001": record(task=task, runs=(run,), condition=condition)}}, {})[0][0]

    def test_actual_slumlordreach_events_stay_separate(self):
        path = Path(__file__).resolve().parents[1] / "data/narratives_meta/event_meta.json"
        segments = resolve_segments(self.scan("slumlordreach"), json.loads(path.read_text()), {}, 5, 5)
        by_name = {segment["name"]: segment for segment in segments}
        self.assertEqual(by_name["slumlordreach_slumlord"]["tr_indices"], list(range(25, 617)))
        self.assertEqual(by_name["slumlordreach_reach"]["tr_indices"], list(range(651, 1191)))
        self.assertEqual(by_name["slumlordreach_reach"]["event_ref"], ["slumlordreach", "reach"])

    def test_all_conditions_require_explicit_mapping(self):
        for task, conditions in (("milkyway", ("original", "vodka", "synonyms")),
                                  ("prettymouth", ("affair", "paranoia"))):
            for condition in conditions:
                with self.subTest(task=task, condition=condition):
                    scan = self.scan(task, condition)
                    events = {task: {task: {"onset": 14, "duration": 269}}}
                    with self.assertRaisesRegex(PreparationError, "Explicit event mapping"):
                        resolve_segments(scan, events, {}, 0, 0)
                    mappings = {f"{task}/{condition}": [{"name": f"{task}_{condition}", "family": task,
                                                       "event_ref": [task, task], "evidence": "condition_events.tsv"}]}
                    result = resolve_segments(scan, events, mappings, 2, 3)
                    self.assertEqual(result[0]["tr_indices"], list(range(16, 280)))

    def test_schema_needs_subject_specific_identity_and_evidence(self):
        scan = self.scan("schema", run=2)
        mapping = {"schema/run-2": [{"name": "story", "family": "schema", "onset": 0, "duration": 10,
                                     "units": "TR", "evidence": "events.tsv"}]}
        with self.assertRaisesRegex(PreparationError, "Explicit event mapping"):
            resolve_segments(scan, {}, mapping, 0, 0)
        mapping[scan["scan_id"]] = mapping.pop("schema/run-2")
        self.assertEqual(resolve_segments(scan, {}, mapping, 0, 0)[0]["name"], "story")
        del mapping[scan["scan_id"]][0]["evidence"]
        with self.assertRaisesRegex(PreparationError, "requires evidence"):
            resolve_segments(scan, {}, mapping, 0, 0)

    def test_variant_timing_cannot_be_invented_or_conflicted(self):
        scan = self.scan("milkyway", "vodka")
        segment = {"name": "vodka", "family": "milkyway", "onset": 0, "duration": 10}
        mapping = {"milkyway/vodka": [segment]}
        with self.assertRaisesRegex(PreparationError, "requires evidence"):
            resolve_segments(scan, {}, mapping, 0, 0)
        segment.update(units="TR", evidence="timing.tsv", event_ref=["milkyway", "milkyway"])
        with self.assertRaisesRegex(PreparationError, "Conflicting"):
            resolve_segments(scan, {}, mapping, 0, 0)

    def test_overlapping_segments_rejected(self):
        scan = self.scan("slumlordreach")
        events = {"slumlordreach": {"slumlord": {"onset": 0, "duration": 10},
                                    "reach": {"onset": 9, "duration": 10}}}
        with self.assertRaisesRegex(PreparationError, "Overlapping"):
            resolve_segments(scan, events, {}, 0, 0)

    def test_family_aliases_and_unknown_conditions(self):
        events = {"piemanpni": {"piemanpni": {"onset": 0, "duration": 10}}}
        segment = resolve_segments(self.scan("piemanpni"), events, {}, 0, 0)[0]
        self.assertEqual(segment["family"], "pieman")
        with self.assertRaisesRegex(PreparationError, "Unknown condition"):
            resolve_segments(self.scan("milkyway", "unknown"), {}, {}, 0, 0)


def save_gifti(path, arrays, intents, hemisphere=None):
    image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data=array, intent=intent)
                                         for array, intent in zip(arrays, intents)])
    if hemisphere:
        image.meta["AnatomicalStructurePrimary"] = {"L": "CortexLeft", "R": "CortexRight"}[hemisphere]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


class GiftiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "surface.func.gii"
        self.data = np.random.default_rng(42).normal(size=(12, 6)).astype(np.float32)

    def test_vector_per_tr_and_both_matrix_orientations(self):
        for arrays in ([row for row in self.data], [self.data], [self.data.T.copy()]):
            with self.subTest(arrays=len(arrays)):
                save_gifti(self.path, arrays, [2001] * len(arrays), "L")
                np.testing.assert_array_equal(load_surface(self.path, "L", 6), self.data)

    def test_shape_intent_hemisphere_and_finiteness_rejected(self):
        invalid = [([self.data[:, :5]], [2001], "L"),
                   ([self.data[:6]], [2001], "L"),
                   ([self.data], [0], "R"),
                   ([np.full((12, 6), np.nan, dtype=np.float32)], [2001], "L"),
                   ([self.data], [1008], "L")]
        for arrays, intents, hemi in invalid:
            save_gifti(self.path, arrays, intents, hemi)
            with self.assertRaises(PreparationError):
                load_surface(self.path, "L", 6)

    def test_zero_variance_and_detrended_linear_targets_rejected(self):
        for values in (np.ones((12, 6)), np.tile(np.arange(12)[:, None], (1, 6))):
            with self.assertRaisesRegex(PreparationError, "variance"):
                preprocess_segment(values, list(range(10)), [0, 1])

    def test_trim_and_detrend_are_within_segment(self):
        segment = preprocess_segment(self.data, list(range(2, 10)), [0, 4])
        changed = self.data.copy()
        changed[:2] = 10000
        changed[10:] = -20000
        np.testing.assert_array_equal(preprocess_segment(changed, list(range(2, 10)), [0, 4]), segment)
        np.testing.assert_allclose(segment.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(segment.std(axis=0), 1, atol=1e-7)
        np.testing.assert_allclose(np.arange(8) @ segment, 0, atol=1e-6)
        with self.assertRaisesRegex(PreparationError, "exceeds"):
            preprocess_segment(self.data, list(range(2, 10)), [0], original_stop=13)

    def test_real_surface_pointset_and_triangle_intents(self):
        points = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        save_gifti(self.path, [points, faces], [1008, 1009], "L")
        np.testing.assert_array_equal(load_triangles(self.path, "L", 4), faces)
        save_gifti(self.path, [points, faces[:-1]], [1008, 1009], "L")
        with self.assertRaisesRegex(PreparationError, "closed"):
            load_triangles(self.path, "L", 4)


class NetworkTests(unittest.TestCase):
    def test_disjointness_required_before_erosion(self):
        with self.assertRaisesRegex(PreparationError, "Overlapping"):
            network_targets(np.array([1, 2, 3]), {"LANG": [1], "MD": [1, 2], "EAC": [3]}, boundary_rings=0)

    def test_original_vertex_ids_and_one_ring_background_removal(self):
        labels = np.array([1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0])
        faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10], [9, 10, 11]])
        membership = {"LANG": [1], "MD": [2], "EAC": [3]}
        targets = network_targets(labels, membership, faces)
        self.assertEqual([target["vertex_id"] for target in targets], [0, 4, 8])
        self.assertEqual(len(network_targets(labels, membership, boundary_rings=0)), 9)
        with self.assertRaisesRegex(PreparationError, "real triangle"):
            network_targets(labels, membership)


class FeatureTests(unittest.TestCase):
    def test_strict_emitted_metadata_contract_and_rejections(self):
        story = dict(name="pieman", event=dict(onset=2, duration=12), tr_indices=list(range(3, 13)))
        arrays = feature_fixture(story)
        metadata = json.loads(arrays["metadata"].item())
        mutations = [
            ("story", "other"), ("story_onset_tr", 3), ("story_duration_tr", 13),
            ("tr_indices", list(range(4, 14))), ("tr_indices", [float(index) for index in range(3, 13)]),
            ("provenance", "synthetic"), ("model", "distilgpt2"), ("layer", 8), ("layer", True),
            ("context_window_tr", 4), ("tr_seconds", 2), ("causal_windows", 1),
            ("full_story_alignment", False), ("timestamp_offset_seconds", float("nan")),
            ("alignment_provenance", dict(metadata["alignment_provenance"], story="other")),
            ("alignment_provenance", dict(metadata["alignment_provenance"], full_story=False)),
            ("alignment_provenance", dict(metadata["alignment_provenance"], coverage_stop_seconds=17)),
            ("baseline_provenance", dict(metadata["baseline_provenance"], story="other")),
            ("baseline_provenance", dict(metadata["baseline_provenance"], causal_windows=False)),
            ("baseline_provenance", dict(metadata["baseline_provenance"], features={})),
            ("source_hashes", {}), ("source_hashes", {"words.csv": "not-a-hash"}),
            ("model_revision", "main"), ("model_file_hashes", {}),
            ("model_weights_sha256", "a" * 64), ("config_sha256", None),
        ]
        baseline = copy.deepcopy(metadata["baseline_provenance"])
        baseline["unigram_reference"]["estimated_on_narratives"] = True
        mutations.append(("baseline_provenance", baseline))
        baseline = copy.deepcopy(metadata["baseline_provenance"])
        baseline["features"]["pitch"]["source_sha256"] = "invalid"
        mutations.append(("baseline_provenance", baseline))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            np.savez(path, **arrays)
            validate_features(path, 10, story)
            sharded = copy.deepcopy(metadata)
            files = sharded["model_file_hashes"]
            shard_digest = files.pop("model.safetensors")
            files.update({"model.safetensors.index.json": "b" * 64,
                          "model-00001-of-00001.safetensors": shard_digest,
                          "unused.safetensors": "c" * 64})
            selected = {name: files[name] for name in ("model.safetensors.index.json",
                                                       "model-00001-of-00001.safetensors")}
            sharded["model_weights_sha256"] = hashlib.sha256(json.dumps(selected, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            np.savez(path, **dict(arrays, metadata=np.array(json.dumps(sharded))))
            validate_features(path, 10, story)
            for key in metadata:
                with self.subTest(missing=key):
                    missing = dict(metadata)
                    del missing[key]
                    np.savez(path, **dict(arrays, metadata=np.array(json.dumps(missing))))
                    with self.assertRaises(PreparationError):
                        validate_features(path, 10, story)
            for key, value in mutations:
                with self.subTest(field=key, value=value):
                    np.savez(path, **dict(arrays, metadata=np.array(json.dumps(dict(metadata, **{key: value})))))
                    with self.assertRaises(PreparationError):
                        validate_features(path, 10, story)

    def test_feature_contract_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            arrays = dict(gpt2=np.zeros((8, 768)), baseline=np.zeros((8, 2)),
                          baseline_names=np.array(["word_count", "audio"]), metadata=np.array('{"source":"fixture"}'))
            np.savez(path, **arrays)
            validate_features(path, 8)
            for replacement in (dict(gpt2=np.zeros((8, 767))), dict(baseline=np.zeros((7, 2))),
                                dict(baseline_names=np.array(["same", "same"])),
                                dict(baseline_names=np.array(["a", "b"], dtype=object)),
                                dict(gpt2=np.full((8, 768), np.inf)), dict(metadata=np.array("[]"))):
                np.savez(path, **dict(arrays, **replacement))
                with self.assertRaises((PreparationError, ValueError)):
                    validate_features(path, 8)


def feature_fixture(story):
    names = ["phoneme_or_syllable_rate", "pitch", "energy", "word_frequency", "sentence_boundary",
             "voiced_fraction", "word_rate", "word_length", "unigram_surprisal"]
    source_digest = hashlib.sha256(b"synthetic test source, not real data").hexdigest()
    weights = {"model.safetensors": hashlib.sha256(b"synthetic test weights").hexdigest()}
    baseline = dict(story=story["name"], provenance="real_narratives", tr_seconds=1.5,
                    causal_windows=True, features={name: dict(source="fixture", source_sha256=source_digest,
                    method="fixture", units="fixture", causal=True) for name in names},
                    unigram_reference=dict(corpus="independent fixture", source_sha256=source_digest,
                    independent_of_evaluation_corpus=True, estimated_on_narratives=False))
    metadata = dict(schema_version=1, model="gpt2", layer=9, context_window_tr=5, tr_seconds=1.5,
                    tr_indices=story["tr_indices"], provenance="real_narratives", causal_windows=True,
                    story=story["name"], story_onset_tr=story["event"]["onset"],
                    story_duration_tr=story["event"]["duration"], word_availability="word_end",
                    window_interval="(start,end]", timestamp_offset_seconds=0, full_story_alignment=True,
                    alignment_provenance=dict(certified=True, full_story=True, story=story["name"],
                    timebase="story_relative_seconds", offset_seconds=0, coverage_start_seconds=0,
                    coverage_stop_seconds=story["event"]["duration"] * 1.5, source="fixture"),
                    model_revision="a" * 40,
                    model_weights_sha256=hashlib.sha256(json.dumps(weights, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")).hexdigest(),
                    model_file_hashes=dict(weights, **{"config.json": source_digest}),
                    source_hashes={"words.csv": source_digest}, config_sha256=source_digest,
                    baseline_provenance=baseline, pooling="mean_all_attended_tokens", padding="none",
                    empty_context_policy="zero_vector", token_overflow_policy="error")
    n_time = len(story["tr_indices"])
    return dict(gpt2=np.zeros((n_time, 768), dtype=np.float32),
                baseline=np.zeros((n_time, len(names)), dtype=np.float32), baseline_names=np.array(names),
                metadata=np.array(json.dumps(metadata)))


class PreparationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "prepare.json"
        self.output = self.root / "output"
        self.config = dict(schema_version=1, provenance="real_narratives", space="fsaverage6",
                           tasks=["pieman"], cache_root="cache", trim_start=1, trim_end=1,
                           metadata={name: f"{name}.json" for name in ("task_meta", "scan_exclude", "event_meta")},
                           atlases={}, geometry={}, network_labels={}, membership_evidence="membership.json",
                           feature_files={}, input_sha256={})
        self.write_json("task_meta.json", {"pieman": {"sub-001": record(runs=(1,))}})
        self.write_json("scan_exclude.json", {})
        self.write_json("event_meta.json", {"pieman": {"pieman": {"onset": 2, "duration": 12}}})
        self.write_json("membership.json", {"source": "synthetic fixture, never actual atlas memberships"})
        vertex = np.arange(N_VERTICES)
        height = 1 - 2 * (vertex + 0.5) / N_VERTICES
        angle = vertex * np.pi * (3 - np.sqrt(5))
        radius = np.sqrt(1 - height ** 2)
        points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), height]).astype(np.float32)
        self.faces = ConvexHull(points).simplices.astype(np.int32)
        self.labels = np.where(height < -0.3, 1, np.where(height > 0.3, 3, 2)).astype(np.int32)
        scan = select_scans({"pieman": {"sub-001": record(runs=(1,))}}, {})[0][0]
        self.scan = scan
        data = np.random.default_rng(21).normal(size=(16, N_VERTICES)).astype(np.float32)
        for hemi in ("L", "R"):
            self.config["atlases"][hemi] = f"{hemi}.label.gii"
            self.config["geometry"][hemi] = f"{hemi}.surf.gii"
            self.config["network_labels"][hemi] = {"LANG": [1], "MD": [2], "EAC": [3]}
            save_gifti(self.root / f"{hemi}.label.gii", [self.labels], [1002], hemi)
            save_gifti(self.root / f"{hemi}.surf.gii", [points, self.faces], [1008, 1009], hemi)
            save_gifti(cache_path(self.root / "cache", scan, hemi), [data], [2001], hemi)
        self.pin_all()

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def pin_all(self):
        self.config["input_sha256"] = {path.relative_to(self.root).as_posix(): sha256(path)
                                        for path in self.root.rglob("*") if path.is_file() and path != self.config_path}
        self.write_json("prepare.json", self.config)

    def test_primary_real_format_pipeline_contract(self):
        source_story = dict(name="pieman", event=dict(onset=2, duration=12), tr_indices=list(range(3, 13)))
        np.savez(self.root / "features.npz", **feature_fixture(source_story))
        self.config["feature_files"] = {"pieman": "features.npz"}
        self.pin_all()
        inputs = Inputs(self.config_path)
        plan = build_plan(inputs)
        self.assertEqual(plan["blockers"], [])
        manifest = prepare(inputs, plan, self.output)
        self.assertTrue(manifest["primary_eligible"])
        self.assertTrue(manifest["feature_metadata_strict"])
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["boundary_rings"], 1)
        story = manifest["stories"][0]
        self.assertEqual(story["tr_indices"], list(range(3, 13)))
        self.assertEqual(story["subject_ids"], ["sub-001"])
        self.assertEqual(story["family"], "pieman")
        self.assertEqual((story["trim_start"], story["trim_end"], story["tr_seconds"]), (1, 1, 1.5))
        self.assertEqual(story["preprocessing"], "segment_linear_detrend_population_zscore")
        for target in manifest["targets"]:
            self.assertIs(type(target["parcel"]), int)
            self.assertEqual(target["parcel"], int(self.labels[target["vertex_id"]]))
        self.assertEqual({target["hemisphere"] for target in manifest["targets"]}, {"L", "R"})
        self.assertEqual({target["network"] for target in manifest["targets"]}, {"LANG", "MD", "EAC"})
        self.assertEqual(manifest["config_sha256"], sha256(self.config_path))
        bold = np.load(self.output / story["bold"], allow_pickle=False)
        self.assertEqual(bold.shape, (1, 10, len(manifest["targets"])))
        self.assertEqual(bold.dtype, np.float32)
        np.testing.assert_allclose(bold.mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(bold.std(axis=1), 1, atol=1e-6)
        for path, digest in manifest["output_hashes"].items():
            self.assertEqual(sha256(self.output / path), digest)
        self.assertEqual(json.loads((self.output / "dataset.json").read_text()), manifest)
        with self.assertRaisesRegex(PreparationError, "already exists"):
            prepare(inputs, plan, self.output)

    def test_invalid_feature_associations_never_publish(self):
        story = dict(name="pieman", event=dict(onset=2, duration=12), tr_indices=list(range(3, 13)))
        valid = feature_fixture(story)
        metadata = json.loads(valid["metadata"].item())
        invalid_metadata = [None, dict(metadata, story="other"), dict(metadata, story_onset_tr=1),
                            dict(metadata, story_duration_tr=11), dict(metadata, tr_indices=list(range(4, 14)))]
        self.config["feature_files"] = {"pieman": "features.npz"}
        for invalid in invalid_metadata:
            with self.subTest(metadata=invalid):
                arrays = dict(valid)
                if invalid is None:
                    del arrays["metadata"]
                else:
                    arrays["metadata"] = np.array(json.dumps(invalid))
                np.savez(self.root / "features.npz", **arrays)
                self.pin_all()
                inputs = Inputs(self.config_path)
                plan = build_plan(inputs)
                self.assertTrue(any("pieman features:" in blocker for blocker in plan["blockers"]))
                with self.assertRaises(PreparationError):
                    prepare(inputs, plan, self.output)
                self.assertFalse(self.output.exists())

    def test_configured_trims_must_match_plan_and_features(self):
        inputs = Inputs(self.config_path)
        plan = build_plan(inputs, zero_ring=True)
        inputs.config["trim_start"] = 2
        with self.assertRaisesRegex(PreparationError, "configured trims"):
            prepare(inputs, plan, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_geometry_blocks_primary_but_zero_ring_is_explicit(self):
        self.config.pop("geometry")
        self.pin_all()
        inputs = Inputs(self.config_path)
        self.assertTrue(any("geometry" in blocker for blocker in build_plan(inputs)["blockers"]))
        plan = build_plan(inputs, zero_ring=True)
        self.assertEqual(plan["blockers"], [])
        manifest = prepare(inputs, plan, self.output)
        self.assertFalse(manifest["primary_eligible"])
        self.assertFalse(manifest["feature_metadata_strict"])
        self.assertEqual(manifest["analysis_tier"], "exploratory_zero_ring")
        self.assertEqual(manifest["status"], "bold_prepared_features_pending")
        self.assertFalse((self.output / manifest["stories"][0]["features"]).exists())

    def test_provenance_changed_missing_and_unpinned_files_block(self):
        path = cache_path(self.root / "cache", self.scan, "R")
        key = path.relative_to(self.root).as_posix()
        self.config["input_sha256"][key] = "0" * 64
        self.write_json("prepare.json", self.config)
        self.assertTrue(any("SHA256 mismatch" in item for item in build_plan(Inputs(self.config_path))["blockers"]))
        del self.config["input_sha256"][key]
        self.write_json("prepare.json", self.config)
        self.assertTrue(any("Missing input_sha256" in item for item in build_plan(Inputs(self.config_path))["blockers"]))
        path.unlink()
        self.assertTrue(any("Missing local input" in item for item in build_plan(Inputs(self.config_path))["blockers"]))

    def test_mismatched_hemi_time_does_not_publish_partial_dataset(self):
        path = cache_path(self.root / "cache", self.scan, "R")
        data = np.random.default_rng(3).normal(size=(15, N_VERTICES)).astype(np.float32)
        save_gifti(path, [data], [2001], "R")
        self.pin_all()
        inputs = Inputs(self.config_path)
        with self.assertRaisesRegex(PreparationError, "Hemisphere time mismatch"):
            prepare(inputs, build_plan(inputs, zero_ring=True), self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("narratives-*")), [])

    def test_empty_selection_is_not_complete(self):
        self.write_json("scan_exclude.json", {"pieman": {"sub-001": ["sub-001_task-pieman_run-1"]}})
        self.pin_all()
        plan = build_plan(Inputs(self.config_path), zero_ring=True)
        self.assertTrue(any("No resolvable story" in item for item in plan["blockers"]))
        with self.assertRaises(PreparationError):
            prepare(Inputs(self.config_path), plan, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_inventory_writes_nothing(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            status = main(["--config", str(self.config_path), "--inventory", "--exploratory-zero-ring"])
        report = json.loads(capture.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "inputs_planned_not_prepared")
        self.assertFalse(report["primary_eligible"])
        self.assertEqual(report["features_pending"], ["pieman"])
        self.assertFalse(self.output.exists())

    def test_schema_excluded_first_stimulus_never_uses_later_run(self):
        self.config["tasks"] = ["schema"]
        self.write_json("task_meta.json", {"schema": {"sub-001": record(task="schema", runs=(1, 2))}})
        self.write_json("scan_exclude.json", {"schema": {"sub-001": ["sub-001_task-schema_run-1"]}})
        segment = dict(name="same_stimulus", family="schema", onset=0, duration=10,
                       units="TR", evidence="schema_events.json")
        self.write_json("schema_events.json", {"source": "synthetic stimulus identity evidence"})
        self.config["event_mappings"] = {f"sub-001_task-schema_run-{run}": [segment] for run in (1, 2)}
        self.pin_all()
        plan = build_plan(Inputs(self.config_path), zero_ring=True)
        self.assertEqual(plan["stories"], {})
        self.assertTrue(any("No resolvable story" in blocker for blocker in plan["blockers"]))

    def test_schema_unknown_excluded_identity_blocks_repeat_safety(self):
        self.config["tasks"] = ["schema"]
        self.write_json("task_meta.json", {"schema": {"sub-001": record(task="schema", runs=(1, 2))}})
        self.write_json("scan_exclude.json", {"schema": {"sub-001": ["sub-001_task-schema_run-1"]}})
        self.pin_all()
        plan = build_plan(Inputs(self.config_path), zero_ring=True)
        self.assertTrue(any("run-1 pre-exclusion schema identity" in blocker for blocker in plan["blockers"]))


if __name__ == "__main__":
    unittest.main()
