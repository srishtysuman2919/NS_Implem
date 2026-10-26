"""Local-only, fail-closed Narratives surface-vertex preparation.

Run --inventory without a config for a metadata/cache blocker summary. Preparation
requires --config and --output (a new directory). All config paths are relative to
the config file, not the working directory. Nothing is downloaded.

Preparation config (separate from the existing gate config)::

    {"schema_version": 1, "provenance": "real_narratives",
     "space": "fsaverage6", "tasks": ["pieman"],
     "metadata": {"task_meta": "../data/narratives_meta/task_meta.json",
                  "scan_exclude": "../data/narratives_meta/scan_exclude.json",
                  "event_meta": "../data/narratives_meta/event_meta.json"},
     "cache_root": "../data/narratives_meta/processed_cache",
     "atlases": {"L": "left.label.gii", "R": "right.label.gii"},
     "geometry": {"L": "left.surf.gii", "R": "right.surf.gii"},
     "network_labels": {"L": {"LANG": [1], "MD": [2], "EAC": [3]},
                        "R": {"LANG": [181], "MD": [182], "EAC": [183]}},
    "membership_evidence": "memberships_source.json",
    "trim_start": 5, "trim_end": 5, "event_mappings": {},
     "feature_files": {}, "input_sha256": {"relative/input/path": "64 hex digits"}}

The example parcel numbers are placeholders, NOT scientific network assignments.
Supply independently justified memberships and hash their source using the
required membership_evidence path. Pin EVERY consumed input in input_sha256,
including metadata, evidence, meshes, atlases, clean GIFTIs and supplied features.
The config itself is hashed separately. Paths for cached GIFTIs are exactly
cache_root/afni-nosmooth/sub-XXX/<scan>_space-fsaverage6_hemi-L_desc-clean.func.gii.

event_mappings keys are exact scan IDs (required for schema, whose run numbers do
not establish shared story identity), or task/condition for milkyway/prettymouth.
Each value is a list of {name, family, event_ref: [task, event]} segments. An
explicit timing variant instead supplies onset, duration, units: "TR", evidence:
"local/path". Schema/condition mappings require evidence even with event_ref. Names
must identify the actual stimulus, not merely a run number. Condition-specific
tasks may not share segment names across conditions. Default slumlordreach
segments refer separately to event_meta[slumlordreach][slumlord|reach]. Ordinary
tasks use their single event_meta entry. TRs are zero-based, stop-exclusive;
trimming and linear detrend/population-standardization happen per segment.

Output dataset.json is schema_version=1, provenance=real_narratives, with
primary_eligible, targets, stories, config_sha256, input_hashes, output_hashes,
status and blockers. BOLD is float32 (subjects, time, targets), never averaged.
Feature NPZs must contain finite gpt2 (T,768), baseline (T,D), and a Unicode/byte
string baseline_names (D,) vector; optional metadata is a JSON string scalar.
Primary preparation requires story-bound narratives_features provenance metadata.
Features are validated/copied if supplied, never extracted or fabricated here.
Missing features yield bold_prepared_features_pending, NOT complete/eligible.
--exploratory-zero-ring explicitly disables mesh erosion and primary eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
from scipy.signal import detrend


N_VERTICES = 40962
NETWORKS = ("LANG", "MD", "EAC")
FAMILIES = {"piemanpni": "pieman", "notthefallintact": "notthefall",
            "shapesphysical": "shapes", "shapessocial": "shapes",
            "notthefalllongscram": "notthefall", "notthefallshortscram": "notthefall"}
CONDITION_STIMULI = ("milkyway", "prettymouth")


class PreparationError(ValueError):
    """An input cannot establish the requested scientific contract."""


SCAN = re.compile(
    r"^(?P<subject>sub-[a-zA-Z0-9]+)_task-(?P<task>[a-zA-Z0-9]+)"
    r"(?:_run-(?P<run>[0-9]+))?_space-fsaverage6_hemi-(?P<hemi>[LR])\.func\.gii$"
)


def select_scans(task_meta, exclusions, tasks=None):
    """Select first exposure before subject-specific exclusions; schema is per-run."""
    selected, rejected = [], []
    for task in sorted(tasks if tasks is not None else task_meta):
        if task not in task_meta:
            raise PreparationError(f"Unknown task: {task}")
        for subject, record in sorted(task_meta[task].items()):
            grouped = {}
            paths = record.get("bold", {}).get("fsaverage6", {}).get("preproc", [])
            if not paths:
                raise PreparationError(f"No fsaverage6 metadata: {task}/{subject}")
            for source in paths:
                match = SCAN.fullmatch(PurePosixPath(source).name)
                if not match or match["subject"] != subject or match["task"] != task:
                    raise PreparationError(f"Unexpected scan identity: {source}")
                run = int(match["run"]) if match["run"] else None
                scan_id = f"{subject}_task-{task}"
                if match["run"]:
                    scan_id += f"_run-{match['run']}"
                group = grouped.setdefault(run, {"scan_id": scan_id, "sources": {}})
                if match["hemi"] in group["sources"]:
                    raise PreparationError(f"Duplicate hemisphere: {source}")
                group["sources"][match["hemi"]] = source
            if None in grouped and len(grouped) > 1:
                raise PreparationError(f"Ambiguous exposure order: {task}/{subject}")
            runs = sorted(grouped, key=lambda value: -1 if value is None else value)
            if task != "schema":
                runs = runs[:1]
            for run in runs:
                group = grouped[run]
                scan = dict(group, task=task, subject=subject, run=run,
                            condition=record.get("condition", "n/a"))
                tokens = exclusions.get(task, {}).get(subject, [])
                if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
                    raise PreparationError(f"Malformed exclusions: {task}/{subject}")
                base = f"{subject}_task-{task}"
                if any(token in (base, scan["scan_id"]) for token in tokens):
                    rejected.append(dict(scan, reason="excluded_first_exposure" if task != "schema" else "excluded_run"))
                else:
                    if set(group["sources"]) != {"L", "R"}:
                        raise PreparationError(f"Both hemispheres required: {scan['scan_id']}")
                    selected.append(scan)
    return selected, rejected


def segment_indices(event, trim_start, trim_end):
    """Event onset/duration and trims are integer original-run TR counts."""
    values = [event.get("onset"), event.get("duration"), trim_start, trim_end]
    if any(type(value) is not int for value in values):
        raise PreparationError("Timing and trims must be integer TR counts")
    onset, duration, start, end = values
    if min(onset, start, end) < 0 or duration - start - end < 3:
        raise PreparationError("Segment must retain at least three TRs")
    return list(range(onset + start, onset + duration - end))


def resolve_segments(scan, events, mappings, trim_start, trim_end):
    """Resolve stimulus identity and timing without guessing variant durations."""
    task, condition = scan["task"], scan["condition"]
    allowed_conditions = {"milkyway": {"original", "synonyms", "vodka"},
                          "prettymouth": {"affair", "paranoia"}}
    if task in allowed_conditions and condition not in allowed_conditions[task]:
        raise PreparationError(f"Unknown condition: {task}/{condition}")
    explicit = mappings.get(scan["scan_id"])
    if explicit is None and task in ("milkyway", "prettymouth"):
        explicit = mappings.get(f"{task}/{condition}")
    if explicit is None:
        if task in ("schema", "milkyway", "prettymouth"):
            raise PreparationError(f"Explicit event mapping required: {scan['scan_id']} ({condition})")
        entries = events.get(task, {})
        expected = {"slumlord", "reach"} if task == "slumlordreach" else {task}
        if set(entries) != expected:
            raise PreparationError(f"Ambiguous/missing event metadata: {task}")
        explicit = [{"name": f"{task}_{key}" if task == "slumlordreach" else task,
                     "family": FAMILIES.get(task, task), "event_ref": [task, key]} for key in sorted(entries)]
    if not isinstance(explicit, list) or not explicit:
        raise PreparationError(f"Empty/malformed event mapping: {scan['scan_id']}")
    segments, seen, occupied = [], set(), set()
    for mapping in explicit:
        if not isinstance(mapping, dict):
            raise PreparationError(f"Malformed segment mapping: {scan['scan_id']}")
        name, family = mapping.get("name"), mapping.get("family")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise PreparationError("Segment names must be safe unique filename stems")
        if name in seen or not isinstance(family, str) or not family:
            raise PreparationError(f"Duplicate name or missing family: {name}")
        seen.add(name)
        if task in ("schema", "milkyway", "prettymouth"):
            if not mapping.get("evidence"):
                raise PreparationError(f"Stimulus identity/timing requires evidence: {scan['scan_id']}")
            if family != task:
                raise PreparationError(f"Condition/run segments must retain family={task}")
        if "event_ref" in mapping:
            if "onset" in mapping or "duration" in mapping:
                raise PreparationError(f"Conflicting timing sources: {name}")
            reference = mapping["event_ref"]
            if not isinstance(reference, list) or len(reference) != 2:
                raise PreparationError(f"Invalid event reference: {name}")
            try:
                event = events[reference[0]][reference[1]]
            except (KeyError, TypeError) as error:
                raise PreparationError(f"Unknown event reference: {reference}") from error
        else:
            if mapping.get("units") != "TR" or not mapping.get("evidence"):
                raise PreparationError(f"Explicit TR timing requires local evidence: {name}")
            event = {"onset": mapping.get("onset"), "duration": mapping.get("duration")}
        indices = segment_indices(event, trim_start, trim_end)
        original = set(range(event["onset"], event["onset"] + event["duration"]))
        if occupied & original:
            raise PreparationError(f"Overlapping event intervals: {scan['scan_id']}")
        occupied.update(original)
        segments.append(dict(mapping, name=name, family=family, task=task,
                             condition=condition, event=event, tr_indices=indices))
    return segments


def _check_hemi(image, hemisphere):
    structure = image.meta.get("AnatomicalStructurePrimary")
    if structure and structure != {"L": "CortexLeft", "R": "CortexRight"}[hemisphere]:
        raise PreparationError(f"GIFTI hemisphere mismatch: {structure} != {hemisphere}")


def load_surface(path, hemisphere, n_vertices=N_VERTICES):
    """Load one time-vector per darray or an unambiguous 2-D GIFTI array."""
    image = nib.load(str(path))
    if not isinstance(image, nib.gifti.GiftiImage) or not image.darrays:
        raise PreparationError(f"Not a functional GIFTI: {path}")
    _check_hemi(image, hemisphere)
    if any(array.intent not in (0, 2001) for array in image.darrays):
        raise PreparationError(f"Non-time-series GIFTI intent: {path}")
    arrays = [np.asarray(array.data) for array in image.darrays]
    if all(array.ndim == 1 and array.size == n_vertices for array in arrays):
        data = np.stack(arrays)
    elif len(arrays) == 1 and arrays[0].ndim == 2:
        data = arrays[0]
        if sum(size == n_vertices for size in data.shape) != 1:
            raise PreparationError(f"Ambiguous GIFTI vertex/time axes: {data.shape}")
        if data.shape[0] == n_vertices:
            data = data.T
    else:
        raise PreparationError(f"Unexpected GIFTI shape: {path}")
    if data.shape[1] != n_vertices or data.shape[0] < 3 or not np.isfinite(data).all():
        raise PreparationError(f"Invalid shape or nonfinite BOLD: {path}")
    return data


def preprocess_segment(data, indices, vertex_ids, original_stop=None):
    if not indices or indices[-1] >= data.shape[0] or (original_stop is not None and original_stop > data.shape[0]):
        raise PreparationError("Event interval exceeds actual GIFTI time dimension")
    values = np.asarray(data[np.ix_(indices, vertex_ids)], dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or np.any(np.ptp(values, axis=0) == 0):
        raise PreparationError("Nonfinite or zero-variance target in segment")
    residual = detrend(values, axis=0, type="linear")
    scale = residual.std(axis=0)
    tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.max(np.abs(values), axis=0)) * 100
    if np.any(scale <= tolerance):
        raise PreparationError("Zero variance after segment-wise detrending")
    return ((residual - residual.mean(axis=0)) / scale).astype(np.float32)


def load_triangles(path, hemisphere, n_vertices=N_VERTICES):
    image = nib.load(str(path))
    if not isinstance(image, nib.gifti.GiftiImage):
        raise PreparationError("Geometry must be a real GIFTI surface with coordinates and triangles")
    _check_hemi(image, hemisphere)
    points = [array.data for array in image.darrays if array.intent == 1008]
    faces = [array.data for array in image.darrays if array.intent == 1009]
    if len(points) != 1 or len(faces) != 1:
        raise PreparationError("Geometry requires one pointset and one triangle array")
    points, faces = np.asarray(points[0]), np.asarray(faces[0])
    if points.shape != (n_vertices, 3) or not np.isfinite(points).all():
        raise PreparationError("Geometry is not aligned with fsaverage6 vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or not faces.size:
        raise PreparationError("Malformed triangle array")
    if faces.min() < 0 or faces.max() >= n_vertices or len(np.unique(faces)) != n_vertices:
        raise PreparationError("Geometry has missing/out-of-range vertex IDs")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise PreparationError("Degenerate triangle indices")
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise PreparationError("Surface must be a closed two-manifold mesh")
    area = np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                   points[faces[:, 2]] - points[faces[:, 0]]), axis=1)
    if np.any(area == 0):
        raise PreparationError("Zero-area geometry triangles")
    return faces


def network_targets(labels, memberships, triangles=None, boundary_rings=1):
    """Erode each network against ALL outside vertices, including background."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise PreparationError("Atlas must contain integer parcel IDs")
    if boundary_rings not in (0, 1) or (boundary_rings == 1 and triangles is None):
        raise PreparationError("Primary preparation requires supplied real triangle geometry")
    masks = {}
    for network in NETWORKS:
        parcels = memberships.get(network)
        if not isinstance(parcels, list) or not parcels or any(type(item) is not int or item <= 0 for item in parcels):
            raise PreparationError(f"Explicit positive atlas labels required: {network}")
        if len(set(parcels)) != len(parcels) or not set(parcels).issubset(set(labels.tolist())):
            raise PreparationError(f"Unknown/duplicate atlas membership: {network}")
        masks[network] = np.isin(labels, parcels)
    for index, network in enumerate(NETWORKS):
        for other in NETWORKS[index + 1:]:
            if np.any(masks[network] & masks[other]):
                raise PreparationError(f"Overlapping networks: {network}/{other}")
    targets = []
    for network, mask in masks.items():
        keep = mask.copy()
        if boundary_rings:
            crossing = np.any(mask[triangles], axis=1) & ~np.all(mask[triangles], axis=1)
            keep[np.unique(triangles[crossing])] = False
        vertices = np.flatnonzero(keep)
        if not vertices.size:
            raise PreparationError(f"No {network} vertices survive boundary removal")
        targets.extend({"vertex_id": int(vertex), "network": network,
                "parcel": int(labels[vertex])} for vertex in vertices)
    return targets


def validate_features(path, n_time, story=None):
    with np.load(path, allow_pickle=False) as archive:
        if not {"gpt2", "baseline", "baseline_names"}.issubset(archive.files):
            raise PreparationError("Features require gpt2, baseline, baseline_names")
        gpt2, baseline, names = (archive[key] for key in ("gpt2", "baseline", "baseline_names"))
        if gpt2.shape != (n_time, 768) or baseline.ndim != 2 or baseline.shape[0] != n_time or baseline.shape[1] < 1:
            raise PreparationError("Feature time/width mismatch")
        if names.shape != (baseline.shape[1],) or names.dtype.kind not in "US":
            raise PreparationError("baseline_names must be a string vector of width D")
        decoded_names = names.astype(str).tolist()
        if len(set(decoded_names)) != len(decoded_names) or any(not name.strip() for name in decoded_names):
            raise PreparationError("baseline_names must be unique nonempty strings")
        if any(array.dtype.kind not in "fiu" or not np.isfinite(array).all() for array in (gpt2, baseline)):
            raise PreparationError("Nonfinite/nonnumeric features")
        if story is not None and "metadata" not in archive.files:
            raise PreparationError("Strict feature metadata required")
        if "metadata" in archive.files:
            metadata = archive["metadata"]
            if metadata.shape != () or metadata.dtype.kind not in "US":
                raise PreparationError("Feature metadata must be a JSON string scalar")
            metadata = json.loads(metadata.item())
            if not isinstance(metadata, dict):
                raise PreparationError("Feature metadata JSON must be an object")
            if story is not None:
                validate_feature_metadata(metadata, decoded_names, n_time, story)


def validate_feature_metadata(metadata, names, n_time, story):
    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def hash_inventory(value):
        return (isinstance(value, dict) and bool(value)
                and all(text(key) and digest(item) for key, item in value.items()))

    onset, duration = story["event"]["onset"], story["event"]["duration"]
    indices = story["tr_indices"]
    if (len(indices) != n_time or any(type(index) is not int for index in indices)
            or not indices or indices != list(range(indices[0], indices[-1] + 1))
            or indices[0] < onset or indices[-1] >= onset + duration):
        raise PreparationError("Invalid story TR association")
    required = dict(schema_version=1, model="gpt2", layer=9, context_window_tr=5,
                    tr_seconds=1.5, causal_windows=True, provenance="real_narratives",
                    story=story["name"], story_onset_tr=onset, story_duration_tr=duration,
                    tr_indices=indices, full_story_alignment=True, word_availability="word_end",
                    window_interval="(start,end]", pooling="mean_all_attended_tokens",
                    padding="none", empty_context_policy="zero_vector", token_overflow_policy="error")
    for key, expected in required.items():
        actual = metadata.get(key)
        if (actual != expected or (type(expected) in (bool, int) and type(actual) is not type(expected))
                or (key == "tr_indices" and (not isinstance(actual, list)
                    or any(type(index) is not int for index in actual)))):
            raise PreparationError(f"Strict feature metadata mismatch: {key}")
    offset = metadata.get("timestamp_offset_seconds")
    if type(offset) not in (int, float) or not np.isfinite(offset):
        raise PreparationError("Feature alignment requires finite timestamp offset")
    alignment = metadata.get("alignment_provenance")
    certification = dict(certified=True, full_story=True, story=story["name"],
                         timebase="story_relative_seconds", offset_seconds=offset,
                         coverage_start_seconds=0, coverage_stop_seconds=duration * 1.5)
    if (not isinstance(alignment, dict) or not text(alignment.get("source"))
            or any(alignment.get(key) != value or (type(value) is bool and alignment.get(key) is not value)
                   for key, value in certification.items())):
        raise PreparationError("Certified full-story feature alignment required")
    baseline = metadata.get("baseline_provenance")
    if (not isinstance(baseline, dict) or baseline.get("story") != story["name"]
            or baseline.get("provenance") != "real_narratives"
            or baseline.get("tr_seconds") != 1.5 or baseline.get("causal_windows") is not True):
        raise PreparationError("Strict baseline story/TR/causal provenance required")
    required_baseline = {"phoneme_or_syllable_rate", "pitch", "energy", "word_frequency",
                         "sentence_boundary", "voiced_fraction", "word_rate", "word_length",
                         "unigram_surprisal"}
    details = baseline.get("features")
    if (not required_baseline.issubset(names) or not isinstance(details, dict)
            or set(details) != set(names)):
        raise PreparationError("Strict per-column extended baseline provenance required")
    for name, detail in details.items():
        if (not isinstance(detail, dict) or detail.get("causal") is not True
                or any(not text(detail.get(key)) for key in ("source", "method", "units"))
                or not digest(detail.get("source_sha256"))):
            raise PreparationError(f"Invalid baseline provenance: {name}")
    reference = baseline.get("unigram_reference")
    if (not isinstance(reference, dict) or not text(reference.get("corpus"))
            or not digest(reference.get("source_sha256"))
            or reference.get("independent_of_evaluation_corpus") is not True
            or reference.get("estimated_on_narratives") is not False):
        raise PreparationError("Exogenous unigram baseline provenance required")
    files = metadata.get("model_file_hashes")
    if (not hash_inventory(metadata.get("source_hashes")) or not digest(metadata.get("config_sha256"))
            or not hash_inventory(files) or "config.json" not in files
            or not digest(metadata.get("model_weights_sha256"))
            or not isinstance(metadata.get("model_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", metadata["model_revision"]) is None):
        raise PreparationError("Feature source hashes and pretrained model weights/revision required")
    for filename in ("model.safetensors", "model.safetensors.index.json",
                     "pytorch_model.bin", "pytorch_model.bin.index.json"):
        if filename in files:
            weights = {filename: files[filename]}
            if filename.endswith(".index.json"):
                suffix = ".safetensors" if filename.startswith("model.") else ".bin"
                if not any(name.endswith(suffix) for name in files):
                    raise PreparationError("Missing model weight shard hashes")
            else:
                expected = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                if metadata["model_weights_sha256"] != expected:
                    raise PreparationError("Model weights digest does not match file hashes")
            break
    else:
        raise PreparationError("Missing model weight file hashes")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Inputs:
    def __init__(self, config_path):
        self.path = Path(config_path).resolve()
        self.base = self.path.parent
        self.config_hash = sha256(self.path)
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.hashes = {}
        self.pins = {}
        for relative, digest in self.config.get("input_sha256", {}).items():
            path = self.resolve(relative)
            if path in self.pins or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise PreparationError(f"Invalid/duplicate input hash pin: {relative}")
            self.pins[path] = digest

    def resolve(self, path):
        return (self.base / path).resolve()

    def checked(self, path):
        path = self.resolve(path)
        if not path.is_file():
            raise PreparationError(f"Missing local input: {path}")
        if path not in self.pins:
            raise PreparationError(f"Missing input_sha256 pin: {path}")
        digest = sha256(path)
        if digest != self.pins[path]:
            raise PreparationError(f"SHA256 mismatch: {path}")
        self.hashes[path.as_posix()] = digest
        return path

    def read_json(self, path):
        return json.loads(self.checked(path).read_text(encoding="utf-8"))


def cache_path(cache_root, scan, hemisphere):
    return Path(cache_root) / "afni-nosmooth" / scan["subject"] / (
        f"{scan['scan_id']}_space-fsaverage6_hemi-{hemisphere}_desc-clean.func.gii")


def build_plan(inputs, zero_ring=False):
    config = inputs.config
    if config.get("schema_version") != 1 or config.get("provenance") != "real_narratives" or config.get("space") != "fsaverage6":
        raise PreparationError("Requires schema_version=1, provenance=real_narratives, space=fsaverage6")
    metadata = config["metadata"]
    tasks = inputs.read_json(metadata["task_meta"])
    exclusions = inputs.read_json(metadata["scan_exclude"])
    events = inputs.read_json(metadata["event_meta"])
    selected, rejected = select_scans(tasks, exclusions, config.get("tasks"))
    blockers, targets, stories = [], [], {}
    first_schema = {}
    schema_segments = {}
    schema_scans = sorted([scan for scan in selected + rejected if scan["task"] == "schema"],
                          key=lambda scan: (scan["subject"], -1 if scan["run"] is None else scan["run"]))
    for scan in schema_scans:
        try:
            segments = resolve_segments(scan, events, config.get("event_mappings", {}),
                                        config["trim_start"], config["trim_end"])
            for segment in segments:
                inputs.checked(segment["evidence"])
                first_schema.setdefault((scan["subject"], segment["name"]), scan["scan_id"])
            schema_segments[scan["scan_id"]] = segments
        except (PreparationError, KeyError, TypeError) as error:
            blockers.append(f"{scan['scan_id']} pre-exclusion schema identity: {error}")
    try:
        inputs.checked(config["membership_evidence"])
    except (PreparationError, KeyError) as error:
        blockers.append(f"Network membership evidence: {error}")
    for hemisphere in ("L", "R"):
        try:
            atlas_path = inputs.checked(config["atlases"][hemisphere])
            atlas = nib.load(str(atlas_path))
            if not isinstance(atlas, nib.gifti.GiftiImage) or len(atlas.darrays) != 1 or atlas.darrays[0].intent != 1002:
                raise PreparationError("Atlas requires one GIFTI LABEL array")
            _check_hemi(atlas, hemisphere)
            labels = np.asarray(atlas.darrays[0].data)
            if labels.shape != (N_VERTICES,):
                raise PreparationError("Atlas must have exactly 40962 original vertices per hemisphere")
            triangles = None if zero_ring else load_triangles(inputs.checked(config["geometry"][hemisphere]), hemisphere)
            targets.extend(dict(target, hemisphere=hemisphere) for target in network_targets(
                labels, config["network_labels"][hemisphere], triangles, 0 if zero_ring else 1))
        except (PreparationError, KeyError, OSError, ValueError) as error:
            blockers.append(f"{hemisphere} atlas/geometry: {error}")
    cache = inputs.resolve(config["cache_root"])
    for scan in selected:
        try:
            if scan["task"] == "schema":
                segments = [segment for segment in schema_segments.get(scan["scan_id"], [])
                            if first_schema[(scan["subject"], segment["name"])] == scan["scan_id"]]
                if not segments:
                    continue
            else:
                segments = resolve_segments(scan, events, config.get("event_mappings", {}),
                                            config["trim_start"], config["trim_end"])
            for segment in segments:
                if segment.get("evidence"):
                    inputs.checked(segment["evidence"])
            paths = {}
            for hemisphere in ("L", "R"):
                try:
                    paths[hemisphere] = inputs.checked(cache_path(cache, scan, hemisphere))
                except PreparationError as error:
                    blockers.append(f"{scan['scan_id']}/{hemisphere}: {error}")
            for segment in segments:
                # Listener conditions with one shared stimulus (exposure order, prior movie) pool into one segment.
                condition = segment["condition"] if segment["task"] in CONDITION_STIMULI else None
                signature = (segment["task"], segment["family"], condition, tuple(segment["tr_indices"]),
                             segment["event"]["onset"], segment["event"]["duration"])
                story = stories.setdefault(segment["name"], dict(segment, signature=signature, scans=[]))
                if story["signature"] != signature or any(previous["subject"] == scan["subject"] for previous in story["scans"]):
                    raise PreparationError(f"Inconsistent segment identity or repeated subject: {segment['name']}")
                story["scans"].append(dict(scan, paths=paths))
        except (PreparationError, KeyError, TypeError) as error:
            blockers.append(f"{scan['scan_id']}: {error}")
    if not stories:
        blockers.append("No resolvable story data; dataset is not complete")
    for name, story in stories.items():
        feature_source = config.get("feature_files", {}).get(name)
        if feature_source:
            try:
                story["feature_source"] = inputs.checked(feature_source)
                validate_features(story["feature_source"], len(story["tr_indices"]), story)
            except (PreparationError, ValueError, OSError) as error:
                blockers.append(f"{name} features: {error}")
    return dict(targets=targets, stories=stories, selected=selected, rejected=rejected,
                blockers=blockers, boundary_rings=0 if zero_ring else 1)


def prepare(inputs, plan, output):
    """Publish only after ALL selected BOLD inputs validate; never skip a bad scan."""
    if plan["blockers"]:
        raise PreparationError("Preparation blocked; use --inventory for all blockers")
    output = Path(output).resolve()
    if output.exists():
        raise PreparationError("Output directory already exists; use a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    with tempfile.TemporaryDirectory(prefix="narratives-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        (staging / "bold").mkdir(parents=True)
        (staging / "features").mkdir()
        manifest = dict(schema_version=1, provenance="real_narratives", primary_eligible=False,
                        targets=plan["targets"], stories=[], config_sha256=inputs.config_hash,
                        boundary_rings=plan["boundary_rings"], feature_metadata_strict=False,
                        input_hashes={}, output_hashes={}, excluded_scans=plan["rejected"])
        for name, story in sorted(plan["stories"].items()):
            if story["tr_indices"] != segment_indices(story["event"], inputs.config["trim_start"], inputs.config["trim_end"]):
                raise PreparationError("Story TR indices mismatch configured trims")
            bold_path = f"bold/{name}.npy"
            feature_path = f"features/{name}.npz"
            shape = (len(story["scans"]), len(story["tr_indices"]), len(plan["targets"]))
            tensor = np.lib.format.open_memmap(staging / bold_path, mode="w+", dtype=np.float32, shape=shape)
            try:
                for subject_index, scan in enumerate(story["scans"]):
                    n_time = None
                    for hemisphere in ("L", "R"):
                        path = inputs.checked(scan["paths"][hemisphere])
                        data = load_surface(path, hemisphere)
                        if n_time is not None and data.shape[0] != n_time:
                            raise PreparationError(f"Hemisphere time mismatch: {scan['scan_id']}")
                        n_time = data.shape[0]
                        columns = [index for index, target in enumerate(plan["targets"]) if target["hemisphere"] == hemisphere]
                        vertices = [plan["targets"][index]["vertex_id"] for index in columns]
                        values = preprocess_segment(data, story["tr_indices"], vertices,
                                                    story["event"]["onset"] + story["event"]["duration"])
                        tensor[subject_index][:, columns] = values
                tensor.flush()
            finally:
                tensor._mmap.close()
            if "feature_source" in story:
                source = inputs.checked(story["feature_source"])
                validate_features(source, shape[1], story)
                shutil.copyfile(source, staging / feature_path)
                manifest["output_hashes"][feature_path] = sha256(staging / feature_path)
            else:
                pending.append(f"Missing features: {feature_path}")
            manifest["output_hashes"][bold_path] = sha256(staging / bold_path)
            manifest["stories"].append(dict(name=name, family=story["family"], task=story["task"],
                subject_ids=[scan["subject"] for scan in story["scans"]], bold=bold_path,
                tr_indices=story["tr_indices"], features=feature_path,
                trim_start=inputs.config["trim_start"], trim_end=inputs.config["trim_end"],
                tr_seconds=1.5, preprocessing="segment_linear_detrend_population_zscore",
                condition=story["condition"] if story["task"] in CONDITION_STIMULI else "pooled",
                scan_conditions=[scan["condition"] for scan in story["scans"]], event=story["event"],
                event_ref=story.get("event_ref"), timing_evidence=story.get("evidence"),
                scan_ids=[scan["scan_id"] for scan in story["scans"]]))
        for path in list(inputs.hashes):
            inputs.checked(path)
        if sha256(inputs.path) != inputs.config_hash:
            raise PreparationError("Config changed during preparation")
        manifest["input_hashes"] = dict(inputs.hashes)
        manifest["status"] = "bold_prepared_features_pending" if pending else "prepared"
        manifest["blockers"] = pending
        manifest["feature_metadata_strict"] = bool(manifest["stories"]) and not pending
        manifest["primary_eligible"] = manifest["feature_metadata_strict"] and plan["boundary_rings"] == 1
        manifest["analysis_tier"] = "primary" if plan["boundary_rings"] == 1 else "exploratory_zero_ring"
        (staging / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
    return manifest


def inventory_without_config(metadata_dir):
    metadata_dir = Path(metadata_dir)
    task_meta = json.loads((metadata_dir / "task_meta.json").read_text(encoding="utf-8"))
    exclusions = json.loads((metadata_dir / "scan_exclude.json").read_text(encoding="utf-8"))
    events = json.loads((metadata_dir / "event_meta.json").read_text(encoding="utf-8"))
    selected, rejected = select_scans(task_meta, exclusions)
    cache = metadata_dir / "processed_cache"
    present = {hemisphere: sum(cache_path(cache, scan, hemisphere).is_file() for scan in selected) for hemisphere in ("L", "R")}
    missing = {hemisphere: len(selected) - present[hemisphere] for hemisphere in ("L", "R")}
    ambiguous = sorted({scan["task"] for scan in selected if scan["task"] in ("schema", "milkyway", "prettymouth")})
    return dict(status="blocked", primary_eligible=False, selected_scans=len(selected),
                excluded_scans=len(rejected), local_clean_files=present, event_tasks=sorted(events),
                missing_clean_files=missing,
                local_atlas_files={hemisphere: (metadata_dir / f"tpl-fsaverage6_hemi-{hemisphere}_desc-MMP_dseg.label.gii").is_file()
                                   for hemisphere in ("L", "R")},
                metadata_sha256={name: sha256(metadata_dir / f"{name}.json") for name in ("task_meta", "scan_exclude", "event_meta")},
                blockers=["No preparation config/input hash pins supplied",
                          "Both hemisphere atlas memberships and membership evidence required",
                          "Both real fsaverage6 triangle meshes required for primary preparation",
                          f"Missing local clean GIFTIs across all tasks: L={missing['L']}, R={missing['R']}; select a manageable task subset",
                          f"Explicit stimulus/event mappings required for: {', '.join(ambiguous)}",
                          "Feature files not extracted; completion and primary eligibility unestablished"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="Pinned local preparation config JSON (not the gate config)")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--metadata-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "narratives_meta")
    parser.add_argument("--inventory", action="store_true", help="Report blockers as JSON without writing output")
    parser.add_argument("--exploratory-zero-ring", action="store_true", help="Explicit non-primary mode without mesh erosion")
    args = parser.parse_args(argv)
    try:
        if args.config is None:
            if not args.inventory:
                parser.error("--config is required for preparation")
            report = inventory_without_config(args.metadata_dir)
        else:
            inputs = Inputs(args.config)
            plan = build_plan(inputs, args.exploratory_zero_ring)
            report = dict(status="blocked" if plan["blockers"] else "inputs_planned_not_prepared",
                          primary_eligible=False, blockers=plan["blockers"],
                          selected_scans=len(plan["selected"]), excluded_scans=len(plan["rejected"]),
                          story_names=sorted(plan["stories"]), target_count=len(plan["targets"]),
                          features_pending=[name for name, story in plan["stories"].items() if "feature_source" not in story],
                          config_sha256=inputs.config_hash, input_hashes=inputs.hashes)
            if not args.inventory:
                if plan["blockers"]:
                    print(json.dumps(report, indent=2))
                    return 2
                if args.output is None:
                    parser.error("--output is required for preparation")
                manifest = prepare(inputs, plan, args.output)
                report = dict(status=manifest["status"], primary_eligible=manifest["primary_eligible"],
                              blockers=manifest["blockers"], dataset=str(args.output / "dataset.json"))
        print(json.dumps(report, indent=2))
        return 2 if report["status"] == "blocked" else 0
    except (PreparationError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="blocked", primary_eligible=False, blockers=[str(error)])), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
