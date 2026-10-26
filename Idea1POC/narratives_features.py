"""Offline, fail-closed GPT-2 layer-9 Narratives features.

Word availability is its certified completion time. TR j ends at (j+1)*1.5
story-relative seconds; its context contains completions in
(max(0, j-4)*1.5, (j+1)*1.5]. No HRF lag is applied here. Empty contexts
are errors, not zero vectors or invented EOS stimuli. Short contexts have no
left padding; token padding, if supplied to pooling, is masked out.

CLI: python -B Idea1POC/narratives_features.py --config local.json --output story.npz
All paths are local and relative to the config. Required config fields::

    {"schema_version": 1, "provenance": "real_narratives",
     "dataset": "prepared/dataset.json", "story": "pieman",
     "tr_seconds": 1.5, "context_window_tr": 5,
     "alignment": {"path": "words.csv", "evidence": "alignment.json",
                   "timebase": "story_relative_seconds", "offset_seconds": 0.0},
     "baseline": {"path": "baseline.npz", "provenance": "baseline.json"},
     "model": {"path": "local-gpt2", "revision": "40-character-upstream-commit",
               "evidence": "pretrained.json"},
     "input_sha256": {"relative/path/of/every/input": "64 hex digits"}}

words.csv has exactly story,word,onset_seconds,end_seconds columns, one story,
and the entire pretrim transcript in chronological order. Converted timestamps
are CSV timestamps + offset_seconds; no offset inference or rounding occurs.
alignment.json must certify certified=true, full_story=true, story, timebase,
offset_seconds, coverage_start_seconds=0, coverage_stop_seconds=duration*1.5,
and a nonempty source description. Certification must include silent edges.

baseline.npz contains baseline, baseline_names, tr_indices covering EVERY
untrimmed original-run story TR, in order. baseline.json supplies story,
provenance=real_narratives, tr_seconds=1.5, causal_windows=true, and features:
each column maps to {source, source_sha256, method, units, causal:true}.
unigram_reference requires corpus, source_sha256,
independent_of_evaluation_corpus=true and estimated_on_narratives=false.
No baseline is computed, fitted, imputed, normalized, or inferred here.

pretrained.json certifies model=gpt2, pretrained=true, revision, source and
weights_sha256. The latter is SHA256 of canonical JSON (sort_keys=True,
separators=(',', ':')) mapping selected weight/index filenames to SHA256.
Every model-directory file, evidence file, CSV, NPZ and dataset manifest must
also be pinned in input_sha256. The config is hashed separately.

Output includes retained gpt2/baseline rows and separate gpt2_history,
baseline_history, history_tr_indices for real rows required by downstream
lags 1..4. Rows before the story do not exist: downstream MUST drop lag edges
as specified in metadata.lag_history, never pad them or cross story boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TR_SECONDS = 1.5
CONTEXT_WINDOW_TR = 5
LAYER = 9
LAGS_TR = (1, 2, 3, 4)
REQUIRED_BASELINE = (
    "phoneme_or_syllable_rate", "pitch", "energy", "word_frequency",
    "sentence_boundary", "voiced_fraction", "word_rate", "word_length",
    "unigram_surprisal",
)


class FeatureError(ValueError):
    """A real, causally aligned feature contract could not be established."""


@dataclass(frozen=True)
class Word:
    text: str
    onset: float
    end: float


def validate_words(words, story_duration_seconds):
    if not np.isfinite(story_duration_seconds) or story_duration_seconds <= 0:
        raise FeatureError("Story duration must be positive and finite")
    if not words:
        raise FeatureError("Full-story aligned words are required")
    previous_end = 0.0
    for word in words:
        if not isinstance(word.text, str) or not word.text.strip():
            raise FeatureError("Words must contain nonempty text")
        if not np.isfinite([word.onset, word.end]).all():
            raise FeatureError("Word timestamps must be finite")
        if not 0 <= word.onset < word.end <= story_duration_seconds:
            raise FeatureError("Word timestamps must lie inside the full story")
        if word.onset < previous_end:
            raise FeatureError("Words must be ordered and nonoverlapping")
        previous_end = word.end
    return tuple(words)


def validate_tr_indices(indices, onset_tr, duration_tr):
    if type(onset_tr) is not int or type(duration_tr) is not int or onset_tr < 0 or duration_tr < 1:
        raise FeatureError("Story onset/duration must be integer original-run TRs")
    if not isinstance(indices, list) or not indices or any(type(item) is not int for item in indices):
        raise FeatureError("tr_indices must be a nonempty integer list")
    if indices != sorted(set(indices)) or indices[0] < onset_tr or indices[-1] >= onset_tr + duration_tr:
        raise FeatureError("tr_indices must be unique, increasing and inside the story")
    return tuple(indices)


def window_words(words, story_tr, duration_tr):
    """Pure completion-time selection on the full, untrimmed story timeline."""
    if type(story_tr) is not int or not 0 <= story_tr < duration_tr:
        raise FeatureError("Window TR lies outside the story")
    words = validate_words(words, duration_tr * TR_SECONDS)
    start = max(0, story_tr - CONTEXT_WINDOW_TR + 1) * TR_SECONDS
    stop = (story_tr + 1) * TR_SECONDS
    return tuple(word for word in words if start < word.end <= stop)


def build_windows(words, tr_indices, onset_tr, duration_tr):
    """Causal window text per TR; an empty string marks a window with no completed words."""
    indices = validate_tr_indices(tr_indices, onset_tr, duration_tr)
    validate_words(words, duration_tr * TR_SECONDS)
    windows = []
    for source_tr in indices:
        selected = window_words(words, source_tr - onset_tr, duration_tr)
        windows.append(" ".join(word.text for word in selected))
    return tuple(windows)


def history_indices(tr_indices, onset_tr, duration_tr):
    """Extra source rows needed for downstream lags; never cross story start."""
    retained = set(validate_tr_indices(tr_indices, onset_tr, duration_tr))
    return tuple(sorted({source_tr - lag for source_tr in retained for lag in LAGS_TR
                         if source_tr - lag >= onset_tr} - retained))


def tokenize_window(tokenizer, text, max_tokens):
    encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                        padding=False, return_attention_mask=True)
    token_ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if not token_ids or len(token_ids) > max_tokens:
        raise FeatureError(f"Context has {len(token_ids)} tokens; permitted 1..{max_tokens}; truncation forbidden")
    if len(mask) != len(token_ids) or any(value != 1 for value in mask):
        raise FeatureError("Individual contexts must be unpadded with all tokens attended")
    return encoded


def mean_token_hidden_states(hidden, attention_mask):
    """Pure NumPy reference pooling; no padding token contributes to the mean."""
    hidden, mask = np.asarray(hidden), np.asarray(attention_mask)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or hidden.shape[2] != 768:
        raise FeatureError("Expected hidden (batch,tokens,768) and aligned mask")
    if not np.isin(mask, [0, 1]).all() or np.any(mask.sum(axis=1) == 0):
        raise FeatureError("Pooling requires binary masks and at least one real token")
    attended = np.where(mask[..., None].astype(bool), hidden, 0)
    if not np.isfinite(attended).all():
        raise FeatureError("Nonfinite attended hidden states")
    return (attended.sum(axis=1) / mask.sum(axis=1, keepdims=True)).astype(np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FeatureError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise FeatureError(f"JSON object required: {path}")
    return value


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value):
    return isinstance(value, str) and bool(value.strip())


class Inputs:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.config_hash = sha256(self.path)
        self.config = _json_object(self.path)
        self.pins = self.config.get("input_sha256")
        if not isinstance(self.pins, dict) or not self.pins or any(not _digest(value) for value in self.pins.values()):
            raise FeatureError("input_sha256 must pin every consumed local file")
        self.hashes = {}

    def local(self, name):
        if not _text(name) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", name):
            raise FeatureError("Only explicit local paths are accepted; no URLs")
        return (self.path.parent / name).resolve(strict=True)

    def checked(self, name):
        path = self.local(name)
        if not path.is_file():
            raise FeatureError(f"Expected local file: {name}")
        key = Path(os.path.relpath(path, self.path.parent)).as_posix()
        actual = sha256(path)
        if self.pins.get(key) != actual:
            raise FeatureError(f"Missing/mismatched input_sha256: {key}")
        self.hashes[key] = actual
        return path

    def verify_unchanged(self):
        for name, expected in self.hashes.items():
            if sha256(self.local(name)) != expected:
                raise FeatureError(f"Input changed during extraction: {name}")
        if sha256(self.path) != self.config_hash:
            raise FeatureError("Config changed during extraction")


def load_alignment(inputs, specification, story, duration_tr):
    evidence = _json_object(inputs.checked(specification["evidence"]))
    offset = specification.get("offset_seconds")
    if type(offset) not in (float, int) or not np.isfinite(offset):
        raise FeatureError("Explicit finite alignment offset_seconds required, including zero")
    if specification.get("timebase") != "story_relative_seconds":
        raise FeatureError("Alignment must explicitly declare story_relative_seconds")
    required = dict(certified=True, full_story=True, story=story,
                    timebase="story_relative_seconds", offset_seconds=offset,
                    coverage_start_seconds=0, coverage_stop_seconds=duration_tr * TR_SECONDS)
    if any(evidence.get(key) != value or (isinstance(value, bool) and evidence.get(key) is not value)
           for key, value in required.items()) or not _text(evidence.get("source")):
        raise FeatureError("Certified full-story alignment coverage and exact offset evidence required")
    path = inputs.checked(specification["path"])
    words = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = {"story", "word", "onset_seconds", "end_seconds"}
        if reader.fieldnames is None or set(reader.fieldnames) != columns or len(reader.fieldnames) != 4:
            raise FeatureError("Alignment CSV requires exactly story,word,onset_seconds,end_seconds")
        for row in reader:
            if set(row) != columns or any(value is None for value in row.values()) or row["story"] != story:
                raise FeatureError("Malformed CSV row or story mismatch")
            words.append(Word(row["word"], float(row["onset_seconds"]) + offset,
                              float(row["end_seconds"]) + offset))
    return validate_words(words, duration_tr * TR_SECONDS), evidence


def load_baseline(inputs, specification, story, onset_tr, duration_tr):
    provenance = _json_object(inputs.checked(specification["provenance"]))
    if (provenance.get("story") != story or provenance.get("provenance") != "real_narratives"
            or provenance.get("tr_seconds") != TR_SECONDS or provenance.get("causal_windows") is not True):
        raise FeatureError("Baseline provenance must certify story, TR and causal windows")
    with np.load(inputs.checked(specification["path"]), allow_pickle=False) as archive:
        if not {"baseline", "baseline_names", "tr_indices"}.issubset(archive.files):
            raise FeatureError("Exogenous baseline requires baseline, baseline_names, tr_indices")
        baseline, names, indices = (archive[key] for key in ("baseline", "baseline_names", "tr_indices"))
    if names.ndim != 1 or names.dtype.kind not in "US":
        raise FeatureError("baseline_names must be a string vector")
    names = names.astype(str)
    if len(set(names.tolist())) != len(names) or any(not name.strip() for name in names):
        raise FeatureError("baseline_names must be unique nonempty strings")
    if not set(REQUIRED_BASELINE).issubset(names.tolist()):
        raise FeatureError("Missing strict extended baseline columns: " + ", ".join(sorted(set(REQUIRED_BASELINE) - set(names))))
    if (baseline.shape != (duration_tr, len(names)) or baseline.dtype.kind not in "fiu"
            or not np.isfinite(baseline).all()):
        raise FeatureError("Baseline must be finite and cover the entire pretrim story (T,D)")
    if (indices.dtype.kind not in "iu" or indices.shape != (duration_tr,)
            or not np.array_equal(indices, np.arange(onset_tr, onset_tr + duration_tr))):
        raise FeatureError("Baseline tr_indices must exactly cover full original-run story TRs")
    details = provenance.get("features")
    if not isinstance(details, dict) or set(details) != set(names):
        raise FeatureError("Per-column baseline provenance required")
    for name, detail in details.items():
        if (not isinstance(detail, dict) or detail.get("causal") is not True
                or any(not _text(detail.get(key)) for key in ("source", "method", "units"))
                or not _digest(detail.get("source_sha256"))):
            raise FeatureError(f"Missing causal source/method/units/hash for baseline {name}")
    reference = provenance.get("unigram_reference", {})
    if (not isinstance(reference, dict) or not _text(reference.get("corpus"))
            or not _digest(reference.get("source_sha256"))
            or reference.get("independent_of_evaluation_corpus") is not True
            or reference.get("estimated_on_narratives") is not False):
        raise FeatureError("Unigram statistics require an exogenous reference, never fitted on Narratives")
    with np.errstate(over="ignore"):
        baseline = baseline.astype(np.float32)
    if not np.isfinite(baseline).all():
        raise FeatureError("Baseline overflows float32")
    return baseline, names, provenance


def model_inventory(inputs, specification):
    directory = inputs.local(specification["path"])
    if not directory.is_dir():
        raise FeatureError("An existing local pretrained GPT-2 directory is required")
    revision = specification.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise FeatureError("Pin the immutable upstream model revision (40 lowercase hex digits)")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            checked = inputs.checked(os.path.relpath(path, inputs.path.parent))
            files[path.relative_to(directory).as_posix()] = sha256(checked)
    weights = None
    use_safetensors = True
    for filename in ("model.safetensors", "model.safetensors.index.json",
                     "pytorch_model.bin", "pytorch_model.bin.index.json"):
        if filename not in files:
            continue
        weights = {filename: files[filename]}
        use_safetensors = filename.startswith("model.safetensors")
        if filename.endswith(".index.json"):
            index = _json_object(directory / filename)
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise FeatureError("Invalid local sharded model index")
            for shard in mapping.values():
                if not isinstance(shard, str) or Path(shard).name != shard or shard not in files:
                    raise FeatureError("Missing/nonlocal model weight shard")
                weights[shard] = files[shard]
        break
    if weights is None or "config.json" not in files:
        raise FeatureError("Local pretrained GPT-2 config and weights are required; nothing is downloaded")
    digest = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    evidence = _json_object(inputs.checked(specification["evidence"]))
    if (evidence.get("model") != "gpt2" or evidence.get("pretrained") is not True
            or evidence.get("revision") != revision or evidence.get("weights_sha256") != digest
            or not _text(evidence.get("source"))):
        raise FeatureError("Pretrained GPT-2 provenance must certify revision and actual weights digest")
    return directory, files, digest, use_safetensors


def load_local_model(directory, use_safetensors):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        from transformers import GPT2Model, GPT2TokenizerFast
    except ImportError as error:
        raise FeatureError("Install torch and transformers separately; extraction never installs/downloads dependencies") from error
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(str(directory), local_files_only=True)
        model, loading = GPT2Model.from_pretrained(str(directory), local_files_only=True,
                                                   use_safetensors=use_safetensors,
                                                   output_loading_info=True)
    except (OSError, ValueError) as error:
        raise FeatureError(f"Complete local pretrained GPT-2/tokenizer files required: {error}") from error
    if (not isinstance(loading, dict) or any(loading.get(key) for key in
            ("missing_keys", "mismatched_keys", "error_msgs"))
            or set(loading.get("unexpected_keys", [])) - {"lm_head.weight"}):
        raise FeatureError("Incomplete/incompatible pretrained weights; random initialization is forbidden")
    config = model.config
    expected = dict(model_type="gpt2", n_embd=768, n_layer=12, n_head=12,
                    n_positions=1024, vocab_size=50257)
    if any(getattr(config, key, None) != value for key, value in expected.items()):
        raise FeatureError("Only pretrained GPT-2 small (12 layers, 768 hidden, 1024 positions) is supported")
    if tokenizer.vocab_size != 50257:
        raise FeatureError("Tokenizer must use the original GPT-2 vocabulary")
    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    return torch, tokenizer, model


def extract_gpt2(windows, directory, use_safetensors):
    """Layer-9 mean token states; empty windows (no completed words) are zero vectors, never fake tokens."""
    torch, tokenizer, model = load_local_model(directory, use_safetensors)
    contexts = [tokenize_window(tokenizer, text, model.config.n_positions) if text else None for text in windows]
    rows = []
    with torch.no_grad():
        for encoded in contexts:
            if encoded is None:
                rows.append(np.zeros(768, dtype=np.float32))
                continue
            tensors = {key: torch.tensor([encoded[key]], dtype=torch.long, device="cpu")
                       for key in ("input_ids", "attention_mask")}
            output = model(**tensors, output_hidden_states=True, use_cache=False, return_dict=True)
            if len(output.hidden_states) != 13:
                raise FeatureError("Expected embeddings plus 12 GPT-2 hidden states")
            hidden = output.hidden_states[LAYER].detach().cpu().numpy()
            if hidden.shape != (1, len(encoded["input_ids"]), 768):
                raise FeatureError("Unexpected GPT-2 hidden-state dimensions")
            rows.append(mean_token_hidden_states(hidden, [encoded["attention_mask"]])[0])
    result = np.stack(rows)
    if result.shape != (len(windows), 768) or not np.isfinite(result).all():
        raise FeatureError("GPT-2 returned nonfinite or malformed features")
    return result


def extract(config_path, output_path):
    output_path = Path(output_path).resolve()
    if output_path.exists() or not output_path.parent.is_dir():
        raise FeatureError("Output must be a new file in an existing local directory")
    inputs = Inputs(config_path)
    config = inputs.config
    if (type(config.get("schema_version")) is not int or config["schema_version"] != 1
            or config.get("provenance") != "real_narratives"
            or config.get("tr_seconds") != TR_SECONDS
            or type(config.get("context_window_tr")) is not int
            or config["context_window_tr"] != CONTEXT_WINDOW_TR):
        raise FeatureError("Require schema_version=1, real_narratives, tr_seconds=1.5, context_window_tr=5")
    manifest = _json_object(inputs.checked(config["dataset"]))
    if manifest.get("schema_version") != 1 or manifest.get("provenance") != "real_narratives":
        raise FeatureError("Dataset must use the narratives_prepare real_narratives schema")
    story_name = config["story"]
    matches = [item for item in manifest["stories"] if item["name"] == story_name]
    if len(matches) != 1:
        raise FeatureError("Story must identify exactly one prepared source segment")
    story = matches[0]
    onset, duration = story["event"]["onset"], story["event"]["duration"]
    retained = validate_tr_indices(story["tr_indices"], onset, duration)
    history = history_indices(list(retained), onset, duration)
    all_indices = sorted(set(retained) | set(history))
    words, alignment = load_alignment(inputs, config["alignment"], story_name, duration)
    baseline, names, baseline_provenance = load_baseline(inputs, config["baseline"], story_name, onset, duration)
    windows = build_windows(words, all_indices, onset, duration)
    directory, model_files, weights_digest, use_safetensors = model_inventory(inputs, config["model"])
    features = extract_gpt2(windows, directory, use_safetensors)
    if (features.shape != (len(all_indices), 768) or features.dtype.kind not in "fiu"
            or not np.isfinite(features).all()):
        raise FeatureError("Invalid extracted GPT-2 feature matrix")
    lookup = {source_tr: index for index, source_tr in enumerate(all_indices)}
    unavailable = {str(source_tr): [lag for lag in LAGS_TR if source_tr - lag < onset]
                   for source_tr in retained if source_tr - max(LAGS_TR) < onset}
    metadata = dict(
        schema_version=1, model="gpt2", layer=LAYER, context_window_tr=CONTEXT_WINDOW_TR,
        tr_seconds=TR_SECONDS, tr_indices=list(retained), provenance="real_narratives",
        causal_windows=True, story=story_name, story_onset_tr=onset, story_duration_tr=duration,
        word_availability="word_end", window_interval="(start,end]",
        timestamp_offset_seconds=config["alignment"]["offset_seconds"], full_story_alignment=True,
        alignment_provenance=alignment, model_revision=config["model"]["revision"],
        model_weights_sha256=weights_digest, model_file_hashes=model_files,
        source_hashes=dict(inputs.hashes), config_sha256=inputs.config_hash,
        baseline_provenance=baseline_provenance, pooling="mean_all_attended_tokens",
        padding="none", empty_context_policy="zero_vector", token_overflow_policy="error",
        empty_context_tr_indices=[source for source, text in zip(all_indices, windows) if not text],
        lag_history=dict(lags_built=False, required_lags_tr=list(LAGS_TR), tr_indices=list(history),
                         edge_policy="drop_rows_without_real_in_story_history", unavailable_lags=unavailable),
    )
    arrays = dict(
        gpt2=features[[lookup[index] for index in retained]].astype(np.float32),
        baseline=baseline[[index - onset for index in retained]], baseline_names=names,
        metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        gpt2_history=features[[lookup[index] for index in history]].astype(np.float32),
        baseline_history=baseline[[index - onset for index in history]],
        history_tr_indices=np.array(history, dtype=np.int64),
    )
    inputs.verify_unchanged()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        metadata = extract(args.config, args.output)
    except (FeatureError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Feature extraction blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(dict(output=str(args.output), story=metadata["story"],
                          n_tr=len(metadata["tr_indices"]), provenance=metadata["provenance"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())