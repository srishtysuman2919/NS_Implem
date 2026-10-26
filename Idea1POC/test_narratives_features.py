"""Synthetic fixtures test contracts only, never claim real feature extraction."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from narratives_features import (
    FeatureError, Inputs, REQUIRED_BASELINE, Word, build_windows, extract,
    extract_gpt2, history_indices, load_local_model, main, mean_token_hidden_states,
    sha256, tokenize_window, validate_tr_indices, validate_words, window_words,
)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.words = [Word(f"word{index}", index * 1.5, (index + 1) * 1.5)
                      for index in range(10)]

    def test_current_and_preceding_four_trs(self):
        self.assertEqual(build_windows(self.words, [15], 10, 10),
                         ("word1 word2 word3 word4 word5",))

    def test_pretrim_history_is_not_discarded(self):
        self.assertEqual(build_windows(self.words, [14, 15], 10, 10)[0],
                         "word0 word1 word2 word3 word4")

    def test_start_is_short_not_left_padded(self):
        self.assertEqual(build_windows(self.words, [10, 11], 10, 10),
                         ("word0", "word0 word1"))

    def test_exact_right_boundary_and_no_future_completion(self):
        words = [Word("at", 0, 1.5), Word("after", 1.5, 1.500001)]
        self.assertEqual(build_windows(words, [0], 0, 2), ("at",))

    def test_crossing_word_waits_for_completion(self):
        words = [Word("done", 0, 1), Word("crossing", 1, 1.6)]
        self.assertEqual(build_windows(words, [0, 1], 0, 2),
                         ("done", "done crossing"))

    def test_exact_left_boundary_excluded(self):
        self.assertEqual([word.text for word in window_words(self.words, 5, 10)],
                         [f"word{index}" for index in range(1, 6)])

    def test_empty_context_is_marked_not_filled(self):
        self.assertEqual(build_windows([Word("later", 2, 3)], [0, 1], 0, 3), ("", "later"))

    def test_invalid_alignment_fails(self):
        cases = ([Word("", 0, 1)], [Word("x", -1, 1)],
                 [Word("x", 0, float("nan"))], [Word("x", 1, 1)],
                 [Word("x", 0, 20)], [Word("x", 1, 2), Word("y", 0, 1)])
        for words in cases:
            with self.subTest(words=words), self.assertRaises(FeatureError):
                validate_words(words, 15)

    def test_invalid_source_indices_fail(self):
        for indices in ([1, 1], [2, 1], [True], [1.0], [-1], [10], []):
            with self.subTest(indices=indices), self.assertRaises(FeatureError):
                validate_tr_indices(indices, 0, 10)

    def test_lag_history_never_pads_or_crosses_story(self):
        self.assertEqual(history_indices([15, 16], 10, 10), (11, 12, 13, 14))
        self.assertEqual(history_indices([10, 11], 10, 10), ())
        self.assertEqual(history_indices([10, 15], 10, 10), (11, 12, 13, 14))


class TokenTests(unittest.TestCase):
    def test_tokenization_never_truncates_or_pads(self):
        def tokenizer(text, **kwargs):
            self.assertEqual(kwargs, dict(add_special_tokens=False, truncation=False,
                                          padding=False, return_attention_mask=True))
            return {"input_ids": [10, 20], "attention_mask": [1, 1]}
        self.assertEqual(tokenize_window(tokenizer, "text", 2)["input_ids"], [10, 20])
        with self.assertRaisesRegex(FeatureError, "truncation forbidden"):
            tokenize_window(tokenizer, "text", 1)

    def test_empty_tokenization_fails(self):
        with self.assertRaises(FeatureError):
            tokenize_window(lambda *args, **kwargs: {"input_ids": [], "attention_mask": []}, "", 10)

    def test_pooling_ignores_padding(self):
        hidden = np.ones((1, 3, 768), dtype=np.float32)
        hidden[:, 1] = 3
        hidden[:, 2] = np.nan
        pooled = mean_token_hidden_states(hidden, [[1, 1, 0]])
        np.testing.assert_array_equal(pooled, np.full((1, 768), 2, dtype=np.float32))
        with self.assertRaises(FeatureError):
            mean_token_hidden_states(hidden, [[0, 0, 0]])


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "features.npz"
        self.config_path = self.root / "config.json"
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "model.safetensors").write_bytes(b"TEST FIXTURE NOT REAL WEIGHTS")
        weights = {"model.safetensors": sha256(self.model_dir / "model.safetensors")}
        self.weights_digest = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.revision = "a" * 40
        self.manifest = dict(schema_version=1, provenance="real_narratives", stories=[
            dict(name="fixture", event=dict(onset=10, duration=10), tr_indices=[15, 16, 17])])
        self.alignment = dict(certified=True, full_story=True, story="fixture",
                              timebase="story_relative_seconds", offset_seconds=0,
                              coverage_start_seconds=0, coverage_stop_seconds=15,
                              source="Synthetic alignment fixture, not real data")
        self.provenance = dict(story="fixture", provenance="real_narratives", tr_seconds=1.5,
            causal_windows=True, features={name: dict(source="Synthetic source fixture", source_sha256="b" * 64,
            method="External fixture values", units="fixture units", causal=True) for name in REQUIRED_BASELINE},
            unigram_reference=dict(corpus="Synthetic independent corpus fixture", source_sha256="c" * 64,
                independent_of_evaluation_corpus=True, estimated_on_narratives=False))
        self.model_evidence = dict(model="gpt2", pretrained=True, revision=self.revision,
            source="Synthetic model provenance fixture", weights_sha256=self.weights_digest)
        self.config = dict(schema_version=1, provenance="real_narratives", dataset="dataset.json", story="fixture",
            tr_seconds=1.5, context_window_tr=5, alignment=dict(path="words.csv", evidence="alignment.json",
            timebase="story_relative_seconds", offset_seconds=0),
            baseline=dict(path="baseline.npz", provenance="baseline.json"),
            model=dict(path="model", revision=self.revision, evidence="pretrained.json"))
        self.baseline = np.arange(90, dtype=np.float32).reshape(10, 9)
        self.names = np.array(REQUIRED_BASELINE)
        self.indices = np.arange(10, 20)
        (self.root / "words.csv").write_text("story,word,onset_seconds,end_seconds\n" + "".join(
            f"fixture,word{index},{index * 1.5},{(index + 1) * 1.5}\n" for index in range(10)), encoding="utf-8")
        self.write_sources()

    def write_sources(self):
        for name, value in (("dataset.json", self.manifest), ("alignment.json", self.alignment),
                            ("baseline.json", self.provenance), ("pretrained.json", self.model_evidence)):
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        np.savez(self.root / "baseline.npz", baseline=self.baseline, baseline_names=self.names,
                 tr_indices=self.indices)
        self.pin()

    def pin(self):
        self.config["input_sha256"] = {path.relative_to(self.root).as_posix(): sha256(path)
            for path in self.root.rglob("*") if path.is_file() and path not in (self.config_path, self.output)}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def run_fixture(self):
        def fixture_features(windows, directory, use_safetensors):
            self.assertEqual(directory, self.model_dir)
            self.assertTrue(use_safetensors)
            return np.array([np.full(768, int(text.split()[-1][4:]), dtype=np.float32) for text in windows])
        with patch("narratives_features.extract_gpt2", side_effect=fixture_features) as extractor:
            metadata = extract(self.config_path, self.output)
        return metadata, extractor

    def assert_blocked(self, pattern):
        with patch("narratives_features.extract_gpt2") as extractor:
            with self.assertRaisesRegex(FeatureError, pattern):
                extract(self.config_path, self.output)
            extractor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_output_contract_and_exact_metadata_schema(self):
        metadata, extractor = self.run_fixture()
        self.assertEqual(set(metadata), {
            "schema_version", "model", "layer", "context_window_tr", "tr_seconds", "tr_indices",
            "provenance", "causal_windows", "story", "story_onset_tr", "story_duration_tr",
            "word_availability", "window_interval", "timestamp_offset_seconds", "full_story_alignment",
            "alignment_provenance", "model_revision", "model_weights_sha256", "model_file_hashes",
            "source_hashes", "config_sha256", "baseline_provenance", "pooling", "padding",
            "empty_context_policy", "token_overflow_policy", "lag_history", "empty_context_tr_indices"})
        self.assertEqual(metadata["empty_context_policy"], "zero_vector")
        self.assertEqual(metadata["model"], "gpt2")
        self.assertEqual(metadata["layer"], 9)
        self.assertEqual(metadata["context_window_tr"], 5)
        self.assertEqual(metadata["tr_seconds"], 1.5)
        self.assertEqual(metadata["tr_indices"], [15, 16, 17])
        self.assertEqual(metadata["provenance"], "real_narratives")
        self.assertIs(metadata["causal_windows"], True)
        self.assertEqual(metadata["model_revision"], self.revision)
        self.assertEqual(metadata["model_weights_sha256"], self.weights_digest)
        self.assertEqual(metadata["baseline_provenance"], self.provenance)
        self.assertEqual(metadata["source_hashes"], self.config["input_sha256"])
        self.assertEqual(metadata["config_sha256"], sha256(self.config_path))
        self.assertEqual(extractor.call_args.args[0][0], "word0 word1")
        with np.load(self.output, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {"gpt2", "baseline", "baseline_names", "metadata",
                                                "gpt2_history", "baseline_history", "history_tr_indices"})
            self.assertEqual(archive["gpt2"].shape, (3, 768))
            np.testing.assert_array_equal(archive["gpt2"][:, 0], [5, 6, 7])
            np.testing.assert_array_equal(archive["baseline"], self.baseline[5:8])
            self.assertEqual(archive["baseline_names"].dtype.kind, "U")
            self.assertEqual(archive["metadata"].shape, ())
            self.assertEqual(json.loads(archive["metadata"].item()), metadata)
            np.testing.assert_array_equal(archive["history_tr_indices"], [11, 12, 13, 14])
            np.testing.assert_array_equal(archive["gpt2_history"][:, 0], [1, 2, 3, 4])
            np.testing.assert_array_equal(archive["baseline_history"], self.baseline[1:5])

    def test_missing_lag_edges_are_explicit_not_padded(self):
        self.manifest["stories"][0]["tr_indices"] = [10, 11, 12]
        self.write_sources()
        metadata, _ = self.run_fixture()
        self.assertEqual(metadata["lag_history"], dict(lags_built=False,
            required_lags_tr=[1, 2, 3, 4], tr_indices=[],
            edge_policy="drop_rows_without_real_in_story_history",
            unavailable_lags={"10": [1, 2, 3, 4], "11": [2, 3, 4], "12": [3, 4]}))
        with np.load(self.output, allow_pickle=False) as archive:
            self.assertEqual(archive["gpt2_history"].shape, (0, 768))
            self.assertEqual(archive["baseline_history"].shape, (0, 9))

    def test_every_required_baseline_is_mandatory(self):
        for index, name in enumerate(REQUIRED_BASELINE):
            with self.subTest(name=name):
                self.names = np.delete(np.array(REQUIRED_BASELINE), index)
                self.baseline = np.zeros((10, 8))
                self.write_sources()
                self.assert_blocked("Missing strict extended")

    def test_no_heldout_unigram_estimation(self):
        self.provenance["unigram_reference"]["estimated_on_narratives"] = True
        self.write_sources()
        self.assert_blocked("exogenous reference")

    def test_baseline_per_feature_provenance_required(self):
        del self.provenance["features"]["pitch"]["source_sha256"]
        self.write_sources()
        self.assert_blocked("baseline pitch")

    def test_noncausal_baseline_rejected(self):
        self.provenance["features"]["energy"]["causal"] = False
        self.write_sources()
        self.assert_blocked("baseline energy")

    def test_trimmed_baseline_is_not_enough(self):
        self.baseline = self.baseline[5:8]
        self.write_sources()
        self.assert_blocked("entire pretrim")

    def test_shifted_baseline_indices_rejected(self):
        self.indices += 1
        self.write_sources()
        self.assert_blocked("exactly cover")

    def test_missing_alignment_certification(self):
        self.alignment["certified"] = False
        self.write_sources()
        self.assert_blocked("Certified full-story")

    def test_missing_pretrim_alignment_coverage(self):
        self.alignment["coverage_start_seconds"] = 7.5
        self.write_sources()
        self.assert_blocked("Certified full-story")

    def test_offset_must_be_explicit_and_certified(self):
        del self.config["alignment"]["offset_seconds"]
        self.pin()
        self.assert_blocked("Explicit finite")
        self.config["alignment"]["offset_seconds"] = 0.1
        self.pin()
        self.assert_blocked("exact offset evidence")

    def test_certified_offset_is_applied_exactly(self):
        self.config["alignment"]["offset_seconds"] = 0.25
        self.alignment["offset_seconds"] = 0.25
        self.write_sources()
        (self.root / "words.csv").write_text("story,word,onset_seconds,end_seconds\n" + "".join(
            f"fixture,word{index},{index * 1.5 - .25},{(index + 1) * 1.5 - .25}\n"
            for index in range(10)), encoding="utf-8")
        self.pin()
        metadata, _ = self.run_fixture()
        self.assertEqual(metadata["timestamp_offset_seconds"], .25)

    def test_context_and_tr_are_required_not_inferred(self):
        for name in ("context_window_tr", "tr_seconds"):
            original = self.config.pop(name)
            self.pin()
            self.assert_blocked("Require schema_version")
            self.config[name] = original

    def test_missing_weights_fail_without_loading_or_downloading(self):
        (self.model_dir / "model.safetensors").unlink()
        self.pin()
        self.assert_blocked("weights are required")

    def test_model_digest_or_revision_cannot_be_substituted(self):
        self.model_evidence["weights_sha256"] = "d" * 64
        self.write_sources()
        self.assert_blocked("actual weights digest")
        self.config["model"]["revision"] = "main"
        self.pin()
        self.assert_blocked("immutable upstream")

    def test_modified_source_hash_fails(self):
        with (self.root / "words.csv").open("a", encoding="utf-8") as stream:
            stream.write("fixture,extra,14,15\n")
        self.assert_blocked("mismatched input_sha256")

    def test_duplicate_json_keys_rejected(self):
        self.config_path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
        with self.assertRaisesRegex(FeatureError, "Duplicate JSON key"):
            Inputs(self.config_path)

    def test_nonfinite_baseline_rejected_without_imputation(self):
        self.baseline[0, 0] = np.nan
        self.write_sources()
        self.assert_blocked("finite and cover")

    def test_object_baseline_names_rejected_without_pickle(self):
        self.names = self.names.astype(object)
        self.write_sources()
        with patch("narratives_features.extract_gpt2") as extractor:
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                extract(self.config_path, self.output)
            extractor.assert_not_called()

    def test_during_extraction_mutation_fails_without_output(self):
        def mutate(*args):
            (self.root / "words.csv").write_text("changed", encoding="utf-8")
            return np.ones((7, 768), dtype=np.float32)
        with patch("narratives_features.extract_gpt2", side_effect=mutate):
            with self.assertRaisesRegex(FeatureError, "Input changed"):
                extract(self.config_path, self.output)
        self.assertFalse(self.output.exists())

    def test_no_overwrite(self):
        self.output.write_bytes(b"existing")
        self.assert_blocked_existing_output()

    def assert_blocked_existing_output(self):
        with self.assertRaisesRegex(FeatureError, "new file"):
            extract(self.config_path, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_remote_source_rejected(self):
        self.config["alignment"]["path"] = "https://example.invalid/words.csv"
        self.pin()
        self.assert_blocked("no URLs")

    def test_cli_fails_closed_on_missing_prerequisite(self):
        (self.root / "words.csv").unlink()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["--config", str(self.config_path), "--output", str(self.output)])
        self.assertEqual(status, 2)
        self.assertIn("Feature extraction blocked", stderr.getvalue())
        self.assertFalse(self.output.exists())


class ModelExecutionTests(unittest.TestCase):
    def fake_runtime(self):
        runtime = SimpleNamespace(long="long", tensor=Mock(side_effect=lambda value, **kwargs: value),
                                  no_grad=Mock(return_value=contextlib.nullcontext()))
        tokenizer = Mock(return_value={"input_ids": [10, 20], "attention_mask": [1, 1]})
        tokenizer.vocab_size = 50257
        model = Mock()
        model.config = SimpleNamespace(model_type="gpt2", n_embd=768, n_layer=12, n_head=12,
                                       n_positions=1024, vocab_size=50257)
        hidden_states = []
        for layer in range(13):
            hidden = Mock()
            hidden.detach.return_value.cpu.return_value.numpy.return_value = np.full((1, 2, 768), layer, np.float32)
            hidden_states.append(hidden)
        model.return_value = SimpleNamespace(hidden_states=hidden_states)
        return runtime, tokenizer, model

    def test_pretrained_loading_is_local_frozen_eval(self):
        runtime, tokenizer, model = self.fake_runtime()
        loading = dict(missing_keys=[], unexpected_keys=["lm_head.weight"], mismatched_keys=[], error_msgs=[])
        model_class = SimpleNamespace(from_pretrained=Mock(return_value=(model, loading)))
        tokenizer_class = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        transformers = SimpleNamespace(GPT2Model=model_class, GPT2TokenizerFast=tokenizer_class)
        with patch.dict("sys.modules", {"torch": runtime, "transformers": transformers}), patch.dict("os.environ"):
            result = load_local_model(Path("local-model"), True)
            import os
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(result, (runtime, tokenizer, model))
        model_class.from_pretrained.assert_called_once_with("local-model", local_files_only=True,
                                                           use_safetensors=True, output_loading_info=True)
        tokenizer_class.from_pretrained.assert_called_once_with("local-model", local_files_only=True)
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with("cpu")

    def test_partial_or_incompatible_weights_never_use_random_fallback(self):
        runtime, tokenizer, model = self.fake_runtime()
        for loading in (dict(missing_keys=["h.0.attn.c_attn.weight"]),
                        dict(mismatched_keys=["wte.weight"]), dict(error_msgs=["bad weights"]),
                        dict(unexpected_keys=["unknown.weight"])):
            with self.subTest(loading=loading):
                transformers = SimpleNamespace(
                    GPT2Model=SimpleNamespace(from_pretrained=Mock(return_value=(model, loading))),
                    GPT2TokenizerFast=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)))
                with patch.dict("sys.modules", {"torch": runtime, "transformers": transformers}), patch.dict("os.environ"):
                    with self.assertRaisesRegex(FeatureError, "random initialization is forbidden"):
                        load_local_model(Path("local-model"), True)
        model.eval.assert_not_called()

    def test_hidden_state_nine_not_embedding_or_final_and_no_grad(self):
        runtime, tokenizer, model = self.fake_runtime()
        with patch("narratives_features.load_local_model", return_value=(runtime, tokenizer, model)):
            result = extract_gpt2(["one two"], Path("local-model"), True)
        np.testing.assert_array_equal(result, np.full((1, 768), 9, dtype=np.float32))
        runtime.no_grad.assert_called_once_with()
        model.assert_called_once_with(input_ids=[[10, 20]], attention_mask=[[1, 1]],
                                      output_hidden_states=True, use_cache=False, return_dict=True)

    def test_empty_window_is_zero_vector_without_model_call(self):
        runtime, tokenizer, model = self.fake_runtime()
        with patch("narratives_features.load_local_model", return_value=(runtime, tokenizer, model)):
            result = extract_gpt2(["", "one two"], Path("local-model"), True)
        np.testing.assert_array_equal(result[0], np.zeros(768, dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.full(768, 9, dtype=np.float32))
        model.assert_called_once()


if __name__ == "__main__":
    unittest.main()