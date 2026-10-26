"""Build the real, hash-pinned Narratives Stage A inputs (everything except BOLD extraction).

Steps (each idempotent; run in order):
  evidence   per-scan BIDS events.tsv for every selected scan; timing must be identical
             within each task/condition or the build stops
  stimuli    story audio, Gentle alignments/transcripts, fsaverage6 meshes
  offsets    verify Gentle timestamps are in stimulus-audio time (envelope cross-correlation)
  features   words.csv + alignment evidence, nine-column comparator, local GPT-2 copy,
             then GPT-2 layer-9 extraction through narratives_features.extract
  config     hash-pinned preparation config for narratives_prepare.py

Word timestamps are converted to story-relative time with
  t_story = t_gentle + lag + wav_start_seconds - event_onset_tr * 1.5
where wav_start_seconds is the first event onset (the stimulus file starts playing there)
and lag is the measured Gentle-to-stimulus audio offset. Only Gentle "success" words that
lie inside the story event are kept; misaligned words placed in lead-in music are dropped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from narratives_prepare import FAMILIES, resolve_segments, select_scans, sha256

REPOSITORY = Path(__file__).resolve().parents[1]
META = REPOSITORY / "data" / "narratives_meta"
WORK = REPOSITORY / "data" / "narratives_stage_a"
S3 = "https://s3.amazonaws.com/fcp-indi/data/Projects/narratives"
BIDS = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds002345/1.1.4"
TEMPLATEFLOW = "https://templateflow.s3.amazonaws.com/tpl-fsaverage"
HF_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--gpt2" / "snapshots"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TR = 1.5
TRIM_START, TRIM_END = 10, 5
TASKS = ["pieman", "shapesphysical", "shapessocial", "notthefallintact", "milkyway", "bronx", "forgot",
         "black", "prettymouth", "sherlock", "merlin", "21styear", "tunnel", "slumlordreach", "lucy"]
CONDITIONS = {"milkyway": {"original": "milkywayoriginal", "vodka": "milkywayvodka", "synonyms": "milkywaysynonyms"},
              "prettymouth": {"affair": "prettymouth", "paranoia": "prettymouth"}}
NETWORK_PARCELS = {
    "LANG": ["44", "45", "47l", "55b", "STGa", "STSda", "STSdp", "STSva", "STSvp", "PSL", "SFL", "TGd", "TE1a"],
    "MD": ["i6-8", "8C", "p9-46v", "a9-46v", "IFJp", "8BM", "SCEF", "AVI", "IP1", "IP2", "MIP", "LIPd", "AIP", "PFm"],
    "EAC": ["A1", "LBelt", "MBelt", "PBelt", "RI"],
}
BASELINE_NAMES = ["voiced_fraction", "word_rate", "word_length", "unigram_surprisal", "phoneme_or_syllable_rate",
                  "pitch", "energy", "word_frequency", "sentence_boundary"]


def curl(url, destination):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(["curl.exe", "-sSfL", "--retry", "5", url, "-o", str(partial)], check=True)
    partial.replace(destination)
    return destination


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def selected_scans():
    selected, rejected = select_scans(json.loads((META / "task_meta.json").read_text()),
                                      json.loads((META / "scan_exclude.json").read_text()), TASKS)
    return selected, rejected


def read_events(path):
    rows = [line.split("\t") for line in Path(path).read_text().strip().splitlines()]
    header, body = rows[0], rows[1:]
    return [dict(zip(header, row)) for row in body]


def gentle_name(task, condition):
    return CONDITIONS[task][condition] if task in CONDITIONS else task


def step_evidence():
    selected, _ = selected_scans()
    events_dir = WORK / "events"

    def fetch(scan):
        name = f"{scan['scan_id']}_events.tsv"
        return scan, curl(f"{BIDS}/{scan['subject']}/func/{name}", events_dir / name)

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(fetch, selected))
    groups = {}
    for scan, path in results:
        signature = tuple((row["onset"], row["duration"], row["trial_type"], row["stim_file"]) for row in read_events(path))
        groups.setdefault((scan["task"], scan["condition"]), {}).setdefault(signature, []).append(scan["scan_id"])
    summary, problems = {}, []
    for (task, condition), variants in sorted(groups.items()):
        if len(variants) != 1:
            problems.append(f"{task}/{condition}: {len(variants)} distinct event layouts "
                            + "; ".join(f"{len(ids)} scans e.g. {ids[0]}" for ids in variants.values()))
            continue
        signature, scans = next(iter(variants.items()))
        stimulus = {row[3] for row in signature if row[3] != "n/a"}
        if len(stimulus) != 1:
            problems.append(f"{task}/{condition}: stimulus files {stimulus}")
            continue
        summary[f"{task}/{condition}"] = dict(scans=len(scans), first_onset_seconds=float(signature[0][0]),
                                              stim_file=stimulus.pop(), events=[list(row) for row in signature],
                                              evidence_files=[f"events/{scan_id}_events.tsv" for scan_id in scans])
    write_json(WORK / "timing_evidence.json", dict(source=f"{BIDS}/<sub>/func/<scan>_events.tsv", groups=summary,
                                                   problems=problems))
    print(json.dumps({key: (value["scans"], value["first_onset_seconds"], value["stim_file"])
                      for key, value in summary.items()}, indent=1))
    if problems:
        print("TIMING PROBLEMS:\n" + "\n".join(problems))
        sys.exit(2)


def step_stimuli():
    timing = json.loads((WORK / "timing_evidence.json").read_text())["groups"]
    jobs = []
    for key, group in timing.items():
        task, condition = key.split("/", 1)
        story = gentle_name(task, condition)
        jobs.append((f"{S3}/stimuli/{group['stim_file']}", WORK / "stimuli" / group["stim_file"]))
        for name in ("align.json", "transcript.txt", "a.wav"):
            jobs.append((f"{S3}/stimuli/gentle/{story}/{name}", WORK / "gentle" / story / name))
    for hemisphere in "LR":
        name = f"tpl-fsaverage_hemi-{hemisphere}_den-41k_sphere.surf.gii"
        jobs.append((f"{TEMPLATEFLOW}/{name}", WORK / "meshes" / name))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda job: curl(*job), sorted(set(jobs))))
    print(f"{len(set(jobs))} stimulus/mesh files present")


def mono(path, rate=16000):
    from math import gcd
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    source_rate, raw = wavfile.read(path, mmap=True)
    scale = {"int16": 32768.0, "int32": 2147483648.0, "uint8": 128.0}.get(raw.dtype.name, 1.0)
    data = np.asarray(raw, dtype=np.float32)
    if raw.dtype.name == "uint8":
        data -= 128.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    data /= scale
    divisor = gcd(int(source_rate), rate)
    return resample_poly(data, rate // divisor, int(source_rate) // divisor).astype(np.float32), rate


def envelope(signal, rate, hop=0.01):
    frame = int(rate * hop)
    usable = len(signal) // frame * frame
    return np.sqrt(np.mean(np.square(signal[:usable].reshape(-1, frame)), axis=1) + 1e-12)


def step_offsets():
    timing = json.loads((WORK / "timing_evidence.json").read_text())["groups"]
    results = {}
    for story, stim_file in sorted({(gentle_name(*key.split("/", 1)), group["stim_file"]) for key, group in timing.items()}):
        gentle, rate = mono(WORK / "gentle" / story / "a.wav")
        stimulus, _ = mono(WORK / "stimuli" / stim_file)
        a, b = np.log(envelope(gentle, rate)), np.log(envelope(stimulus, rate))
        length = min(len(a), len(b), 30000)
        a, b = a[:length] - a[:length].mean(), b[:length] - b[:length].mean()
        lags = np.arange(-500, 501)
        scores = [np.corrcoef(a[max(0, lag):length + min(0, lag)], b[max(0, -lag):length - max(0, lag)])[0, 1] for lag in lags]
        best = int(lags[int(np.argmax(scores))])
        from scipy.signal import butter, correlate, sosfiltfilt
        lowpass = butter(6, 3800, fs=rate, output="sos")
        shift = int(round(best * 0.01 * rate))
        search = int(0.05 * rate)
        windows = []
        for start in (60, 200, 400):
            begin, end = start * rate, (start + 60) * rate
            if end + search + abs(shift) > min(len(gentle), len(stimulus)):
                continue
            x = sosfiltfilt(lowpass, gentle[begin:end])
            y = sosfiltfilt(lowpass, stimulus[begin - shift - search:end - shift + search])
            sample_lag = int(np.argmax(correlate(y, x, mode="valid", method="fft"))) - search
            r = float(np.corrcoef(x, y[search + sample_lag:search + sample_lag + len(x)])[0, 1])
            windows.append(dict(window_seconds=[start, start + 60], sample_lag_ms=1000 * sample_lag / rate, waveform_r=r))
        results[story] = dict(stim_file=stim_file, lag_seconds=best * 0.01, peak_correlation=float(max(scores)),
                              waveform_windows=windows,
                              gentle_seconds=len(gentle) / rate, stimulus_seconds=len(stimulus) / rate,
                              gentle_sha256=sha256(WORK / "gentle" / story / "a.wav"),
                              stimulus_sha256=sha256(WORK / "stimuli" / stim_file))
        print(f"{story:17s} lag={best * 0.01:+.2f}s envelope_r={max(scores):.3f} "
              + " ".join(f"[{w['sample_lag_ms']:+.2f}ms r={w['waveform_r']:.3f}]" for w in windows))
    write_json(WORK / "offset_evidence.json", dict(
        method=("log-RMS envelope (10 ms) cross-correlation over +/-5 s selects the coarse lag; acceptance requires "
                "two or three 60 s windows (60, 200, 400 s; as many as fit) whose <3.8 kHz waveform cross-correlation peaks within +/-1 ms of that lag "
                "(search +/-50 ms) with waveform r >= 0.85"),
        stories=results))
    bad = {story: value for story, value in results.items()
           if len(value["waveform_windows"]) < 2 or any(abs(w["sample_lag_ms"]) > 1 or w["waveform_r"] < 0.85
                                                       for w in value["waveform_windows"])}
    if bad:
        print("AUDIO MISMATCH:", bad)
        sys.exit(2)


def segments():
    """One record per prepared story segment, using the same resolver as narratives_prepare."""
    selected, rejected = selected_scans()
    events = json.loads((META / "event_meta.json").read_text())
    timing = json.loads((WORK / "timing_evidence.json").read_text())["groups"]
    mappings = event_mappings()
    seen = {}
    for scan in selected + rejected:
        for segment in resolve_segments(scan, events, mappings, TRIM_START, TRIM_END):
            key = f"{scan['task']}/{scan['condition']}"
            record = dict(name=segment["name"], family=segment["family"], task=scan["task"],
                          event=segment["event"], tr_indices=segment["tr_indices"],
                          gentle=gentle_name(scan["task"], scan["condition"]),
                          wav_start_seconds=timing[key]["first_onset_seconds"], stim_file=timing[key]["stim_file"])
            if seen.setdefault(segment["name"], record) != record:
                raise ValueError(f"Inconsistent segment definition: {segment['name']}")
    return [seen[name] for name in sorted(seen)]


def event_mappings():
    timing = json.loads((WORK / "timing_evidence.json").read_text())["groups"]
    mappings = {}
    for task, conditions in CONDITIONS.items():
        for condition in conditions:
            evidence = f"events/timing_{task}_{condition}.json"
            write_json(WORK / evidence, timing[f"{task}/{condition}"])
            mappings[f"{task}/{condition}"] = [dict(name=f"{task}_{condition}", family=task,
                                                   event_ref=[task, task], evidence=evidence)]
    return mappings


def word_frequency_source():
    import wordfreq
    path = Path(wordfreq.__file__).parent / "data" / "large_en.msgpack.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    from importlib.metadata import version
    return path, version("wordfreq")


def sentence_final(transcript, end_offset):
    index = end_offset
    while index < len(transcript) and transcript[index] in "\"'\u201d\u2019)]":
        index += 1
    return index < len(transcript) and transcript[index] in ".!?"


def build_segment_inputs(record, offsets, tools):
    from wordfreq import word_frequency, zipf_frequency
    name, onset, duration = record["name"], record["event"]["onset"], record["event"]["duration"]
    gentle_dir = WORK / "gentle" / record["gentle"]
    words = json.loads((gentle_dir / "align.json").read_text())["words"]
    transcript = (gentle_dir / "transcript.txt").read_text(encoding="utf-8")
    lag = offsets[record["gentle"]]["lag_seconds"]
    offset = lag + record["wav_start_seconds"] - onset * TR
    stop = duration * TR
    kept, dropped, clamped, previous_end = [], 0, 0, None
    for word in words:
        if word.get("case") != "success":
            continue
        gentle_start, gentle_end = float(word["start"]), float(word["end"])
        if previous_end is not None and gentle_start < previous_end:
            # Gentle often starts a word within float rounding of the previous end; keep the word, trim its onset.
            gentle_start = previous_end
            clamped += 1
        start, end = gentle_start + offset, gentle_end + offset
        if start < 0 or end > stop or end <= start:
            dropped += 1
            continue
        kept.append((word, start, end, gentle_start, gentle_end))
        previous_end = gentle_end
    folder = WORK / "segments" / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "words.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["story", "word", "onset_seconds", "end_seconds"])
        for word, _, _, gentle_start, gentle_end in kept:
            writer.writerow([name, word["word"], repr(gentle_start), repr(gentle_end)])
    kept = [item[:3] for item in kept]
    write_json(folder / "alignment.json", dict(
        certified=True, full_story=True, story=name, timebase="story_relative_seconds", offset_seconds=offset,
        coverage_start_seconds=0, coverage_stop_seconds=duration * TR,
        source=(f"Gentle forced alignment {S3}/stimuli/gentle/{record['gentle']}/align.json; offset = measured audio lag "
                f"{lag:+.2f}s + wav start {record['wav_start_seconds']}s (BIDS events.tsv) - event onset {onset} TR x 1.5s; "
                f"kept {len(kept)} success words inside the story event ({clamped} onsets trimmed to the previous word end), "
                f"dropped {dropped} outside the event or with no remaining duration"),
        kept_words=len(kept), dropped_words=dropped, clamped_onsets=clamped))

    edges = np.arange(duration + 1) * TR
    bins = lambda times: np.clip(np.ceil(np.asarray(times) / TR).astype(int) - 1, 0, duration - 1)
    columns = {name_: np.zeros(duration) for name_ in BASELINE_NAMES}
    ends = np.asarray([end for _, _, end in kept])
    end_bins = bins(ends)
    for (word, start, end), tr_bin in zip(kept, end_bins):
        text = word["word"].lower()
        columns["word_rate"][tr_bin] += 1
        columns["word_length"][tr_bin] += len(word["word"])
        columns["unigram_surprisal"][tr_bin] += -np.log2(word_frequency(text, "en", minimum=1e-8))
        columns["word_frequency"][tr_bin] += zipf_frequency(text, "en")
        columns["sentence_boundary"][tr_bin] += sentence_final(transcript, word["endOffset"])
        phone_end = start
        for phone in word.get("phones", []):
            phone_end += phone["duration"]
            columns["phoneme_or_syllable_rate"][bins([phone_end])[0]] += 1
        for tr_index in range(int(start // TR), min(duration, int(np.ceil(end / TR)))):
            overlap = min(end, edges[tr_index + 1]) - max(start, edges[tr_index])
            columns["voiced_fraction"][tr_index] += max(overlap, 0) / TR
    counts = np.maximum(columns["word_rate"], 1)
    columns["word_length"] /= counts
    columns["word_frequency"] /= counts
    columns["voiced_fraction"] = np.clip(columns["voiced_fraction"], 0, 1)

    audio, rate, pitch_times, pitch_values = tools[record["stim_file"]]
    audio_offset = onset * TR - record["wav_start_seconds"]
    for tr_index in range(duration):
        left, right = audio_offset + edges[tr_index], audio_offset + edges[tr_index + 1]
        samples = audio[max(0, int(left * rate)):max(0, int(right * rate))]
        columns["energy"][tr_index] = np.log10(np.mean(np.square(samples)) + 1e-10) if len(samples) else -10.0
        voiced = pitch_values[(pitch_times > left) & (pitch_times <= right) & (pitch_values > 0)]
        columns["pitch"][tr_index] = voiced.mean() if len(voiced) else 0.0
    baseline = np.column_stack([columns[name_] for name_ in BASELINE_NAMES]).astype(np.float32)
    np.savez(folder / "baseline.npz", baseline=baseline, baseline_names=np.array(BASELINE_NAMES),
             tr_indices=np.arange(onset, onset + duration, dtype=np.int64))
    frequency_path, frequency_version = tools["wordfreq"]
    gentle_hash, stim_hash, frequency_hash = (sha256(gentle_dir / "align.json"), sha256(WORK / "stimuli" / record["stim_file"]),
                                              sha256(frequency_path))
    text = f"{S3}/stimuli/gentle/{record['gentle']}/align.json"
    detail = {
        "voiced_fraction": (text, gentle_hash, "fraction of the TR covered by aligned word spans", "proportion"),
        "word_rate": (text, gentle_hash, "count of words whose end falls in the TR", "words/TR"),
        "word_length": (text, gentle_hash, "mean characters of words ending in the TR (0 if none)", "characters"),
        "phoneme_or_syllable_rate": (text, gentle_hash, "count of Gentle phones ending in the TR", "phones/TR"),
        "sentence_boundary": (f"{S3}/stimuli/gentle/{record['gentle']}/transcript.txt", sha256(gentle_dir / "transcript.txt"),
                              "count of words ending in the TR followed by . ! ? in the transcript", "count/TR"),
        "unigram_surprisal": (f"wordfreq {frequency_version} large_en", frequency_hash,
                              "sum of -log2 p(word), p floored at 1e-8", "bits"),
        "word_frequency": (f"wordfreq {frequency_version} large_en", frequency_hash,
                           "mean Zipf frequency of words ending in the TR (0 if none)", "Zipf"),
        "pitch": (f"{S3}/stimuli/{record['stim_file']}", stim_hash,
                  "mean Praat autocorrelation F0 (parselmouth, 75-500 Hz, 10 ms) over voiced frames in the TR, 0 if none", "Hz"),
        "energy": (f"{S3}/stimuli/{record['stim_file']}", stim_hash,
                   "log10 mean squared amplitude (16 kHz mono, full scale) in the TR", "log10 power"),
    }
    write_json(folder / "baseline.json", dict(
        story=name, provenance="real_narratives", tr_seconds=TR, causal_windows=True,
        features={key: dict(source=value[0], source_sha256=value[1], method=value[2], units=value[3], causal=True)
                  for key, value in detail.items()},
        unigram_reference=dict(corpus=f"wordfreq {frequency_version} 'large' English list (multi-source web/subtitle/book corpora)",
                               source_sha256=frequency_hash, independent_of_evaluation_corpus=True,
                               estimated_on_narratives=False)))
    return dict(name=name, kept=len(kept), dropped=dropped, offset=offset)


def step_features():
    import parselmouth
    offsets = json.loads((WORK / "offset_evidence.json").read_text())["stories"]
    records = segments()
    write_json(WORK / "segments.json", dict(schema_version=1, provenance="real_narratives",
                                            stories=[dict(name=r["name"], family=r["family"], task=r["task"],
                                                          event=r["event"], tr_indices=r["tr_indices"]) for r in records]))
    tools = {"wordfreq": word_frequency_source()}
    for stim_file in sorted({record["stim_file"] for record in records}):
        audio, rate = mono(WORK / "stimuli" / stim_file)
        pitch = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=rate).to_pitch_ac(
            time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        tools[stim_file] = (audio, rate, np.asarray(pitch.xs()), np.asarray(pitch.selected_array["frequency"]))
        print(f"audio {stim_file}: {len(audio) / rate:.0f}s, voiced frames {(tools[stim_file][3] > 0).mean():.2f}", flush=True)
    for record in records:
        print(build_segment_inputs(record, offsets, tools), flush=True)

    model_dir = WORK / "model" / "gpt2"
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot = HF_SNAPSHOT / GPT2_REVISION
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if not (model_dir / name).is_file():
            shutil.copyfile((snapshot / name).resolve(), model_dir / name)
    weights = {"model.safetensors": sha256(model_dir / "model.safetensors")}
    write_json(WORK / "model" / "pretrained.json", dict(
        model="gpt2", pretrained=True, revision=GPT2_REVISION,
        source=f"Hugging Face gpt2 (openai-community/gpt2) revision {GPT2_REVISION}, copied from local HF cache",
        weights_sha256=hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()))

    import narratives_features
    model_files = sorted(path for path in model_dir.iterdir() if path.is_file())
    for record in records:
        folder = WORK / "segments" / record["name"]
        output = WORK / "features" / f"{record['name']}.npz"
        if output.is_file():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        consumed = [WORK / "segments.json", folder / "words.csv", folder / "alignment.json", folder / "baseline.npz",
                    folder / "baseline.json", WORK / "model" / "pretrained.json", *model_files]
        config = dict(schema_version=1, provenance="real_narratives", dataset="../../segments.json", story=record["name"],
                      tr_seconds=TR, context_window_tr=5,
                      alignment=dict(path="words.csv", evidence="alignment.json", timebase="story_relative_seconds",
                                     offset_seconds=json.loads((folder / "alignment.json").read_text())["offset_seconds"]),
                      baseline=dict(path="baseline.npz", provenance="baseline.json"),
                      model=dict(path="../../model/gpt2", revision=GPT2_REVISION, evidence="../../model/pretrained.json"),
                      input_sha256={Path(_relative(path, folder)).as_posix(): sha256(path) for path in consumed})
        write_json(folder / "features_config.json", config)
        metadata = narratives_features.extract(folder / "features_config.json", output)
        print(f"features {record['name']}: {len(metadata['tr_indices'])} TRs, empty windows "
              f"{len(metadata['empty_context_tr_indices'])}", flush=True)


def _relative(path, base):
    import os
    return os.path.relpath(path, base)


def step_config():
    import nibabel as nib
    selected, _ = selected_scans()
    records = segments()
    labels = {}
    for hemisphere in "LR":
        table = nib.load(str(META / f"tpl-fsaverage6_hemi-{hemisphere}_desc-MMP_dseg.label.gii")).labeltable.get_labels_as_dict()
        lookup = {value: key for key, value in table.items()}
        labels[hemisphere] = {network: [int(lookup[f"{hemisphere}_{parcel}_ROI"]) for parcel in parcels]
                              for network, parcels in NETWORK_PARCELS.items()}
    write_json(WORK / "membership.json", dict(
        atlas="Glasser HCP-MMP1.0 on fsaverage6 (Narratives afni-nosmooth tpl-fsaverage6 MMP dseg)",
        parcels=NETWORK_PARCELS, label_ids=labels,
        sources={"MD": "Assem, Glasser, Van Essen & Duncan (2020) Cerebral Cortex, core MD MMP parcels",
                 "LANG": "Perisylvian MMP parcels as declared in narratives_gate_config.json (PSL/SFL were listed as "
                         "PSROI/SFROI, a label-string replacement artifact for the same parcels)",
                 "EAC": "Nastase et al. (2021) roi_masks.py early auditory parcels"}))
    mappings = event_mappings()
    consumed = [META / "task_meta.json", META / "scan_exclude.json", META / "event_meta.json",
                META / "tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii", META / "tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii",
                WORK / "meshes" / "tpl-fsaverage_hemi-L_den-41k_sphere.surf.gii",
                WORK / "meshes" / "tpl-fsaverage_hemi-R_den-41k_sphere.surf.gii", WORK / "membership.json"]
    consumed += [WORK / mapping[0]["evidence"] for mapping in mappings.values()]
    consumed += [WORK / "features" / f"{record['name']}.npz" for record in records]
    cache = META / "processed_cache"
    from narratives_prepare import cache_path
    gifti = [cache_path(cache, scan, hemisphere) for scan in selected for hemisphere in "LR"]
    missing = [path for path in gifti if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} BOLD files not downloaded yet, e.g. {missing[0]}")
    with ThreadPoolExecutor(8) as pool:
        digests = dict(zip(consumed + gifti, pool.map(sha256, consumed + gifti)))
    config = dict(schema_version=1, provenance="real_narratives", space="fsaverage6", tasks=TASKS,
                  metadata={"task_meta": "../narratives_meta/task_meta.json",
                            "scan_exclude": "../narratives_meta/scan_exclude.json",
                            "event_meta": "../narratives_meta/event_meta.json"},
                  cache_root="../narratives_meta/processed_cache",
                  atlases={hemisphere: f"../narratives_meta/tpl-fsaverage6_hemi-{hemisphere}_desc-MMP_dseg.label.gii" for hemisphere in "LR"},
                  geometry={hemisphere: f"meshes/tpl-fsaverage_hemi-{hemisphere}_den-41k_sphere.surf.gii" for hemisphere in "LR"},
                  network_labels=labels, membership_evidence="membership.json", trim_start=TRIM_START, trim_end=TRIM_END,
                  event_mappings=mappings,
                  feature_files={record["name"]: f"features/{record['name']}.npz" for record in records},
                  input_sha256={Path(_relative(path, WORK)).as_posix(): digest for path, digest in digests.items()})
    write_json(WORK / "prepare.json", config)
    print(f"prepare.json: {len(config['input_sha256'])} pinned inputs, {len(records)} segments")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["evidence", "stimuli", "offsets", "features", "config"])
    step = parser.parse_args().step
    WORK.mkdir(parents=True, exist_ok=True)
    {"evidence": step_evidence, "stimuli": step_stimuli, "offsets": step_offsets,
     "features": step_features, "config": step_config}[step]()


if __name__ == "__main__":
    main()
