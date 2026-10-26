# Narratives Vertex-Wise Stage A

## Status

Update 2026-10-26: real inputs are built and the specification is frozen.
- Config v1.5.0 was frozen in commit `57feb11` and superseded before any
  brain-to-model fitting by v1.5.1 (commit `dde57b8`; listener conditions of a
  shared stimulus are pooled). Primary pool: 15 tasks / 19 segments (schema
  excluded: no public word alignments). See the changelog for every input decision.
- The first real primary run uses v1.5.1.
- `narratives_download.py`: 1,242 first-exposure bilateral GIFTIs (146.7 GB).
- `narratives_build_inputs.py`: per-scan BIDS timing evidence (identical within
  every task/condition, 621 scans), 0 ms Gentle-to-stimulus audio lag for all 17
  stimulus files, word timings, nine-column comparator, local GPT-2 revision
  `607a30d…` layer-9 features, and the hash-pinned `prepare.json` (1,274 inputs).
- On Windows, launch long runs through `conda.exe run -n ns-implem`; calling the
  environment's `python.exe` directly fails to load the BLAS/LAPACK DLLs.

## Status (2026-10-25)

The separate preparation, feature and primary-gate engine are implemented and
software-tested. No real Narratives primary encoding experiment has been run.
Synthetic test fixtures are not scientific findings. The original numbered
scripts still implement the Natural Stories fROI workflow and are unchanged.

Validation on 2026-10-25: 95 software tests passed in the `ns-implem` interpreter,
including strict temporary GIFTI preparation-to-loader integration, known-signal
and null examples, leakage boundaries, hash/story mismatches and freeze guards.
The read-only real-cache smoke loaded 10 participants x 266 TRs x 3,405 LH MD
vertices, with finite ISC and verified normalization. Mean vertex pairwise ISC
was 0.00808019 for this zero-ring sample. This is not parcel ISC, GPT-2 prediction
performance, an erosion-mask primary result or a gate verdict. It does not
certify the uncollected full pool or calibrate achieved inference error rates.

The notebook is descriptive EDA, not the source of model-ready Stage A arrays.
This pipeline preserves bilateral fsaverage6 vertex identities, participants,
original TR indices and provenance. It never substitutes parcels for MD targets,
fabricates missing features, replaces excluded first exposures with repeats, or
downloads the full dataset automatically.

## Components

- `narratives_prepare.py`: local metadata, first-exposure/exclusion policy,
  explicit condition/schema timing, supplied bilateral geometry, atlas masks,
  one-ring erosion, segment trims/detrending/z-scoring, pinned input hashes.
  LANG parcel identities are retained as nuisance covariates, not MD outcomes.
- `narratives_features.py`: local pretrained frozen GPT-2, layer 9, completed-word
  causal five-TR contexts, mean attended-token pooling, separately preserved lag
  history. Certified timing and exogenous extended baseline columns are required.
  Long/empty contexts fail explicitly; no silent truncation or fake EOS stimulus.
- `narratives_stage_a.py`: train-family-only pooled ISC/BH vertex selection;
  fixed-penalty ridge with train-only feature scaling; out-of-family encoding;
  disjoint identity-stable subject-half nuisance regression in both directions;
  refitted coherent circular-shift nulls; Gate 1 two-test BH, singleton Gate 2;
  Gate 3 family/block conditional bootstrap. Output includes folds, support,
  p/q, actual null statistics and reference-only Shain anchors.

Full module docstrings describe the separate preparation and feature JSON
contracts. They are not the gate config. Every consumed input needs an actual
SHA256 pin; hashes establish byte identity, not scientific validity of labels or
timing evidence. Human review of that evidence remains necessary.

## Obtaining The Data

No data is stored in this repository. Everything is public and is fetched into
ignored local folders (about 150 GB for the primary pool).

1. Metadata into `data/narratives_meta/`: `participants.tsv`, `participants.json`,
   `dataset_description.json` and `README` from
   `https://raw.githubusercontent.com/OpenNeuroDatasets/ds002345/1.1.4/`;
   `task_meta.json`, `scan_exclude.json` and `event_meta.json` from
   `https://raw.githubusercontent.com/snastase/narratives/master/code/`; and the
   Glasser MMP fsaverage6 label files
   `tpl-fsaverage6_hemi-{L,R}_desc-MMP_dseg.label.gii` from
   `https://s3.amazonaws.com/fcp-indi/data/Projects/narratives/derivatives/afni-nosmooth/tpl-fsaverage6/`.
   The first cell of `eda_narratives_processed.ipynb` downloads the metadata files.
2. BOLD: first-exposure, bilateral, unsmoothed, denoised fsaverage6 surface data
   (`derivatives/afni-nosmooth/sub-*/func/*_space-fsaverage6_hemi-{L,R}_desc-clean.func.gii`)
   from the public `fcp-indi` S3 bucket. The resumable downloader selects scans
   with the same first-exposure and exclusion policy as preparation:

   ```powershell
   python -B Idea1POC/narratives_download.py --tasks pieman shapesphysical shapessocial notthefallintact milkyway bronx forgot black prettymouth sherlock merlin 21styear tunnel slumlordreach lucy
   ```

3. Stimuli, timing evidence, features and the hash-pinned preparation config:
   `narratives_build_inputs.py` downloads BIDS event files, stimulus WAVs, Gentle
   word alignments and the TemplateFlow fsaverage6 sphere, then builds everything
   under `data/narratives_stage_a/`. Run its steps in order:

   ```powershell
   python -B Idea1POC/narratives_build_inputs.py evidence
   python -B Idea1POC/narratives_build_inputs.py stimuli
   python -B Idea1POC/narratives_build_inputs.py offsets
   python -B Idea1POC/narratives_build_inputs.py features
   python -B Idea1POC/narratives_build_inputs.py config
   ```

   The `features` step needs pretrained GPT-2 weights available locally
   (Hugging Face `gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`).

## Local Workflow

Run from the repository root in the `ns-implem` environment:

```powershell
python -B Idea1POC/narratives_prepare.py --inventory
python -B -m unittest discover -s Idea1POC -p "test_narratives*.py" -v
python -B Idea1POC/validate_narratives_cache.py
```

Inventory exit code 2 means prerequisites are missing, not a successful analysis.
The current cache audit found 20 selected left-hemisphere files and no selected
right-hemisphere files across all metadata tasks. That is not the complete
declared 16-task Regime C pool. Real meshes, condition/schema mappings, certified
alignment, local pretrained GPT-2 weights, and complete comparator inputs are
also required. Do not infer that the cached ten-person Pie Man EDA meets these.

For actual preparation, supply a reviewed, hash-pinned preparation config:

```powershell
python -B Idea1POC/narratives_prepare.py --config <prep.json> --output <bold-dataset>
python -B Idea1POC/narratives_features.py --config <story-features.json> --output <story.npz>
```

Preparation without features publishes a noneligible BOLD manifest that the
feature extractor can use for exact segment identity. After extracting each
story, add those feature paths/hash pins to the preparation config and prepare
into a NEW final directory. The strict final handoff binds feature metadata to
story identity, original TRs, timing, pretrained model and baseline sources.

```powershell
python -B Idea1POC/narratives_stage_a.py --dataset <final/dataset.json> --config Idea1POC/narratives_gate_config.json --check-inputs
```

Input checks do not freeze or run the experiment. Review the config, then freeze
and commit it before a real run.
The CLI refuses placeholder freeze/commit fields and existing result paths.
The timezone-aware freeze time must be in the past. Commit the frozen config
and the three engine modules, then record that full commit ID in
`git_commit_before_first_run`. Only this self-referential pointer is excluded
from the committed-config comparison. Every other config field and the engine
sources must match that commit; no commits are made automatically.

```powershell
python -B Idea1POC/narratives_stage_a.py --dataset <final/dataset.json> --config Idea1POC/narratives_gate_config.json --output <new-result.json>
```

## Estimator And Limits

Only training families determine vertex masks. No selected LANG or MD vertices
in any outer fold produces `NOT_EVALUABLE`, not zero-filled predictions. Primary
inputs must include all configured tasks and both hemispheres; a subset is not
silently promoted to primary evidence. Segment feature lags use original TR
history. Two additional leading BOLD rows are omitted everywhere for LANG lags.

Ridge penalties are fixed, not tuned: encoding 100, nuisance 1. Nuisance training
targets are inner-family cross-fitted; held-out nuisance targets use all outer
training families. Residuals are averaged over subject splits/directions, then
standardized per segment. Different nuisance training sizes remain a limitation.
Cross-subject predictors reduce same-subject nuisance but do not repair noisy
LANG measurement or establish neural/computational specificity.

Gate 1/2 concatenate standardized observed/predicted segments for each vertex
where that vertex was training-selected, then average per-vertex Pearson r.
Vertices have differing test coverage; exact outer-fold supports are reported.
Gate 3 instead averages within-story mean-vertex paired differences equally over
families. Its interval conditions on the fitted outputs and their fixed
within-fold supports. It does not account for selection/refitting uncertainty.

Each encoding-null draw shifts a segment's whole lagged GPT-2 design coherently
across all folds and targets, and refits the readout. BOLD-only selection and
nuisance fits stay fixed. Independent segment shifts assume segment-wise
stationarity, including within related task families. Circular shifts do not
make arbitrary nonstationary fMRI exchangeable by definition.

Gate 4, noise-ceiling normalization/CI, secondary layers, alternative masks,
smoothed data, Regime B and the static/untrained/shuffled comparator ladder are
not implemented. A primary-gate pass is explicitly pending those diagnostics,
not an unconditional Stage B GO. No achieved power or Narratives expected
correlation is asserted by this implementation.
