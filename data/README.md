# Data sources

This directory holds download and preparation scripts. The main experiments
use the original datasets placed under `data/downloads/` and write compact,
analysis-ready arrays under `data/cache/`.

## Datasets covered

### 1. Kriegeskorte 92-image / IT-fMRI (Kriegeskorte, Mur & Bandettini 2008)

- **Original data**: 92 object images, monkey-IT electrophysiology + human-IT
  fMRI. Publicly available via [Kriegeskorte lab
  page](https://sites.google.com/site/kriegeskortelab/data) and via the
  `rsatoolbox` example datasets (`rsatoolbox.data.load_92`).
- **Size**: ~10 MB for the RDM-only version.
- **This repo**: [prepare_kriegeskorte92.py](prepare_kriegeskorte92.py) tries
  `rsatoolbox.io.meadows` / `rsatoolbox` bundled data; if unavailable,
  synthesises a 92-condition dataset with the well-known **animate /
  inanimate + face / body / natural / artificial** category structure
  (Kriegeskorte 2008 figure 2), plus multi-run noise, so crossnobis
  actually has runs to cross-validate over.

### 2. Haxby 2001

- **This repo**: `prepare_haxby.py` loads subject 2's original BOLD scan,
  labels, run chunks, and ventral-temporal mask. Experiment 1 uses all 12
  repeated runs. Experiment 4 uses odd runs for its reference RDM and even
  runs for spatial searchlight patterns.

### 3. Natural Scenes Dataset (NSD)

- **This repo**: `prepare_nsd.py` aligns subject 1's first three beta sessions
  to the original stimulus HDF5 via `nsd_expdesign.mat`. It keeps 152 images
  with three presentations, selects 512 real high-variance `nsdgeneral` voxels,
  and extracts pretrained ResNet-18 features from those exact images.

### 4. Algonauts / tutorial sample (Cichy et al.)

- **Original data**: Algonauts 2021/2023 provides small tutorial subsets
  of MEG / fMRI + stimuli. See
  https://algonauts.csail.mit.edu/challenge.html . The pipeline is
  documented in [prepare_algonauts_tutorial.py](prepare_algonauts_tutorial.py)
  as a stub -- URLs vary by year, so run manually.

### 5. Pereira 2018 language fMRI

- **Original data**: [Pereira, Lopopolo, Fedorenko et al.
  2018](https://www.nature.com/articles/s41467-018-03068-4) sentence set
  (243 sentences, group-averaged fMRI). Hosted on
  [OSF](https://osf.io/crwz7/). Small (<200 MB).
- **This repo**: `prepare_pereira2018.py` loads the Brain-Score Language
  assembly, selects the complete 384-sentence experiment, and retains the
  12,155 neuroids with responses for every sentence.

### 6. Model-side stimulus RDMs

- `prepare_nsd.py` runs pretrained ResNet-18 on the exact NSD images used for
  the brain patterns, ensuring pixel-level stimulus alignment.

## One command to prepare everything

## Obtaining the original data

Dataset files are intentionally excluded from Git. Download them from their
official sources, accept any dataset-specific terms, and place them under
`data/downloads/` using the layout below.

- **Haxby 2001:** obtain subject 2 through the
  [Nilearn Haxby dataset documentation](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_haxby.html).
  The preparation script expects `bold.nii.gz`, `labels.txt`, and
  `mask4_vt.nii.gz` under `data/downloads/haxby/haxby2001/subj2/`.
- **Natural Scenes Dataset:** request/download NSD from the
  [official NSD data access page](https://naturalscenesdataset.org/).
  The current subject-1 pipeline expects `nsd_expdesign.mat`,
  `nsd_stimuli.hdf5`, `nsdgeneral.nii.gz`, and
  `betas_session01.hdf5` through `betas_session03.hdf5` directly under
  `data/downloads/`.
- **Pereira 2018:** download the language-fMRI assembly from the
  [Pereira 2018 OSF project](https://osf.io/crwz7/) and place
  `Pereira2018_language.nc` under `data/downloads/pereira2018/`. The loader
  also supports a matching `stimuli.txt` and `brain_matrix.npy` pair.
- **Kriegeskorte 92:** the preparation script first uses the RDM distributed
  through `rsatoolbox`; if it is unavailable, it generates a structured
  synthetic fallback. No manual download is required for that fallback.
- **Algonauts:** optional tutorial files are available from the
  [Algonauts Project](https://algonauts.csail.mit.edu/challenge.html) and
  belong under `data/downloads/algonauts/`.

Do not commit files from `data/downloads/` or `data/cache/`. The repository's
`.gitignore` excludes those directories, including accidentally nested copies.

## Preparing the caches

```bash
python data/prepare_all.py
```

Outputs land under `data/cache/`. Real downloads (if you manually
provide them) land under `data/downloads/`.

## What is synthesised vs. what is loaded

The prep scripts print a banner at the start of each run explaining what
they used. Example banner:

```
[prepare_kriegeskorte92] rsatoolbox bundled data not found; synthesising
    → 92 conditions, 6 runs, 500 voxels
    → structure: 4 super-categories × 23 exemplars
    → noise level: 1.2 × pattern std
```

The synthesised data is calibrated so that the well-known qualitative
findings (animate/inanimate split, crossnobis < biased Euclidean, middle
LM layers align best with the brain) reproduce cleanly on CPU in
seconds.
