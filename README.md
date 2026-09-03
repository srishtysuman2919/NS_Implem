# NS_Implem — Representational Comparison Methods, From Scratch

A study repository that implements and compares the core representational-comparison
methods from the RSA (Representational Similarity Analysis) citation lineage, so I can
understand each method **mechanically** rather than just conceptually.

Everything runs on CPU. From-scratch NumPy/SciPy is the primary implementation;
[`rsatoolbox`](https://rsatoolbox.readthedocs.io) and library helpers are only used to
sanity-check my own code.

## Repository layout

```
methods/         one file per method, from-scratch implementation
data/            download / prep scripts + dataset README
experiments/     scripts that run all methods on a shared dataset and compare them
results/         saved comparison outputs (tables + plots)
```

## Methods (one paragraph each)

Each paragraph summarises **the paper, the problem it solves, how it differs from its
predecessors, and a link.**

### 1. Classic RSA — Kriegeskorte, Mur & Bandettini (2008)
The founding paper of RSA. Introduces the **Representational Dissimilarity Matrix
(RDM)**: for each pair of stimuli, measure the dissimilarity between their neural
response patterns (typically `1 − Pearson r`), producing a stimulus×stimulus matrix.
Comparing brains, models and species then reduces to comparing RDMs (Spearman ρ over
the upper triangles). Fixes the previous approach of comparing individual voxels or
individual units across systems that live in incompatible feature spaces.
File: [methods/classic_rsa.py](methods/classic_rsa.py). Paper: https://doi.org/10.3389/neuro.06.004.2008

### 2. Searchlight RSA — Kriegeskorte, Goebel & Bandettini (2006)
Introduced the **searchlight**: instead of asking "is information present anywhere in
this ROI?", slide a small sphere through the volume and compute a local RDM (or a
local decoding score) at every centre voxel. Produces a whole-brain map of *where*
information is represented, without pre-committing to an ROI. Predecessors could only
report a single number per ROI. File: [methods/searchlight_rsa.py](methods/searchlight_rsa.py).
Paper: https://doi.org/10.1073/pnas.0600244103

### 3. Crossnobis — Walther et al. (2016); Diedrichsen et al. (2021)
Classic RSA distances (correlation, Euclidean) are **positively biased by noise** —
even identical conditions produce a nonzero distance. Crossnobis (cross-validated
Mahalanobis) fixes this by (a) noise-normalising patterns with the residual covariance
and (b) computing distances as an inner product between independent runs so the noise
terms have zero expectation. The distance is then an **unbiased estimator of the true
pattern separation** (can even be negative). File: [methods/crossnobis.py](methods/crossnobis.py).
Papers: https://doi.org/10.1016/j.neuroimage.2015.12.012, https://doi.org/10.1016/j.neuroimage.2020.117382

### 4. Centered Kernel Alignment (CKA) — Kornblith et al. (2019)
RSA-with-correlation is not invariant to invertible linear transforms of features, and
raw linear CCA is dominated by low-variance directions. **Linear CKA** measures
similarity between two representations as the normalized Frobenius inner product of
their centered gram matrices `⟨K, L⟩_F / (‖K‖_F ‖L‖_F)`. It is invariant to
orthogonal transforms and isotropic scaling but sensitive to the covariance structure
that CCA discards. RBF-CKA extends this to nonlinear kernels.
File: [methods/cka.py](methods/cka.py). Paper: https://arxiv.org/abs/1905.00414

### 5. Procrustes / Riemannian Shape Distance — Williams et al. (2021)
Argues that CKA is a *similarity score*, not a proper metric (fails the triangle
inequality), which makes clustering, averaging, and statistical testing of
representations ill-defined. Introduces the **generalized shape metric**: align two
neural response matrices with an orthogonal Procrustes rotation, then take the
Riemannian distance on the shape manifold. Result is a genuine metric that respects
symmetry and triangle inequality. File: [methods/shape_distance.py](methods/shape_distance.py).
Paper: https://arxiv.org/abs/2110.14739

### 6. Linear Ridge Encoding Models — Schrimpf et al. (Brain-Score, 2018 / 2020)
A completely different family: instead of comparing structural summaries (RDMs, gram
matrices) of two systems, **regress brain responses directly from model features**
using ridge regression, and score by held-out predictivity. Brain-Score demonstrated
that this correlates with core-object-recognition performance and drove a decade of
"deep-nets-as-models-of-cortex" work. Included here as the non-similarity baseline.
File: [methods/ridge_encoding.py](methods/ridge_encoding.py). Paper: https://www.biorxiv.org/content/10.1101/407007v2

### 7. Pattern Component Modelling (PCM) — Diedrichsen & Kriegeskorte (2017)
Where RSA compares systems at the level of **pairwise distances**, PCM compares them
at the level of the **second-moment matrix `G = U Uᵀ / P`** of the true response
patterns (where `U` is conditions × voxels). Different feature models predict
different `G`'s; PCM fits `G` as a mixture and does model comparison via marginal
likelihood. This module implements the minimal core: unbiased `G`-estimation and
model comparison by log-likelihood over a fixed noise model.
File: [methods/pcm.py](methods/pcm.py). Paper: https://doi.org/10.1371/journal.pcbi.1005508

### 8. LLM ↔ Brain Alignment — Jozwik et al. (2025) and lineage
Applies the same toolkit to language: extract layer-wise hidden states from a small
pretrained LM (GPT-2 small) on a public sentence set, build a per-layer RDM, and
compare against a brain RDM from a language fMRI dataset (Pereira 2018 sentence
stimuli). Reproduces the community-wide observation that **middle transformer layers
align best with high-level language cortex.** Uses the same RSA / CKA /
ridge-encoding functions from `methods/` so the toolkit is unified across vision-
and language-NeuroAI. File: [methods/llm_alignment.py](methods/llm_alignment.py).

## Paper references

These are the primary papers behind the methods implemented in `methods/`:

1. **Classic RSA** — Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008).
  “Representational similarity analysis — connecting the branches of systems
  neuroscience.” *Frontiers in Systems Neuroscience, 2*, 4.
  [DOI: 10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)
2. **Searchlight analysis** — Kriegeskorte, N., Goebel, R., & Bandettini,
  P. A. (2006). “Information-based functional brain mapping.” *Proceedings of
  the National Academy of Sciences, 103*(10), 3863–3868.
  [DOI: 10.1073/pnas.0600244103](https://doi.org/10.1073/pnas.0600244103)
3. **Crossnobis distance** — Walther, A., Nili, H., Ejaz, N., Alink, A.,
  Kriegeskorte, N., & Diedrichsen, J. (2016). “Reliability of dissimilarity
  measures for multi-voxel pattern analysis.” *NeuroImage, 137*, 188–200.
  [DOI: 10.1016/j.neuroimage.2015.12.012](https://doi.org/10.1016/j.neuroimage.2015.12.012)
4. **Crossnobis inference** — Diedrichsen, J., Provost, S., & Zareamoghaddam,
   H. (2021). “On the distribution of crossvalidated Mahalanobis distances.”
   *NeuroImage, 227*, 117382.
   [DOI: 10.1016/j.neuroimage.2020.117382](https://doi.org/10.1016/j.neuroimage.2020.117382)
5. **Centered Kernel Alignment** — Kornblith, S., Norouzi, M., Lee, H., &
  Hinton, G. (2019). “Similarity of neural network representations revisited.”
  *Proceedings of the 36th International Conference on Machine Learning*.
  [PMLR paper](https://proceedings.mlr.press/v97/kornblith19a.html)
6. **Generalized shape metrics** — Williams, A. H., Kunin, D., Dapello, J.,
  Ganguli, S., Kolter, J. Z., & Tanaka, H. (2021). “Generalized shape metrics
  on neural representations.” *Advances in Neural Information Processing
  Systems 34*.
  [arXiv: 2110.14739](https://arxiv.org/abs/2110.14739)
7. **Ridge encoding / Brain-Score** — Schrimpf, M., Kubilius, J., Hong, H.,
  Majaj, N. J., Rajalingham, R., Issa, E. B., Kar, K., Bashivan, P., Prescott-
  Roy, J., Schmidt, K., Nayebi, A., Bear, D., Yamins, D. L. K., & DiCarlo,
  J. J. (2018). “Brain-Score: Which artificial neural network for object
  recognition is most brain-like?”
  [bioRxiv: 10.1101/407007](https://doi.org/10.1101/407007)
8. **Pattern Component Modelling** — Diedrichsen, J., & Kriegeskorte, N.
  (2017). “Representational models: A common framework for understanding
  encoding, pattern-component, and representational-similarity analysis.”
  *PLOS Computational Biology, 13*(4), e1005508.
  [DOI: 10.1371/journal.pcbi.1005508](https://doi.org/10.1371/journal.pcbi.1005508)

The LLM–brain module combines RSA, CKA, and ridge encoding rather than
implementing a separate estimator. Its layer-wise analysis follows this research
lineage:

9. Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher,
  N., Tenenbaum, J. B., & Fedorenko, E. (2021). “The neural architecture of
  language: Integrative modeling converges on predictive processing.” *PNAS,
  118*(45), e2105646118.
  [DOI: 10.1073/pnas.2105646118](https://doi.org/10.1073/pnas.2105646118)
10. Caucheteux, C., & King, J.-R. (2022). “Brains and algorithms partially
   converge in natural language processing.” *Communications Biology, 5*, 134.
   [DOI: 10.1038/s42003-022-03036-1](https://doi.org/10.1038/s42003-022-03036-1)
11. Toneva, M., & Wehbe, L. (2019). “Interpreting and improving natural-language
   processing (in machines) with natural language-processing (in the brain).”
   *Advances in Neural Information Processing Systems 32*.
   [arXiv: 1905.11833](https://arxiv.org/abs/1905.11833)

## What each experiment answers

- [experiments/01_rdm_agreement.py](experiments/01_rdm_agreement.py) — On the 12
  repeated runs from Haxby 2001 subject 2, do classic RSA, crossnobis, and PCM
  agree with an independent odd-run reference RDM?
- [experiments/02_model_vs_brain.py](experiments/02_model_vs_brain.py) — Across
  152 exactly matched NSD images, does the ranking of pretrained ResNet-18 layers
  against subject-1 fMRI change between classic RSA, crossnobis-RSA, CKA, shape
  distance, and ridge encoding?
- [experiments/03_llm_layerwise.py](experiments/03_llm_layerwise.py) — Layer-wise
  alignment curve for DistilGPT-2 against the Pereira 2018 384-sentence language
  fMRI experiment, using RSA, CKA, and ridge encoding.
- [experiments/04_searchlight.py](experiments/04_searchlight.py) — Where in the
  Haxby subject-2 volume does the even-run local geometry match the independent
  odd-run reference RDM?

## Current data and results

The checked-in `data/cache/` directory currently contains analysis-ready data:

| Cache | Current source | Shape / scope |
|---|---|---|
| `haxby.npz` | Haxby 2001 subject 2 | 12 runs, 8 conditions, 464 VT voxels |
| `nsd_model_brain.npz` | NSD subject 1 + pretrained ResNet-18 | 3 repeats, 152 matched images, 512 voxels, 5 layers |
| `pereira2018.npz` | Pereira 2018 Brain-Score Language assembly | 384 sentences, 12,155 complete neuroids |
| `kriegeskorte92.npz` | Kriegeskorte hIT RDM + synthetic repeated patterns | 92 conditions, 6 runs |

The four experiments have already produced CSV tables and PNG figures under
[`results/`](results/). A consolidated snapshot is available as
[`results/experimental_results_report.pdf`](results/experimental_results_report.pdf).

As of 3 September 2026, all eight method self-tests pass. The real-data results
should be treated as exploratory: the saved NSD model-brain scores are weak, and
the Pereira layer-wise curves do not currently show a clear middle-layer peak.

## Quick start

The repo ships a two-file conda setup: a slim base environment for the RSA /
CKA / crossnobis / ridge-encoding methods, plus an optional overlay that adds
the CPU-only deep-learning stack (torch / torchvision / transformers) used by
the LLM and vision-model RDM steps.

### 1. Create the base environment

```bash
conda env create -f environment.yml
conda activate ns-implem
```

This is enough to run the method self-tests and experiments from the existing
analysis-ready caches:

- every self-test in `python experiments/00_run_all_selftests.py`
- `experiments/01_rdm_agreement.py`
- `experiments/02_model_vs_brain.py`
- `experiments/04_searchlight.py`

### 2. (Optional) Add the deep-learning overlay

Needed for `experiments/03_llm_layerwise.py`, for rebuilding the matched NSD
ResNet-18 cache, and for the legacy model-RDM preparation script:

```bash
conda env update -f environment-full.yml
```

The overlay installs the CPU build of PyTorch (`cpuonly`) so nothing pulls a
CUDA runtime.

> Prefer mamba? Substitute `mamba env create` / `mamba env update` — the YAML
> files work unchanged.

> Prefer pip? `requirements.txt` / `requirements-optional.txt` list the same
> packages if you want to install into an existing venv instead.

### Verify + run

```bash
# 1. sanity-check every method module against rsatoolbox / scipy
python experiments/00_run_all_selftests.py

# 2. run the comparison experiments from the existing caches
python experiments/01_rdm_agreement.py
python experiments/02_model_vs_brain.py
python experiments/03_llm_layerwise.py       # requires the -full overlay
python experiments/04_searchlight.py
```

Results are written to [results/](results/) as CSV tables and PNG plots.

### Rebuild the data caches

The current real-data pipeline expects the original Haxby, NSD, and Pereira
files under `data/downloads/`; see [data/README.md](data/README.md) for their
layout and provenance. With those files present and the full environment
installed, rebuild all caches with:

```bash
python data/prepare_all.py
```

Preparation is deterministic apart from external library/model changes. It can
be substantially slower than running the cached experiments because NSD
preparation reads the original beta volumes and extracts ResNet-18 activations.

## Design notes

- **Correctness > speed.** Loops are unrolled where a vectorised version would hide
  the math. Vectorised paths are provided only when they are equally readable.
- **Sanity checks against `rsatoolbox` / `scipy`** are in each module's `_selftest`
  block; run `python -m methods.classic_rsa` etc. to verify.
- Every method file starts with a paper docstring and ends with a **"what shortcoming
  of the previous method does this one address?"** note.
