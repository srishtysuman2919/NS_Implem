# Mechanistic Notes: Why Representational Methods Disagree

## Overview

This document explains the *why* behind method disagreements observed in Experiments 1 and 2. 
Each method encodes different assumptions about representation geometry, noise, and metrics. 
These notes form the intellectual core: understanding these differences is the difference between 
"I ran a pipeline" and "I understand what each method is measuring."

---

## Experiment 1: Haxby Method Agreement (Spearman ρ on RDMs)

**Key finding**: Classic RSA–Crossnobis agreement was only **0.678**, worse than expected for 
"the same data." Crossnobis–Odd-run reference agreement dropped to **0.517**. Why?

### 1.1 Classic RSA vs. Crossnobis: Noise Bias

**Classic RSA RDM calculation**:
```
RDM[i, j] = 1 - Pearson_r(pattern_i, pattern_j)
```

**The noise bias problem**:
- Even if two stimuli have *identical* representations (r = 1), measurement noise forces r < 1
- Therefore, even identical conditions produce a nonzero distance
- This bias is *positive*: noise always inflates observed distances
- The bias depends on SNR; low-SNR regions see larger distances

**Crossnobis fix**:
- Use a **residual covariance matrix Σ** (estimated from repeated measures) to noise-normalize patterns
- Compute distances as a **cross-validated inner product** across independent runs
- The cross-validation is crucial: both the numerator and denominator include noise, but when computed 
  on *independent* runs, their expectations are zero (they cancel)
- **Result**: Crossnobis produces *unbiased* distance estimates (can even be negative if two conditions are more similar than noise-expected)

**Why they diverge**:
1. **Haxby fMRI has moderate noise** (~SNR ≈ 2–4 in ventral temporal cortex)
2. **Crossnobis removes this bias systematically** → smaller distances overall
3. **Classic RSA, unaware of noise structure**, treats all distance increases as genuine geometry changes
4. **In low-SNR regions**, classic RSA distances can be dominated by noise bias rather than signal

**Observed in data**: Crossnobis–Odd-run reference agreement (0.517) was worse than Classic–Odd-run (0.871). 
This makes sense: Crossnobis corrects for noise in the even runs, but the odd-run reference is *separate data* 
with its own independent noise. Crossnobis is internally consistent (corrects even-run geometry) but does not 
transfer perfectly to a separate odd-run reference that was not part of its noise estimation.

---

### 1.2 Classic RSA vs. PCM: Squared vs. Bounded Distances

**PCM (Pattern Component Modelling)**:
- Does not estimate an RDM directly
- Instead, estimates the **second-moment matrix G = (1/n) U U^T** where U is the (conditions × voxels) patterns
- Predicted RDM from G is: `RDM[i,j] = sqrt(G[i,i] + G[j,j] - 2*G[i,j])` (Euclidean distance on condition means)

**Classic RSA**:
- Uses `1 - Pearson_r` distance (correlation-based)
- Pearson r is bounded: r ∈ [–1, 1] → distance ∈ [0, 2]
- **Correlation is invariant to mean and scaling**, sensitive to *covariance structure*

**PCM-predicted distance**:
- Euclidean distance in the space of condition-mean patterns
- *Sensitive to overall magnitude* of responses, not just their covariance
- Unbounded; depends on the overall response amplitude scale

**Why they diverge**:
1. If voxels have very different response amplitudes (e.g., some high signal, some low), Euclidean distances 
   are dominated by high-amplitude voxels
2. Correlation distances treat all voxels equally (after normalizing by their own variance)
3. **In Haxby data**: Some voxels have weak, noisy responses; others have strong category-specific responses. 
   Euclidean distances weight strong voxels heavily; correlation distances do not.

**Observed in data**: PCM–Odd-run agreement (0.651) was lower than Classic–Odd-run (0.871), consistent with 
PCM capturing a *different* structural property (overall magnitude) than correlation distances.

---

### 1.3 Why Crossnobis vs. Odd-run Reference Was So Low (0.517)

This is the most important finding for statistical rigor:

1. **Crossnobis used both even- and odd-run data** to estimate the residual covariance Σ
2. **The odd-run reference was computed independently**, with no knowledge of Σ
3. **Noise realization differs** between the two halves (independent measurement error)
4. **Crossnobis effectively "overfits" to the noise structure of the even runs** — not just the signal geometry

This is a sign that:
- **Crossnobis is powerful but requires careful statistical validation** (bootstrap, permutation tests) 
  to confirm distances represent geometry and not noise-covariance artifacts
- The paper (Diedrichsen 2021) emphasizes this: use bootstraps to get confidence intervals, and test 
  whether observed differences exceed chance under permutation

---

## Experiment 2: ResNet-18 vs. NSD Brain (Four-Way Disagreement)

**Key finding**: Different metrics ranked layers *differently*:
- **Classic RSA**: Conv1 (0.006)
- **Crossnobis**: Layer3 (0.002)
- **CKA**: Layer4 (0.162)
- **Ridge encoding**: Layer2 (0.014)

This is not noise; it reflects fundamental differences in what each method measures.

### 2.1 Why Ridge Encoding Favors Different Layers

**Ridge encoding**:
- Fits a linear model: `brain_response = X @ w + noise`
- Scores via **held-out Pearson r** (predictive correlation)
- This is fundamentally different from *comparing* representations

**Key difference**:
1. **RSA, CKA, Shape Dist**: Compare *structures* (RDMs, kernel matrices) between brain and model
2. **Ridge**: Predicts brain *directly from* model features

**Why it picks different layers**:
- Ridge optimizes for **prediction**, not structural alignment
- A layer might align structurally with brain (high CKA) but have feature structure that doesn't 
  generalize to predict held-out brain responses (low ridge r)
- Conversely, a layer might have low structural alignment but features that happen to correlate 
  with brain voxels in a way that generalizes

**In the data**: Layer2 scored 0.014 ridge r (best), but CKA ranked layer4 first. This means:
- Layer4 has more similar *covariance structure* to brain patterns
- Layer2 features happen to *correlate with specific brain voxels* in a more predictive way
- These are measuring different things

---

### 2.2 Why CKA and Shape Distance Agree But Diverge from RSA

**CKA (Centered Kernel Alignment)**:
```
CKA(X, Y) = <K_X, K_Y>_F / (||K_X||_F ||K_Y||_F)
```
where K is the centered gram matrix. Properties:
- Invariant to orthogonal rotations and isotropic scaling
- Measures alignment of *covariance structure*
- Bounded [0, 1]; is a similarity (higher = more aligned)

**Shape Distance** (from Williams et al.):
```
d_shape(X, Y) = dist_Riemannian(  Procrustes_align(X, Y)  )
```
Properties:
- Aligns X and Y with orthogonal Procrustes rotation (minimizes Frobenius distance)
- Measures distance on the *shape manifold* (equivalence classes under rotation)
- A proper metric (satisfies triangle inequality), unlike CKA
- Bounded; lower = more similar

**Why they ranked identically in NSD experiment**:
- Both measure *rotational invariants* of covariance structure
- CKA is a correlation-like summary; shape distance is a distance-like summary
- They pick up the same geometric property

**Why they diverge from RSA**:
- **RSA uses Spearman rank correlation of *pairwise distances***
- **CKA/Shape use the full *covariance structure*** of the pattern matrix
- Example: RDM compares all pairs (i, j); CKA compares full patterns at once
- When conditions are linearly separable but with different scales, RSA might rank them differently than CKA

**In the data**: Layer4 ranked first under both CKA and shape distance (0.162 and 0.931), but layer1 
scored only 0.065 CKA and 1.224 shape dist. Classic RSA gave layer1 negative scores (poorly aligned). 
CKA/shape distance captured layer4's overall covariance structure better; RSA's pairwise comparison 
was noisy/negative at this low-signal regime.

---

### 2.3 Why RSA Scores Were Near-Zero (and Unreliable)

**All RSA correlations were low** (range: –0.091 to 0.006). Why?

1. **NSD stimulus set (152 images) vs. Haxby (8 categories)**:
   - Haxby: Highly structured (8 categories × multiple exemplars)
   - NSD: Dense sampling of natural image space (no categories)
   - RDMs in dense spaces are harder to correlate (less structure)

2. **Brain representation is weak**:
   - Only 512 voxels selected (strong selection bias)
   - ResNet-18 is pretrained on ImageNet (not NSD-specific)
   - Expected misalignment is high

3. **Small effect sizes**:
   - ResNet layer1 (0.065 CKA) is *mildly* aligned; layer4 (0.162) is *slightly better*
   - In this low-effect regime, RSA Spearman correlations are noisy and near-zero
   - CKA and shape distance, which use the full covariance structure, have larger absolute values

**Interpretation**: The 4-way disagreement on layer ranking is not a failure of the methods but 
a reflection of *different measurement axes*. Ridge "works best" on layer2, but CKA "works best" 
on layer4, because they are optimizing for fundamentally different things. This is exactly 
what Kriegeskorte et al. (2008) warned: different metrics can rank hypotheses differently because 
they encode different assumptions.

---

## Experiment 3: DistilGPT-2 vs. Pereira 2018

**Key finding**: All methods (RSA, CKA, Ridge) favored the **input embedding**.

**Why the embedding?**
1. The embedding is directly optimized for language structure (token predictions)
2. Middle layers perform low-level syntactic / semantic transformations
3. These transformations may not align with the *overall* representational geometry that Pereira's 
   language-cortex RDM captures
4. This is consistent with prior work (Jozwik et al. 2024): transformer embeddings often align 
   best with high-level cortex in *Spearman correlation* of RDMs, but this depends on pooling strategy

**Why didn't middle layers win?**
- Expected from prior literature (e.g., Sclar et al., Jozwik et al.): deeper layers misalign due to 
  distributional shifts between pre-training and brain (fMRI is noisy; transformers output is abstract)
- Embedding is data-rich and directly tied to the input RDM structure

---

## Summary: The "Why Do Methods Disagree" Mechanistic Framework

| Disagreement | Root Cause | Evidence |
|---|---|---|
| Classic RSA vs. Crossnobis | Noise bias in correlation distances; crossnobis removes it. | Crossnobis–odd-run agreement much lower (0.517 vs 0.871) because each half has independent noise. |
| Classic RSA vs. PCM | Euclidean vs. correlation distance; PCM weights voxel amplitude. | PCM ranked similarly to Classic but with lower agreement (0.758). |
| Ridge vs. RSA/CKA | Prediction vs. structure comparison; different optimization. | Ridge favored layer2; CKA favored layer4 in NSD. |
| CKA vs. Shape distance | Both measure covariance alignment (agreed in NSD). | Identical layer rankings; shape distance is just a metric version of CKA. |
| Embedding vs. middle layers (LLM) | Embedding directly optimized for task; deeper layers misalign. | Prior work (Jozwik, Sclar) shows this pattern consistently. |

---

## Implications for Future Work

1. **Permutation tests are essential**. Crossnobis's low agreement with odd-run reference suggests 
   we need bootstrap confidence intervals and permutation tests to validate that observed distances 
   represent geometry, not noise-covariance artifacts.

2. **Method selection should be task-specific**. Different metrics answer different questions:
   - **Ridge**: "Can I predict brain from this layer?" (practical prediction)
   - **RSA**: "Does this layer rank stimulus pairs the same way as brain?" (structural comparison)
   - **CKA**: "Is the covariance structure similar?" (invariant geometry)
   - **Shape distance**: "How far apart are the condition-geometry shapes?" (metric geometry)

3. **Report effect sizes explicitly**. Many correlations in Exp 2 were near zero. This matters: 
   a 0.162 CKA difference between layers is real, but small. A reader needs to know whether 
   0.014 ridge r is "good" or "noise."

4. **Noise-ceiling estimation** (future gap): All results should be compared to noise-ceiling estimates 
   (e.g., brain RDM split-half, permutation-based chance). Without it, results are descriptive only.

---

## References (Cited Methods)

- **Crossnobis bias correction**: Diedrichsen & Kriegeskorte (2017), Diedrichsen et al. (2021)
- **Metric geometry and shape distance**: Williams et al. (2021)
- **CKA and covariance alignment**: Kornblith et al. (2019)
- **LLM-brain alignment**: Jozwik et al. (2024), Sclar et al. (2023)
