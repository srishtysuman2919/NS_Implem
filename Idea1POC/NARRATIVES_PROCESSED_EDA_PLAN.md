# Narratives (`ds002345`) Processed-Data EDA & MD Feasibility Gate Plan

This document specifies the **9-step Exploratory Data Analysis (EDA) and Feasibility Gate** for using the preprocessed **Narratives** fMRI dataset (Nastase et al., 2021; `ds002345`) to test selective **Multiple-Demand (MD)** vs. **Language (LANG)** network alignment and misalignment (extending [Idea1POC/README.md](README.md) and [Idea1POC/03_gate_analysis.py](03_gate_analysis.py)).

---

## 0. Critical Repository & Data-Source Findings

Before running the 9 steps, note the distinction between the two derivative sources:

1. **Local `ds002345-fmriprep/` (`OpenNeuroDerivatives/ds002345-fmriprep`):**
   - Covers `sub-001`..`sub-200` and was generated with fMRIPrep `--level minimal`.
   - `sub-*/func/` contains **only** 3D reference images (`_boldref.nii.gz`) and registration transforms (`_xfm.txt` / `_xfm.json`).
   - It does **not** contain the 4D preprocessed BOLD time series (`_desc-preproc_bold.nii.gz`), surface time series (`.func.gii`), or confound tables (`_desc-confounds_regressors.tsv`).

2. **Official Narratives Derivatives (`labs/hasson/narratives` & `snastase/narratives`):**
   - Covers all **345 subjects** across **27 stories** (891 functional scans).
   - Hosted via DataLad (`http://datasets.datalad.org/?dir=/labs/hasson/narratives/derivatives`) with companion scripts and metadata in `https://github.com/snastase/narratives`:
     - `derivatives/fmriprep/`: 4D unsmoothed preprocessed BOLD in `MNI152NLin2009cAsym` (`res-native`) and `fsaverage6` (`hemi-L/R.func.gii`), plus `desc-confounds_regressors.tsv`.
     - `derivatives/afni-nosmooth/`: Pre-denoised (`desc-clean`) unsmoothed BOLD in MNI and `fsaverage6` (regressed via AFNI `3dTproject` with 6 motion parameters + derivatives, aCompCor CSF/WM, cosine drift, and quadratic polynomial trend).
     - `derivatives/afni-smooth/`: Pre-denoised (`desc-clean`) 6 mm FWHM smoothed BOLD (`desc-sm6` and `desc-clean`).
     - `derivatives/mriqc/`: Participant- and group-level MRIQC metrics (framewise displacement, tSNR, smoothness).
   - Cached metadata in [data/narratives_meta](../data/narratives_meta):
     - [data/narratives_meta/task_meta.json](../data/narratives_meta/task_meta.json): Exact derivative paths per story and subject.
     - [data/narratives_meta/scan_exclude.json](../data/narratives_meta/scan_exclude.json): Official list of excluded/flagged scans per story (low early-auditory-cortex ISC, excessive motion, or noncompliance).
     - [data/narratives_meta/event_meta.json](../data/narratives_meta/event_meta.json): Exact stimulus audio `onset` and `duration` (in TRs) per story.

---

## Execution Priority

* **Phase 1 (Immediate Gating Checks — ~1 hour):** Run **Step 1 (Inventory & Usability)** and **Step 5 (Signal Reliability / ISC Noise Ceiling in EAC, LANG, and MD)**, supported by **Step 2 (Timing)** and **Step 4 (Network Masks)**.
  * *Why:* These steps directly answer whether passive story listening in Narratives drives enough reliable MD-network signal to run the selective misalignment experiment.
* **Phase 2 (If Gate Passes):** Run **Step 3 (QC & Denoising Verification)**, **Step 6 (Simple Acoustic/Lexical Baselines)**, **Step 7 (LANG–MD Relationship & Residual Autocorrelation)**, **Step 8 (Feature-Side Diagnostics)**, and **Step 9 (Splits, Cohorts & Power)**.

---

## Simple Question Matrix: What Each of the 9 EDA Steps Answers

| Step | Stage Name | Simple Plain-English Questions Answered | Concrete Metric / Output Artifact | Current Status |
|---|---|---|---|---|
| **Step 1** | **Inventory & Usability** | • How many participants, stories, and runs do we have?<br/>• How many stories did each person listen to?<br/>• Which scans are broken/excluded, and which stories have $\ge 40$ clean subjects?<br/>• How do we handle subjects scanned twice on the same story (`run-1`, `run-2`)? | Story-by-story usability table ($N_{\text{raw}}$, $N_{\text{excluded}}$, $N_{\text{usable}}$, multi-run count, TR, duration) + multi-run policy | **Mostly Answered** (metadata parsed; final clean table ready to render) |
| **Step 2** | **Stimulus & Timing Alignment** | • What is the TR (sampling interval)?<br/>• Is scan length fixed for everyone who hears a story, and where does the actual story start/end inside the scan?<br/>• How many words occur per TR, how many TRs are silent, and what hemodynamic delay (in TRs) matches auditory cortex? | Table of `(total_TRs, onset_TR, duration_TRs, words_per_TR, silent_TR_%)` + EAC lag cross-correlation curve (lags $-2..+8$ TRs) | **Partially Answered** (`event_meta.json` onsets/durations known; word-rate & lag check pending) |
| **Step 3** | **Data Quality, Space & Variance Hygiene** | • Which subjects/runs have too much head motion or low SNR?<br/>• Should we use surface (`fsaverage6`) or volume (`MNI`), smoothed or unsmoothed?<br/>• Are some voxels "always lit" (high static baseline or near-zero temporal variance) that could cause fake/overfitted predictions? | Motion (FD) & tSNR plots; constant/low-variance voxel audit; verification of per-voxel within-run z-scoring ($\mu=0, \sigma=1$) + high-pass drift removal | **Not Answered Yet** (requires loading processed BOLD + confounds) |
| **Step 4** | **Network Masks (EAC, LANG, MD)** | • What brain regions are recorded?<br/>• Exactly which voxels/vertices belong to EAC, LANG, and MD?<br/>• Do LANG and MD masks overlap spatially, and how many shared voxels must be removed? | Parcel-by-parcel voxel/vertex count table for EAC, LANG, and MD; overlap count $|\text{LANG} \cap \text{MD}|$ removed; brain/tSNR mask intersection | **Not Answered Yet** (requires projecting LANG/MD atlases to `fsaverage6` / `MNI`) |
| **Step 5** | **Signal Reliability (Key Feasibility Gate)** | • Does passive story listening actually drive a reliable signal in the MD network across subjects?<br/>• How many MD voxels/parcels pass the noise-ceiling threshold ($\text{ISC} > 0.05$)?<br/>• Does across-subject ISC underestimate MD reliability compared to within-subject `run-1` vs. `run-2` correlation? | Histograms of per-voxel/parcel ISC & group noise ceiling ($\text{NC}_{\text{group}}$) in EAC vs. LANG vs. MD; count of MD channels $> 0.05$; within-subject vs. across-subject MD reliability on `pieman` | **Not Answered Yet** (**Top Priority Gate**) |
| **Step 6** | **Simple Acoustic & Lexical Baselines** | • Does audio loudness predict auditory cortex (proving our timing is right)?<br/>• How much of LANG and MD activity is already explained by simple features (word rate, word length, frequency, surprisal) before using GPT-2? | Leave-one-story-out ridge $r$ for Acoustic (envelope) and Lexical (rate/length/freq/surprisal) baselines in EAC, LANG, and MD | **Not Answered Yet** |
| **Step 7** | **LANG–MD Correlation & Temporal Autocorrelation** | • How strongly correlated are LANG and MD outputs (at the network-mean, parcel-pair, and voxel-pair levels)?<br/>• How much correlation/noise exists between consecutive TRs ($t$ and $t+1$), and what permutation block length is needed so consecutive-TR autocorrelation doesn't give false positives? | Network-mean correlation $\text{corr}(\bar{y}_{\text{LANG}}, \bar{y}_{\text{MD}})$, full parcel-by-parcel cross-correlation matrix $(C_{\text{LANG} \times \text{MD}})$, residual MD autocorrelation function (ACF) up to 30 TRs, and minimum circular-shift block size | **Not Answered Yet** |
| **Step 8** | **Feature-Side Diagnostics (GPT-2)** | • How many tokens fall into each multi-TR context window?<br/>• Do any stories share identical/scrambled text that would leak across train/test splits?<br/>• What does the PCA variance spectrum of GPT-2 look like across layers 1–12? | Token-per-window distribution; story-pair n-gram overlap matrix (grouping `pieman*` and `notthefall*`); layerwise PCA effective dimensionality curve | **Partially Answered** (duplicate story families identified; token/PCA checks pending) |
| **Step 9** | **Splits, Cohorts & Statistical Power** | • After accounting for consecutive-TR autocorrelation, how many *effective* independent samples ($T_{\text{eff}}$) do we have per story?<br/>• Since different stories have different subject subsets, how do we pool stories and handle multi-run subjects without cohort leakage? | Table of $(T_s, \rho_1, T_{\text{eff},s})$ per held-out story; subject-by-story cohort overlap heatmap; standardization & run-handling protocol | **Not Answered Yet** |

---

## Audit of the 12 Planning Questions: What Is Answered vs. What Remains

| # | Question | Status | What We Already Know (from Raw/Metadata EDA) | What Still Needs Processed-Data EDA (and Where It Lives in Plan) |
|---|---|---|---|---|
| **1** | **Number of participants** | **Answered** | **345 unique participants** (`sub-001`..`sub-345`) in the full dataset (`participants.tsv` and `task_meta.json`). Note: the local minimal clone `ds002345-fmriprep` only has folders for `sub-001`..`sub-100`/`200`, whereas the official `labs/hasson/narratives` derivatives cover all 345. | Tabulate exact **usable** participants per story after applying `scan_exclude.json` (**Step 1**). |
| **2** | **Number of stories** | **Answered** | **19 BIDS `task` names** in `participants.tsv` / `task_meta.json`, which correspond to **27 unique spoken story stimuli** (~4.6 hours total). Why 19 vs. 27? Because `slumlordreach` has 2 stories in one scan (`slumlord` + `reach`), and `schema` has 8 short stories (`run-1`..`run-8`). Also note condition variants (`notthefallintact`/`longscram`/`shortscram`, `pieman`/`piemanpni`). | Finalize the subset of stories with $N_{\text{usable}} \ge 40$ and intact narrative structure (**Step 1**). |
| **3** | **Number of stories each participant got scanned on** | **Answered** | Across the 345 participants, there are **769 subject–task entries** (891 total functional runs when counting multi-run tasks):<br/>• **1 story:** 107 subjects<br/>• **2 stories:** 134 subjects<br/>• **3 stories:** 40 subjects<br/>• **4 stories:** 51 subjects<br/>• **5 stories:** 9 subjects<br/>• **6 stories:** 3 subjects<br/>• **7 stories:** 1 subject | Plot the subject-by-story cohort overlap matrix to document which stories share participants (**Step 9**). |
| **4** | **TR (interval at which scan was recorded)** | **Mostly Answered** | **$\text{TR} = 1.5\text{ s}$** is the standard sampling interval across Narratives (verified on `pieman` where $\text{TR}=1.5\text{ s}$ and in `run_regression.py`). | Programmatically verify TR across all 19 tasks' BIDS `.json` sidecars to confirm if any task/subject deviates from $1.5\text{ s}$ (**Step 1**). |
| **5** | **What is the input length for each scan? Is it based on TR / how much story they reach, and is it fixed for everyone?** | **Answered (Conceptually + `event_meta.json`)** | **It is strictly fixed per story for all participants**—the audio is pre-recorded, so every subject hears the exact same words at the exact same second/TR (they do *not* read at their own pace).<br/>Each story has a fixed number of total scan TRs and a fixed active story window `[onset, onset + duration)` in `event_meta.json` (e.g., `pieman` is 300 total TRs = $450\text{ s}$, with story audio starting at `onset = 10` TRs and lasting `duration = 281` TRs; `tunnel` has `duration = 1023` TRs; `21styear` has `duration = 2225` TRs). | Verify that every usable subject's 4D file for story $s$ has the exact expected TR count after slicing `[onset : onset + duration]`, and compute words per TR (**Step 2**). |
| **6** | **What all regions were recorded?** | **Partially Answered** | Acquisition is **whole-brain** (e.g., $64 \times 64 \times 27$ grid covering cortex and subcortex in `pieman`, or $40,962$ cortical vertices per hemisphere in `fsaverage6`). However, raw whole-brain grids are not labeled by functional network. | Project the **EAC** (Glasser MMP), **LANG** (Fedorenko 12-parcel language atlas), and **MD** (Fedorenko 20-parcel executive atlas) masks onto the data and verify coverage (**Step 4**). |
| **7** | **What all voxels in each region were recorded?** | **Not Answered Yet** | In Cell 12 of `eda_narratives.ipynb`, only a rough intensity threshold ($\sim 25\text{k}$ whole-brain voxels) was computed on one raw scan—not separated by brain region or network. | Count exact voxels (in `MNI`) or surface vertices (in `fsaverage6`) inside each individual EAC, LANG, and MD parcel after intersecting with the group brain/tSNR mask (**Step 4**). |
| **8** | **Is there overlap in voxel recording (between LANG and MD)?** | **Not Answered Yet** | Group-level probabilistic LANG and MD atlases can abut or overlap at anatomical boundaries (especially in left MFG / IFG / inferior precentral sulcus), and spatial smoothing (`sm6`) further blurs signal across adjacent voxels. | Compute exact voxel/vertex intersection $|\text{LANG} \cap \text{MD}|$, remove shared voxels from both masks, and quantify border proximity (**Step 4**). |
| **9** | **Will there be noise / bias due to correlation in consecutive TRs?** | **Answered Conceptually; Empirical Check Pending** | **Yes—strongly.** Because the hemodynamic response (HRF) is a slow ~15–20 s low-pass filter and narrative speech is continuous, consecutive TRs ($t, t+1, t+2$, spaced $1.5\text{ s}$ apart) are heavily autocorrelated ($\rho_1 \approx 0.6\text{--}0.85$).<br/>**Why it matters:** (1) Random TR train/test splits will severely overfit by leaking adjacent TRs (must split **by story**); (2) standard $p$-values assume independent time points and will be wildly over-optimistic (must use **within-story circular-shift / block permutations** and report effective sample size $T_{\text{eff}}$). | Measure the empirical autocorrelation function (ACF) up to 30 TRs for LANG, MD, and residual MD; compute $T_{\text{eff}}$ and set permutation block length (**Step 7 & Step 9**). |
| **10** | **How to check variance in voxel output for MD & LANG (e.g., if a few voxels are "always lit" for all inputs, making prediction trivially easy / spuriously good)?** | **Answered Methodologically; Empirical Check Pending** | **Why "always lit" happens & how we prevent/check it:**<br/>1. **Static baseline ("always high mean"):** Raw BOLD has huge static tissue-intensity differences across voxels (seen in your Cell 12 mean slice plot). If you evaluate across voxels without centering, a model predicting the static mean gets a fake high score. **Fix:** Z-score every voxel/parcel *across time within each story run* ($\mu_v = 0, \sigma_v = 1$) and evaluate correlation **across time per voxel/fROI** (`pearson_cols`), where a constant prediction gets $r = 0$.<br/>2. **Slow scanner drift / low variance:** Dead/dropout voxels or unremoved drift can create near-zero or step-function variance. **Check:** Plot temporal SD ($\sigma_v$), tSNR ($\mu_v/\sigma_v$), and low-frequency drift power before z-scoring, and drop voxels with $\sigma_v \approx 0$ or low tSNR. | Add explicit **Voxel Variance & Constant-Baseline Diagnostic** in **Step 3** and enforce per-run temporal z-scoring + zero-variance filtering. |
| **11** | **How to check for correlation in voxel output between MD and LANG?** | **Answered Methodologically; Empirical Check Pending** | In `Idea1POC/03_gate_analysis.py`, this is checked at multiple levels:<br/>1. **Network-mean time-course correlation:** $\text{corr}(\bar{y}_{\text{LANG}}(t), \bar{y}_{\text{MD}}(t))$ per story.<br/>2. **Parcel-by-parcel / voxel-by-voxel cross-correlation matrix:** $C \in \mathbb{R}^{V_{\text{LANG}} \times V_{\text{MD}}}$ where $C_{ij} = \text{corr}(y_{\text{LANG},i}(t), y_{\text{MD},j}(t))$.<br/>3. **Multi-lag ridge variance partitioning ($R^2_{\text{LANG}\to\text{MD}}$):** Fit a cross-validated ridge model predicting each MD voxel/parcel from all LANG channels across lags $(0, 1, 2)$ TRs (`residualize` in `03_gate_analysis.py`), which captures *all* linear shared variance between the two networks. | Compute and plot (a) $\text{corr}(\bar{y}_{\text{LANG}}, \bar{y}_{\text{MD}})$, (b) the $V_{\text{LANG}} \times V_{\text{MD}}$ cross-correlation heatmap, and (c) $R^2_{\text{LANG}\to\text{MD}}$ per MD parcel (**Step 7**). |
| **12** | **How to take care of samples where a participant has multiple runs for the same story?** | **Answered (Exact Counts Found + Policy Defined)** | From `task_meta.json`, multi-run subjects occur in two distinct cases:<br/>• **Case A — True repeat listens of the same story (`pieman`):** **14 subjects** have both `run-1` and `run-2` (though in `scan_exclude.json`, `sub-001` has both runs excluded and `sub-013`/`sub-014` have `run-2` excluded).<br/>• **Case B — Separate mini-stories (`schema`):** **31 subjects** have `run-1`..`run-8`, where each run is a *different* 3-minute film/story excerpt.<br/>**Policy for Case A (`pieman` repeats):**<br/>1. **Never put `run-1` and `run-2` of the same story in different CV folds** (all CV splits are by story).<br/>2. **For primary group-mean / ISC analysis:** Use **only `run-1`** (first exposure) so novelty/comprehension matches single-run subjects (or average `run-1` and `run-2` *within* the subject before across-subject averaging so a repeat subject isn't double-weighted).<br/>3. **Bonus for Step 5:** Use `run-1` vs. `run-2` on the clean repeat subjects to measure **within-subject test–retest MD reliability**! | Implement explicit multi-run handling rules in **Step 1, Step 5, and Step 9**. |

---

## Detailed 9-Step Processed-Data EDA Plan (Merged Master Specification)

### Step 1. Inventory and Usability (Cheap Gating Check #1)
- **Tabulate per story ( answers Q1, Q2, Q3, Q4, Q12 ):**
  - Total subjects in `participants.tsv` (345 total; 1–7 stories per subject, totaling 769 subject–task pairs) vs. subjects indexed in `task_meta.json`.
  - Number of runs per subject (`pieman`: 14 subjects with 2 runs; `schema`: 31 subjects with 8 separate mini-story runs).
  - Excluded scans from `scan_exclude.json` and resulting **usable subject count ($N_{\text{usable}}$)**.
  - Total scan TRs vs. story-active TRs (`duration` in `event_meta.json`).
- **Verify TR uniformity ( answers Q4 ):**
  - Confirm $\text{TR} = 1.5\text{ s}$ across stories and flag any exceptions (e.g., `schema` or mismatched acquisitions noted in `exclude_scans.py`).
- **Multi-run handling policy ( answers Q12 ):**
  - Distinguish **same-stimulus repeat runs** (`pieman` `run-1` vs. `run-2`) from **multi-part stimulus runs** (`schema` `run-1`..`run-8`).
  - For across-subject group means and leave-one-out ISC, either restrict to `run-1` (first-listen novelty match) or average clean runs *within* subject first so no participant receives $2\times$ weight.
- **Flag stimulus families that share underlying text/audio (to prevent train/test leakage):**
  - `pieman` and `piemanpni`
  - `notthefallintact`, `notthefalllongscram`, and `notthefallshortscram`
  - `prettymouth` (between-subject conditions: `affair` vs. `paranoia` share identical audio)
  - `milkyway` (conditions: `original`, `vodka`, `synonyms`)
- **Story selection rule:**
  - Prioritize stories with **$N_{\text{usable}} \ge 40$** (e.g., `pieman`, `tunnel`, `lucy`, `prettymouth`, `milkyway`, `slumlordreach`, `21styear`, `notthefallintact`) for stable across-subject MD estimation.

### Step 2. Stimulus and Timing Alignment (Cheap Gating Check #2)
- **Align transcripts, audio, and fMRI TR windows ( answers Q5 ):**
  - Every story scan is fixed-paced (pre-recorded audio) and includes lead-in silence and post-story buffer TRs specified in `event_meta.json`:
    - Example (`pieman`): `onset = 10` TRs ($15\text{ s}$), `duration = 281` TRs out of 300 total TRs.
    - Example (`slumlordreach`): two stories inside a single scan (`slumlord`: onset 20, duration 602 TRs; `reach`: onset 646, duration 550 TRs).
  - Verify that Gentle forced-aligned word timestamps (`stimuli/gentle/<story>/align.csv`) and audio waveforms align with the sliced fMRI window $[\text{onset}, \text{onset} + \text{duration})$, and that all usable subjects for story $s$ have identical sliced TR length $T_s$.
- **Stimulus statistics per story:**
  - Compute words per TR ($1.5\text{ s}$ bin), fraction of silent/empty TRs, mean speaking rate, and total story length in TRs.
- **Hemodynamic response delay justification:**
  - In Natural Stories ($\text{TR} = 2.0\text{ s}$), a 2-TR shift equals $4.0\text{ s}$.
  - In Narratives ($\text{TR} = 1.5\text{ s}$), a $4.5\text{--}6.0\text{ s}$ BOLD peak corresponds to **3–4 TRs** (or lags $1..4$ TRs = $1.5..6.0\text{ s}$).
  - Validate the chosen lag empirically by cross-correlating the audio RMS envelope / word rate against **Early Auditory Cortex (EAC)** BOLD across lags $-2..+8$ TRs.

### Step 3. Data Quality, Preprocessing Choices, and Voxel Variance Hygiene (Cheap Gating Check #3)
- **Motion and tSNR distributions:**
  - From `desc-confounds_regressors.tsv` and `mriqc`, plot per-subject and per-story **Framewise Displacement (FD)** (mean FD, max FD, proportion of TRs $> 0.5\text{ mm}$) and **temporal SNR (tSNR)**. Define explicit scan exclusion thresholds on top of `scan_exclude.json`.
- **Voxel variance & "always-lit" baseline diagnostic ( answers Q10 ):**
  - Inspect the across-time mean $\mu_v$ and across-time standard deviation $\sigma_v$ of every voxel/vertex in LANG and MD before normalization.
  - Identify and drop dead/flatline voxels ($\sigma_v < 10^{-6}$) or severe susceptibility-dropout voxels (bottom tSNR percentile).
  - Verify that **within-run temporal z-scoring** ($y_{v}^{\text{z}}(t) = (y_v(t) - \mu_v)/\sigma_v$) and **detrending / high-pass filtering** (`polort 2` + cosine drift in `afni-nosmooth`/`afni-smooth`) completely remove static baseline offsets so a model cannot achieve high correlation by predicting a constant "always-lit" voxel baseline.
  - Ensure all encoding metrics (`pearson_cols`, `r2_cols` in [Idea1POC/03_gate_analysis.py](03_gate_analysis.py)) are computed **along the time axis per voxel/parcel**, never across uncentered voxels.
- **Space & smoothing decision (keep fixed once chosen):**
  - **Surface (`fsaverage6`, 40,962 vertices/hemisphere) vs. Volume (`MNI152NLin2009cAsym_res-native`):**
    - Note that `MNI152NLin2009cAsym_res-native` keeps each story's native voxel resolution, so volumetric grid dimensions differ across stories unless resampled or parcel-averaged first. `fsaverage6` uses an identical vertex mesh across all 345 subjects and 27 stories.
  - **Unsmoothed (`afni-nosmooth`) vs. 6 mm Smoothed (`afni-smooth`):**
    - Compare parcel-level and vertex/voxel-level reliability before locking the choice.
- **Document exact BOLD denoising pipeline:**
  - Record confound regression (6 realignment parameters + derivatives, aCompCor CSF/WM components, cosine/polynomial drift), temporal filtering, story-window slicing, and per-run z-scoring.

### Step 4. Network Masks (EAC, LANG, and MD)
- **Atlas selection & projection ( answers Q6, Q7 ):**
  - Because Narratives lacks individual functional localizers per participant, use group-level probabilistic functional atlases projected to the chosen space (`MNI152NLin2009cAsym` or `fsaverage6`):
    - **EAC (Positive Control):** Glasser MMP1.0 Early Auditory Cortex (`A1`, `LBelt`, `MBelt`, `PBelt`, `RI`), matching `roi_masks.py` in `snastase/narratives`.
    - **LANG Network:** Fedorenko Lab Language Atlas (6 LH / 12 bilateral parcels: IFGorb, IFG, MFG, AntTemp, PostTemp, AngG).
    - **MD Network:** Fedorenko Lab Multiple-Demand Atlas (20 bilateral frontoparietal parcels: precentral, IFS, MFG, insula, IPS, ACC/pre-SMA).
- **Mask hygiene & spatial overlap removal ( answers Q7, Q8 ):**
  - Report exact voxel/vertex counts per parcel and per network (`n_vox_EAC`, `n_vox_LANG`, `n_vox_MD`).
  - Measure spatial overlap $|\text{LANG} \cap \text{MD}|$ (particularly in left MFG/IFG) and **explicitly remove shared voxels/vertices** (plus an optional boundary buffer if using unsmoothed/smoothed volumes) so shared spatial footprint cannot masquerade as shared functional signal.
  - Intersect with the group coverage mask and drop low-tSNR locations.

### Step 5. Signal Reliability — The Key Feasibility Check (Do With Step 1)
- **Across-subject Leave-One-Out ISC:**
  - For each story $s$ ($N_s$ usable subjects) and each voxel/vertex or fROI parcel $v$, compute leave-one-out ISC:
    $$\text{ISC}_i(v) = \text{corr}\!\left(y_{i,v},\; \frac{1}{N_s - 1}\sum_{j \ne i} y_{j,v}\right), \qquad \text{ISC}(v) = \frac{1}{N_s}\sum_{i=1}^{N_s} \text{ISC}_i(v)$$
  - Compute the **group-mean noise ceiling** (reliability of the subject-averaged time course $\bar{y}_v$):
    $$\text{NC}_{\text{group}}(v) = \frac{N_s \cdot \text{ISC}(v)}{1 + (N_s - 1)\cdot \text{ISC}(v)}$$
- **Core feasibility metrics to report per story:**
  1. Distribution of single-subject $\text{ISC}(v)$ and group-mean $\text{NC}_{\text{group}}(v)$ across **EAC**, **LANG**, and **MD**.
  2. **Count and percentage of MD voxels/vertices/parcels with $\text{ISC} > 0.05$** (and $\text{NC}_{\text{group}} > 0.10$) per story.
  3. **Sanity check:** Confirm that EAC and LANG show strong positive ISC on intact stories.
  4. **Within-subject vs. across-subject MD reliability ( answers Q12 bonus ):** For the clean repeat-listen subjects in `pieman` (`run-1` vs. `run-2`), compute within-subject correlation $\text{corr}(y_{i,\text{run1}}, y_{i,\text{run2}})$ in MD and compare against across-subject ISC to quantify how much anatomical/functional idiosyncrasy depresses group-level MD ISC.

### Step 6. Simple Acoustic and Lexical Baselines
- Fit leave-one-story-out ridge regressions predicting EAC, LANG, and MD from low-level features:
  - **Acoustic baseline:** RMS audio envelope.
    - *Gate check:* EAC must be reliably predicted by the audio envelope; failure indicates a timing/onset bug.
  - **Lexical/cognitive-load baseline:** word rate (words/TR), mean word length, log word frequency, and token surprisal.
    - *Benchmark bar:* Quantify how much MD variance is explained by simple word rate + surprisal alone; GPT-2 representations in [Idea1POC/03_gate_analysis.py](03_gate_analysis.py) must beat this baseline on residual MD prediction.

### Step 7. Empirical LANG–MD Relationship & Temporal Autocorrelation
- **Multi-level LANG–MD output correlation ( answers Q11 ):**
  1. **Network-mean correlation:** Compute $\text{corr}(\bar{y}_{\text{LANG}}(t), \bar{y}_{\text{MD}}(t))$ per story and across stories, and compare with the $-0.33$ correlation observed in Shain et al. (2020) (noting sensitivity to CompCor / global signal regression).
  2. **Parcel-by-parcel / voxel-by-voxel cross-correlation matrix:** Compute the full pairwise correlation matrix $C \in \mathbb{R}^{V_{\text{LANG}} \times V_{\text{MD}}}$ where $C_{ij} = \text{corr}(y_{\text{LANG},i}(t), y_{\text{MD},j}(t))$, and inspect its singular values and maximum parcel-pair correlations (e.g., left MFG in LANG vs. left MFG/IFS in MD).
  3. **Lagged cross-network explainable variance ($R^2_{\text{LANG}\to\text{MD}}$):** Predict each MD channel from all LANG channels across lags $(0, 1, 2)$ TRs via leave-one-story-out ridge regression (`residualize` in [Idea1POC/03_gate_analysis.py](03_gate_analysis.py)) to measure total linear variance in MD explainable by LANG before testing GPT-2 on the residual $Y_{\text{MD}} - \hat{Y}_{\text{MD}|\text{LANG}}$.
- **Consecutive-TR autocorrelation & permutation block length ( answers Q9 ):**
  - Estimate the autocorrelation function (ACF) of $\bar{y}_{\text{LANG}}(t)$, $\bar{y}_{\text{MD}}(t)$, and the LANG-residualized time course $Y_{\text{MD}} - \hat{Y}_{\text{MD}|\text{LANG}}$ across lags $1..30$ TRs ($1.5..45\text{ s}$).
  - Find the lag $L_{\text{decorr}}$ where ACF drops below $0.1$ (typically $6\text{--}10$ TRs $\approx 9\text{--}15\text{ s}$) and use it to set the minimum circular shift / block length in the permutation test of [Idea1POC/03_gate_analysis.py](03_gate_analysis.py).

### Step 8. Feature-Side Diagnostics (GPT-2)
- **Window statistics:**
  - Inspect token counts per TR context window (e.g., 5-TR = $7.5\text{ s}$ window) across stories.
- **Train/test text leakage audit:**
  - Compute n-gram overlap across all story transcripts and enforce **story-family grouping** in cross-validation so related stimuli (`pieman`/`piemanpni`, `notthefall*`) never split across train and test folds.
- **Layerwise geometry:**
  - Plot the PCA / singular-value spectrum and effective dimensionality of GPT-2 features across layers 1–12.

### Step 9. Cross-Validation Splits, Subject Cohorts, and Statistical Power
- **Sample size & effective degrees of freedom ( answers Q9 ):**
  - Tabulate story-locked TRs $T_s$ and autocorrelation-adjusted effective sample size:
    $$T_{\text{eff}, s} = T_s \cdot \frac{1 - \rho_1}{1 + \rho_1}$$
    per held-out story and across the selected multi-story benchmark.
- **Subject-modeling strategy, cohort overlap & multi-run protocol ( answers Q3, Q12 ):**
  - Plot the subject-by-story participation matrix showing how the 345 subjects distribute across the 19 tasks.
  - Compare group-mean time courses per story vs. subject-level models.
  - Enforce within-subject, within-story temporal z-scoring prior to subject averaging so cohort differences or multi-run subjects cannot create spurious between-story variance.
