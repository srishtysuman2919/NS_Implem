# Narratives (`ds002345`) Processed-Data EDA — Summary & Draft Gate Specification (`v1.3.0`)

**Implementation update (2026-10-25):** The separate vertex-wise preparation,
feature and primary-gate engine now exists; see [NARRATIVES_STAGE_A.md](NARRATIVES_STAGE_A.md).
The current unfrozen specification is v1.4.0, superseding the v1.3.0 draft
described below. Software tests do not establish a real Narratives result.
Complete bilateral data and certified model/comparator features are unavailable;
no real primary run has occurred. The historical EDA discussion below does not
certify that engine or its outstanding diagnostics/sensitivities.

**Notebook:** [eda_narratives_processed.ipynb](../eda_narratives_processed.ipynb)  
**Plan Reference:** [Idea1POC/NARRATIVES_PROCESSED_EDA_PLAN.md](NARRATIVES_PROCESSED_EDA_PLAN.md)  
**Pre-Specified Gate Config:** [Idea1POC/narratives_gate_config.json](narratives_gate_config.json) (`v1.3.0`, unfrozen draft)  
**Stage A Status:** **Not established by this EDA.** The existing Stage A script operates on the older fROI dataset and does not consume this config. Vertex-wise Narratives encoding, pooled training-fold selection, and cross-half residualization remain to be implemented and validated. The EDA mixes surface-vertex QC/reliability with parcel-level baseline analyses; it is not a completed vertex-wise GPT-2 experiment.

**v1.3.0 Update (2026-10-25):** Gate 1 now requires both LANG and raw MD to have positive prediction correlation above their within-story circular-shift null, BH-adjusted across the two primary Gate 1 tests (`q < 0.05`), and each observed `r >= 0.04`. The floor is a practical planning choice, not an exact FDR critical value. Report Shain screening observations (LANG `0.1340`, raw MD `0.0987`, layer 9, lag 0) alongside future results as reference-only anchors, not Narratives expectations or thresholds. Validated Narratives raw-network expected values have not been calculated. The existing residual-MD transfer range is an assumption, not a measured Narratives effect.

---

## 1. Five Final Pre-Stage-A Corrections & Verifications (Cell 22)

The Step 10 checks informed the historical `v1.2.0` draft below; they did not lock the config or verify the unimplemented Stage A pipeline. Power figures below are approximate planning calculations, not measured permutation/FDR power.

1. **Gate 2 & Gate 3 Decision Rules Aligned With Actual Power:**
   - At $T_{\text{eff}} = 2,270.8$ (Regime C, 16 intact tasks), the standard error of a correlation is $\text{SE}_r \approx \frac{1}{\sqrt{2270.8 - 3}} = \mathbf{0.0210}$, and the two-sided $\alpha = 0.05$ critical value is **$r_{\text{crit}} = 0.0411$**.
   - Previously, requiring $r_{\text{obs}} \ge 0.060$ meant that if the true effect was $r_{\text{true}} = 0.0625$, the observed correlation would clear $0.060$ only **$54.7\%$** of the time!
   - In `v1.2.0`, **Gate 2** gates on **`permutation q < 0.05 AND observed r >= 0.04`** (with explicit verdict bands: `NO_GO` if $q \ge 0.05$ or $r < 0.04$; `SUBTLE_GO` if $0.04 \le r < 0.06$ with $q < 0.05$; `GO` if $r \ge 0.06$ with $q < 0.05$). Under this rule, pass probability is **$84.5\%$** at $r_{\text{true}} = 0.0625$ and **$96.0\%\text{--}97.4\%$** at $r_{\text{true}} = 0.078\text{--}0.082$.
   - **Gate 3** replaces the fixed $\Delta r$ threshold with a **clustered paired bootstrap over stories** (circular blocks $\ge 10\text{ TRs}$ within story, $B = 10,000$), passing if the **95% CI lower bound for $\Delta r = r_{\text{GPT2}}(\text{MD}_{\text{resid}}) - r_{\text{comparator}}(\text{MD}_{\text{resid}})$ exceeds $0$**, alongside a 4-rung comparator claim ladder.

2. **Cross-Half Disattenuation Uncertainty at $N = 5$ per Half — Why We Do Not Claim "$>90\%$ Unexplained":**
   - On a 10-subject sample (`pieman`), splitting into two halves of $N_{\text{half}} = 5$ yields theoretical half-sample variance reliabilities (from pairwise $\hat{\rho}$) of **$\text{NC}_{\text{var}}(5) = 0.189$ (`LANG`)** and **$\text{NC}_{\text{var}}(5) = 0.105$ (`MD`)**.
   - Because the attenuation denominator $\sqrt{\text{NC}_{\text{var, LANG}}(5) \cdot \text{NC}_{\text{var, MD}}(5)} = \mathbf{0.141}$ is small, dividing any cross-half correlation by $0.141$ amplifies estimation noise by **$7.1\times$**. Across $K=20$ random 5-vs-5 splits in Cell 22, the raw cross-half network-mean correlation was $-0.018 \pm 0.031$ and the disattenuated value was $-0.130 \pm 0.218$. This interval is too wide to estimate how much reliable `MD` variance `LANG` explains. It is also inconsistent with an earlier rough planning estimate of $0.5\text{--}0.6$, so that question remains open until the full cohorts are used.
   - Therefore, we **remove the "$>90\%$ unexplained" claim** and use $K=20$ random cross-half splits across the full subject cohorts of all 16 stories in Stage A.

3. **Single-Story ($N = 10$) Vertex $\hat{\rho} > 0.015$ Is Near Chance — Out-of-Fold Pooled Circular-Shift BH-FDR Required:**
   - In Cell 22, we generated $n_{\text{null}} = 200$ independent within-subject circular-shift nulls across the 3,405 LH `MD` vertices on `pieman` ($N=10, T=266$).
   - The empirical null SE of a single vertex's $\hat{\rho}$ was **$0.0105$**, about half the earlier planning estimate of $0.02$. Under the null, **$8.2\%$** of vertices exceeded $0.015$, compared with **$25.9\%$** of observed `MD` vertices, which is about $3\times$ enrichment rather than chance.
   - Even so, **0 of 3,405 `MD` vertices survive BH-FDR $q < 0.05$** on one story. This is partly a resolution limit: with 200 shifts, the smallest possible $p$ is $1/201 \approx 0.005$, which BH over 3,405 tests can rarely pass.
   - Thus, the single-story $N=10$ count (`882` vertices with $\hat{\rho} > 0.015$) is **not** quoted as evidence of reliable vertex counts. Instead, `v1.2.0` mandates that vertex selection in Stage A is computed **strictly inside training folds by pooling $\hat{\rho}$ across all training stories weighted by $T_{\text{eff}, s}$**, and requiring **both BH-FDR $q < 0.05$ (against $\ge 200$ circular shifts) and $\hat{\rho}_{\text{pooled}} \ge 0.015$**.

4. **Reconciling Shain's `0.377` vs. Narratives' `0.644` Ceiling Definition:**
   - Inspecting [Idea1POC/01_prepare_data.py](01_prepare_data.py#L254) and [Idea1POC/common.py](common.py#L174) revealed why the two ceilings looked $2\times$ apart:
     - `01_prepare_data.py` computed Shain's noise ceiling via `spearman_brown(r_half) = 2 * r_half / (1 + r_half)` **without taking the square root**. That quantity is the **variance reliability ($\text{NC}_{\text{var}} = 0.3768$)**, whereas its **correlation ceiling is $\text{NC}_r = \sqrt{0.3768} = \mathbf{0.614}$**.
     - In Narratives (Regime C, 16 tasks), the $T_{\text{eff}}$-weighted mean **variance reliability is $\overline{\text{NC}}_{\text{var}} = \mathbf{0.421}$** and the **correlation ceiling is $\overline{\text{NC}}_r = \sqrt{\text{NC}_{\text{var}}} = \mathbf{0.644}$**.
   - Comparing **like-for-like**:
     - Under $\text{NC}_r = \sqrt{\text{NC}_{\text{var}}}$: Shain's Layer 9 residual `MD` ($r = 0.0784$) is $r / \text{NC}_r = 0.0784 / 0.6138 = \mathbf{0.128}$.
     - Under $\text{NC}_{\text{var}}$: Shain's Layer 9 residual `MD` is $r / \text{NC}_{\text{var}} = 0.0784 / 0.3768 = \mathbf{0.208}$.
     - Because Narratives' group ceiling ($\overline{\text{NC}}_{\text{var}} = 0.421, \overline{\text{NC}}_r = 0.644$) is slightly higher than Shain's ($\text{NC}_{\text{var}} = 0.377, \text{NC}_r = 0.614$), transferring Shain's effect predicts $r_{\text{expected}} \approx 0.082$ (or $0.060\text{--}0.082$ after accounting for group-atlas vs. individual-localizer dilution). In `v1.2.0`, we report **both $r / \text{NC}_r$ and $r / \text{NC}_{\text{var}}$** so comparisons with Shain are unambiguous.

5. **Versioned Changelog & Pre-Declared Primary Layer 9:**
   - [Idea1POC/narratives_gate_config.json](narratives_gate_config.json) is now versioned at **`1.3.0`**, superseding `1.2.0`, with a dated Gate 1 rationale. It proposes **GPT-2 Layer 9** as the sole primary gating layer (Layers 6 & 12 report-only). Freeze timestamp and commit hash remain placeholders; the specification is not yet frozen or implemented.

---

## 2. Summary of the Draft `v1.3.0` Gate Rules ([Idea1POC/narratives_gate_config.json](narratives_gate_config.json))

| Component | `v1.3.0` Specification (Planned, Not Executed) |
|---|---|
| **Primary Story Pool** | **Regime C (16 intact tasks incl. `schema`, $T_{\text{eff}} = 2,270.8$)** as primary; **Regime B (15 continuous stories, $T_{\text{eff}} = 2,080.2$)** as sensitivity. `piemanpni` merged into the Pie Man CV fold; `notthefall*scram` excluded from encoding folds. |
| **Primary Model & Layer** | `gpt2`, 5-TR ($7.5\text{ s}$) context window, lags `[1, 2, 3, 4]` TRs ($1.5\text{--}6.0\text{ s}$). **Primary layer = Layer 9** (declared in advance from Shain). |
| **Primary Target Level** | **Vertex-level** (out-of-fold pooled $\hat{\rho}_{\text{train}}$ across training stories with BH-FDR $q < 0.05$ under $\ge 200$ circular shifts AND $\hat{\rho}_{\text{train}} \ge 0.015$). **Secondary:** Parcel-level weighted by $w_p \propto \max(\hat{\rho}_{\text{train}}(p), 0)$. |
| **Residualization** | Disjoint subject-half cross-fitting (`LANG` from random half A, `MD` from random half B, both directions averaged over $K=20$ random splits, lags `[0, 1, 2]` TRs). |
| **Gate 1 (Blocking)** | LANG and raw MD must **each** have permutation **$q < 0.05$ AND $r \ge 0.04$** on held-out stories at Layer 9. Positive-tail circular-shift tests, at least 999 permutations, BH adjustment across the two primary network tests. Show observed r/p/q and reference-only Shain values beside results; Narratives expectations remain unestimated. |
| **Gate 2 (Blocking)** | Predict cross-half residualized `MD` at Layer 9: **permutation $q < 0.05$ ($n_{\text{perm}} \ge 999$, shift $\ge 10\text{ TRs}$) AND $r_{\text{obs}} \ge 0.04$**. (`NO_GO`: $q \ge 0.05$ or $r < 0.04$; `SUBTLE_GO`: $0.04 \le r < 0.06$; `GO`: $r \ge 0.06$). |
| **Gate 3 (Blocking)** | $\Delta r = r_{\text{GPT2}}(\text{MD}_{\text{resid}}) - r_{\text{comparator}}(\text{MD}_{\text{resid}})$ tested via **clustered paired bootstrap over stories** ($\ge 10\text{-TR}$ circular blocks within story, $B = 10,000$). **Pass rule:** 95% CI lower bound $> 0$ vs. extended low-level comparator; evaluated up the 4-rung comparator ladder (extended low-level $\to$ static embeddings $\to$ untrained GPT-2 $\to$ word-shuffled GPT-2). |
| **Gate 4 (Non-Blocking Diagnostic)** | Mean cosine of top 5 principal angles (`LANG` vs. `MD` weights) compared against the `LANG` split-half weight ceiling and a permuted-`MD`-target null (reported with story-bootstrap 95% CI). |

---

## 3. Master Empirical Table From [eda_narratives_processed.ipynb](../eda_narratives_processed.ipynb)

| Check | Empirical Finding |
|---|---|
| **Inventory & Timing** | 345 subjects, 19 BIDS tasks (27 stories), uniform $\text{TR} = 1.5\text{ s}$. WAV lead-in offset (`onset_TR * 1.5s`) fixed; transient-trimmed `[10 : -5]` speech activity correlates with `EAC` at **$r = 0.352$** (lag 1 TR). |
| **Pairwise ISC ($\hat{\rho}$) on `pieman` ($N=10$)** | `EAC` $\hat{\rho} = 0.1466$; `LANG` $\hat{\rho} = 0.0444$; **`MD` $\hat{\rho} = 0.0229$** (within-subject `run-1` vs. `run-2` repeat $r = 0.0569$, $2.49\times$ pairwise $\hat{\rho}$). |
| **True `MD` Group Ceilings (from $\hat{\rho}=0.0229$)** | $\text{NC}_r(N=10/14/20/46/75) = \mathbf{0.435 / 0.497 / 0.565 / 0.720 / 0.798}$; $\text{NC}_{\text{var}}(N=10/14/20/46/75) = \mathbf{0.190 / 0.247 / 0.319 / 0.518 / 0.637}$. |
| **Shain vs. Narratives Ceiling Reconciliation** | Shain's `nc_md_mean = 0.377` in `01_prepare_data.py` is $\text{NC}_{\text{var}}$ (no sqrt $\rightarrow \text{NC}_r = \sqrt{0.377} = \mathbf{0.614}$). Narratives Regime C weighted mean is $\overline{\text{NC}}_{\text{var}} = \mathbf{0.421}, \overline{\text{NC}}_r = \mathbf{0.644}$. |
| **Single-Story ($N=10$) Vertex Null ($n_{\text{null}}=200$)** | Empirical null $\text{SE}(\hat{\rho}) = 0.0105$; **0 / 3,405 `MD` vertices survive BH-FDR $q < 0.05$ on 1 story ($N=10$)** $\rightarrow$ confirms vertex selection must pool $\hat{\rho}$ across all training stories out-of-fold. |
| **Intact vs. Scrambled (`notthefall`, $N=5$)** | Pairwise $\hat{\rho}$ drops by $>57\%$ on scrambled speech in `LANG` ($+0.1457 \to +0.0625$) and `MD` ($+0.1053 \to +0.0441$), while remaining high in `EAC` ($+0.1667 \to +0.1384$). |
