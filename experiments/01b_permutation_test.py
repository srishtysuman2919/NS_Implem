"""Permutation test for Experiment 1 (Haxby method agreement).

Tests whether the observed RDM agreement is significantly different from chance.
Method: shuffle stimulus labels independently for each run, recompute RDM, compare
to real RDM under each metric. Report p-value and effect size.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from data._utils import load_dataset
from methods.classic_rsa import rdm, rdm_similarity
from methods.crossnobis import crossnobis_rdm


def _rdm_to_vector(D: np.ndarray) -> np.ndarray:
    """Extract upper triangle of RDM as vector for correlation."""
    return D[np.triu_indices(len(D), k=1)]


def compute_real_rdm_agreement() -> dict:
    """Compute observed agreement between classic RSA and crossnobis on Haxby."""
    ds = load_dataset("haxby")
    patterns = ds["patterns"]  # (n_runs, n_conds, n_vox)
    
    mean_patterns = patterns.mean(axis=0)
    rdm_classic = rdm(mean_patterns, metric="correlation")
    rdm_cn = crossnobis_rdm(patterns)
    
    # Agreement metric: Spearman correlation of RDM upper triangles
    v_classic = _rdm_to_vector(rdm_classic)
    v_cn = _rdm_to_vector(rdm_cn)
    
    rho_real, _ = stats.spearmanr(v_classic, v_cn)
    return {
        "rho_real": rho_real,
        "rdm_classic": rdm_classic,
        "rdm_cn": rdm_cn,
        "patterns": patterns,
    }


def permutation_test_crossnobis(patterns: np.ndarray, rdm_classic: np.ndarray,
                                 n_perm: int = 1000, random_seed: int = 42) -> dict:
    """
    Permutation test: shuffle condition labels within each run independently,
    recompute crossnobis RDM, and compare to real classic RSA RDM.
    
    Null hypothesis: agreement is due to chance label shuffling.
    """
    np.random.seed(random_seed)
    n_runs, n_conds, n_vox = patterns.shape
    
    rho_perm_list = []
    
    for perm_idx in range(n_perm):
        # Shuffle condition labels within each run independently
        perm_patterns = patterns.copy()
        for run_idx in range(n_runs):
            perm_order = np.random.permutation(n_conds)
            perm_patterns[run_idx] = perm_patterns[run_idx, perm_order, :]
        
        # Compute permuted crossnobis RDM
        rdm_cn_perm = crossnobis_rdm(perm_patterns)
        
        # Compare to real classic RSA RDM
        v_classic = _rdm_to_vector(rdm_classic)
        v_cn_perm = _rdm_to_vector(rdm_cn_perm)
        
        rho_perm, _ = stats.spearmanr(v_classic, v_cn_perm)
        rho_perm_list.append(rho_perm)
        
        if (perm_idx + 1) % 200 == 0:
            print(f"  completed {perm_idx + 1}/{n_perm} permutations")
    
    return {"rho_perm": np.array(rho_perm_list)}


def compute_p_value_and_effect(rho_real: float, rho_perm: np.ndarray) -> dict:
    """
    Compute p-value and effect size (Cohen's d) for observed agreement.
    
    Two-tailed test: observed rho is significantly different from null distribution.
    """
    # One-tailed p-value: proportion of permutations with rho >= observed
    # (higher agreement is "more significant")
    p_value = np.mean(rho_perm >= rho_real)
    
    # Cohen's d: (observed - mean_null) / std_null
    mean_perm = np.mean(rho_perm)
    std_perm = np.std(rho_perm)
    cohens_d = (rho_real - mean_perm) / (std_perm + 1e-10)
    
    return {
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mean_null": mean_perm,
        "std_null": std_perm,
        "rho_real": rho_real,
    }


def main(n_permutations: int = 1000) -> None:
    print("=" * 70)
    print("Permutation Test: Haxby Classic RSA vs. Crossnobis Agreement")
    print("=" * 70)
    print()
    
    print("Computing observed RDM agreement...")
    results_real = compute_real_rdm_agreement()
    rho_real = results_real["rho_real"]
    
    print(f"  Observed Spearman ρ (Classic vs. Crossnobis): {rho_real:.3f}")
    print()
    
    print(f"Running permutation test ({n_permutations} shuffles)...")
    results_perm = permutation_test_crossnobis(
        results_real["patterns"],
        results_real["rdm_classic"],
        n_perm=n_permutations,
    )
    rho_perm = results_perm["rho_perm"]
    print()
    
    # Compute statistics
    stats_dict = compute_p_value_and_effect(rho_real, rho_perm)
    
    print("Permutation Test Results:")
    print(f"  Observed ρ:                {stats_dict['rho_real']:.3f}")
    print(f"  Mean null ρ:               {stats_dict['mean_null']:.3f}")
    print(f"  Std null ρ:                {stats_dict['std_null']:.3f}")
    print(f"  P-value (one-tailed):      {stats_dict['p_value']:.4f}")
    print(f"  Cohen's d:                 {stats_dict['cohens_d']:.3f}")
    print()
    
    # Interpretation
    if stats_dict["p_value"] < 0.05:
        print("✓ SIGNIFICANT: Observed agreement is significantly higher than chance.")
        print("  (Crossnobis and Classic RSA genuinely agree on this data.)")
    else:
        print("✗ NOT SIGNIFICANT: Observed agreement could arise from chance.")
        print("  (Consider that shuffled labels produce similar RDM agreement.)")
    print()
    
    # Report percentiles
    pctl_5 = np.percentile(rho_perm, 5)
    pctl_95 = np.percentile(rho_perm, 95)
    print(f"Null distribution percentiles:")
    print(f"  5th percentile:  {pctl_5:.3f}")
    print(f"  95th percentile: {pctl_95:.3f}")
    print()
    
    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(rho_perm, bins=40, alpha=0.7, label="Null (shuffled labels)",
                color="steelblue", density=True)
        ax.axvline(rho_real, color="red", linestyle="--", linewidth=2,
                   label=f"Observed ρ = {rho_real:.3f}")
        ax.axvline(stats_dict["mean_null"], color="black", linestyle=":",
                   linewidth=2, label=f"Null mean = {stats_dict['mean_null']:.3f}")
        ax.set_xlabel("Spearman ρ (Classic RSA vs. Crossnobis)")
        ax.set_ylabel("Density")
        ax.set_title("Permutation Test: Classic RSA–Crossnobis Agreement\n"
                     f"(p = {stats_dict['p_value']:.4f}, d = {stats_dict['cohens_d']:.2f})")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        RESULTS_DIR = REPO_ROOT / "results"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_DIR / "01_permutation_test.png", dpi=140)
        print(f"Plot saved to results/01_permutation_test.png")
        plt.close(fig)
    except ImportError:
        print("(matplotlib not available; skipping plot)")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main(n_permutations=1000)
