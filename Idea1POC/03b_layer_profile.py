#!/usr/bin/env python
"""Run the GO/NO-GO gate across layers and summarize the alignment profile."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


def load_gate_module():
    path = Path(__file__).with_name("03_gate_analysis.py")
    spec = spec_from_file_location("gate_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values in original order."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def run_layer(
    prepared: str,
    features_dir: str,
    out_dir: str,
    layer: int,
    n_perm: int,
    seed: int,
    resume: bool,
) -> tuple[int, str]:
    feature_path = os.path.join(features_dir, f"layer{layer}.npy")
    result_path = os.path.join(out_dir, f"layer{layer}.json")
    if resume and os.path.exists(result_path):
        with open(result_path) as result_file:
            json.load(result_file)
        return layer, result_path

    gate = load_gate_module()
    gate.run(prepared, feature_path, (0,), result_path, n_perm, seed)
    return layer, result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", default="results/gpt2_layer_profile")
    parser.add_argument("--n_perm", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    os.makedirs(args.out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_layer,
                args.prepared,
                args.features_dir,
                args.out_dir,
                layer,
                args.n_perm,
                args.seed,
                args.resume,
            ): layer
            for layer in args.layers
        }
        for future in as_completed(futures):
            layer, result_path = future.result()
            print(f"completed layer {layer} -> {result_path}", flush=True)

    rows = []
    for layer in args.layers:
        result_path = os.path.join(args.out_dir, f"layer{layer}.json")
        with open(result_path) as result_file:
            result = json.load(result_file)
        rows.append({
            "layer": layer,
            "r_lang": result["A_encoding"]["r_lang_mean"],
            "r_md": result["A_encoding"]["r_md_mean"],
            "r_md_residual": result["D_residual_MD"]["r_md_residual_mean"],
            "md_retained_fraction": result["D_residual_MD"]["retained_fraction"],
            "r_lang_residual": result["D_residual_LANG"]["r_lang_residual_mean"],
            "subspace_mean_cos": result["E_weight_subspace"]["mean_cos"],
            "p_md_residual": result["F_permutation"]["p_value"],
        })

    adjusted = bh_fdr([row["p_md_residual"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["q_md_residual_fdr"] = q_value

    csv_path = os.path.join(args.out_dir, "profile.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out_dir, "profile.json"), "w") as json_file:
        json.dump(rows, json_file, indent=2)

    print("\n=== layer profile ===")
    for row in rows:
        print(
            f"L{row['layer']:>2}: LANG={row['r_lang']:.3f} MD={row['r_md']:.3f} "
            f"MD-resid={row['r_md_residual']:.3f} "
            f"p={row['p_md_residual']:.3f} q(FDR)={row['q_md_residual_fdr']:.3f}"
        )
    print(f"\nsaved profile -> {csv_path}")


if __name__ == "__main__":
    main()