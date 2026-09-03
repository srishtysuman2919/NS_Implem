"""Prepare real Haxby 2001 subject-2 patterns for RSA experiments."""
from __future__ import annotations

import sys
import gzip
import shutil
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import numpy as np

from _utils import DOWNLOAD_DIR, banner, save_dataset


def main() -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to prepare Haxby data") from exc

    source_dir = DOWNLOAD_DIR / "haxby" / "haxby2001" / "subj2"
    bold_path = source_dir / "bold.nii.gz"
    uncompressed_bold_path = source_dir / "bold.nii"
    labels_path = source_dir / "labels.txt"
    mask_path = source_dir / "mask4_vt.nii.gz"
    for path in (bold_path, labels_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(f"missing Haxby source file: {path}")

    labels_table = np.genfromtxt(labels_path, names=True, dtype=None, encoding="utf8")
    trial_labels = np.asarray(labels_table["labels"], dtype=str)
    chunks = np.asarray(labels_table["chunks"], dtype=np.int64)
    condition_names = np.array(sorted(set(trial_labels) - {"rest"}))
    run_ids = np.unique(chunks)

    if not uncompressed_bold_path.exists():
        banner("decompressing Haxby BOLD file for memory-mapped access")
        with gzip.open(bold_path, "rb") as source, uncompressed_bold_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    bold_img = nib.load(uncompressed_bold_path, mmap=True)
    mask = np.asarray(nib.load(mask_path).dataobj, dtype=bool)
    if bold_img.shape[:3] != mask.shape or bold_img.shape[3] != len(trial_labels):
        raise ValueError("Haxby BOLD, mask, and labels have incompatible shapes")

    mask_indices = np.flatnonzero(mask.ravel())
    time_series = np.empty((bold_img.shape[3], len(mask_indices)), dtype=np.float32)
    for timepoint in range(bold_img.shape[3]):
        volume = np.asarray(bold_img.dataobj[..., timepoint], dtype=np.float32)
        time_series[timepoint] = volume.ravel()[mask_indices]

    patterns = np.empty(
        (len(run_ids), len(condition_names), len(mask_indices)), dtype=np.float32
    )
    for run_index, run_id in enumerate(run_ids):
        run_mask = chunks == run_id
        run_data = time_series[run_mask]
        run_data = (run_data - run_data.mean(axis=0)) / (
            run_data.std(axis=0) + 1e-6
        )
        run_labels = trial_labels[run_mask]
        for condition_index, condition in enumerate(condition_names):
            condition_mask = run_labels == condition
            if not condition_mask.any():
                raise ValueError(f"run {run_id} has no {condition!r} samples")
            patterns[run_index, condition_index] = run_data[condition_mask].mean(axis=0)

    odd_mean = patterns[::2].mean(axis=0)
    even_mean = patterns[1::2].mean(axis=0)
    from methods.classic_rsa import rdm

    reference_rdm = rdm(odd_mean, metric="correlation").astype(np.float32)
    searchlight_volume = np.zeros((*mask.shape, len(condition_names)), dtype=np.float32)
    for condition_index in range(len(condition_names)):
        searchlight_volume[..., condition_index][mask] = even_mean[condition_index]

    path = save_dataset(
        "haxby",
        patterns=patterns,
        labels=np.arange(len(condition_names), dtype=np.int64),
        label_names=condition_names,
        volume=searchlight_volume,
        mask=mask,
        reference_rdm=reference_rdm,
        affine=np.asarray(bold_img.affine),
        source=np.array("Haxby 2001 subject 2, ventral-temporal ROI (real data)"),
    )
    banner(
        f"wrote {path} ({len(run_ids)} runs, {len(condition_names)} conditions, "
        f"{len(mask_indices)} VT voxels)"
    )


if __name__ == "__main__":
    main()