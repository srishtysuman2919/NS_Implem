"""Prepare an aligned real NSD/ResNet subset for model-brain comparison."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import numpy as np

from _utils import DOWNLOAD_DIR, banner, save_dataset


def _trial_stimulus_ids(n_trials: int) -> np.ndarray:
    from scipy.io import loadmat

    design = loadmat(DOWNLOAD_DIR / "nsd_expdesign.mat")
    order = design["masterordering"].ravel()[:n_trials].astype(np.int64) - 1
    return design["subjectim"][0, order].astype(np.int64)


def _load_repeated_patterns(
    stimulus_ids: np.ndarray, selected_ids: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    import h5py

    mask_hdf5 = np.transpose(mask, (2, 1, 0))
    voxel_coordinates = np.argwhere(mask_hdf5)
    trial_patterns = np.empty(
        (len(stimulus_ids), len(voxel_coordinates)), dtype=np.float32
    )
    for session in range(3):
        with h5py.File(
            DOWNLOAD_DIR / f"betas_session{session + 1:02d}.hdf5", "r"
        ) as handle:
            betas = handle["betas"]
            session_patterns = trial_patterns[session * 750:(session + 1) * 750]
            for voxel_index, (z, y, x) in enumerate(voxel_coordinates):
                session_patterns[:, voxel_index] = betas[:, z, y, x]
    trial_patterns /= 300.0

    occurrences = {
        stimulus_id: np.flatnonzero(stimulus_ids == stimulus_id)
        for stimulus_id in selected_ids
    }
    patterns = np.empty((3, len(selected_ids), len(voxel_coordinates)), dtype=np.float32)
    for condition, stimulus_id in enumerate(selected_ids):
        patterns[:, condition] = trial_patterns[occurrences[stimulus_id][:3]]
    return patterns


def _extract_resnet(stimulus_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import h5py
    import torch
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).eval()
    layer_names = np.array(["conv1", "layer1", "layer2", "layer3", "layer4"])
    modules = dict(model.named_modules())
    activations: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
    hooks = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            pooled = output.mean(dim=(-2, -1))
            activations[name].append(pooled.detach().cpu().numpy())
        return hook

    for name in layer_names:
        hooks.append(modules[name].register_forward_hook(make_hook(str(name))))

    preprocess = weights.transforms()
    with h5py.File(DOWNLOAD_DIR / "nsd_stimuli.hdf5", "r") as handle:
        images = handle["imgBrick"]
        with torch.no_grad():
            for start in range(0, len(stimulus_ids), 16):
                batch_ids = stimulus_ids[start:start + 16]
                batch = torch.stack([
                    preprocess(torch.from_numpy(np.asarray(images[int(index)])).permute(2, 0, 1))
                    for index in batch_ids
                ])
                model(batch)
    for hook in hooks:
        hook.remove()

    embeddings = np.empty(len(layer_names), dtype=object)
    for index, name in enumerate(layer_names):
        embeddings[index] = np.concatenate(activations[str(name)], axis=0).astype(np.float32)
    return layer_names, embeddings


def main(n_voxels: int = 512) -> None:
    import nibabel as nib

    n_trials = 3 * 750
    stimulus_ids = _trial_stimulus_ids(n_trials)
    unique_ids, counts = np.unique(stimulus_ids, return_counts=True)
    selected_ids = unique_ids[counts >= 3]
    mask = np.asarray(nib.load(DOWNLOAD_DIR / "nsdgeneral.nii.gz").dataobj) > 0
    patterns = _load_repeated_patterns(stimulus_ids, selected_ids, mask)

    variances = patterns.reshape(-1, patterns.shape[-1]).var(axis=0)
    selected_voxels = np.argsort(variances)[-n_voxels:]
    patterns = patterns[..., selected_voxels]
    layer_names, embeddings = _extract_resnet(selected_ids)

    path = save_dataset(
        "nsd_model_brain",
        patterns=patterns,
        stimulus_ids=selected_ids,
        layer_names=layer_names,
        embeddings=embeddings,
        source=np.array(
            "NSD subject 1 sessions 1-3 (real betas/images) + pretrained ResNet-18"
        ),
    )
    banner(
        f"wrote {path} (3 repeats, {len(selected_ids)} images, "
        f"{patterns.shape[-1]} voxels)"
    )


if __name__ == "__main__":
    main()