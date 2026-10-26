"""Read-only real Pie Man LH cache smoke; not an encoding experiment."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from narratives_prepare import inventory_without_config, load_surface, preprocess_segment
from narratives_stage_a import pairwise_isc


def main():
    repository = Path(__file__).resolve().parents[1]
    root = repository / "data" / "narratives_meta"
    atlas = nib.load(str(root / "tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii"))
    labels = np.asarray(atlas.darrays[0].data)
    names = {value.replace("L_", "").replace("_ROI", ""): key
             for key, value in atlas.labeltable.get_labels_as_dict().items() if key > 0}
    config = json.loads((repository / "Idea1POC" / "narratives_gate_config.json").read_text())
    vertices = np.flatnonzero(np.isin(labels, [names[name] for name in config["targets"]["networks"]["MD_parcels_per_hemi"]]))
    event = json.loads((root / "event_meta.json").read_text())["pieman"]["pieman"]
    indices = list(range(event["onset"] + 10, event["onset"] + event["duration"] - 5))
    subjects = []
    for number in range(2, 12):
        identity = f"sub-{number:03d}"
        candidates = [path for path in (root / "processed_cache" / "afni-nosmooth" / identity).glob(
            "*task-pieman*hemi-L_desc-clean.func.gii") if "_run-2_" not in path.name]
        if len(candidates) != 1:
            raise ValueError(f"Expected one cached first-exposure LH file for {identity}")
        data = load_surface(candidates[0], "L")
        subjects.append(preprocess_segment(data, indices, vertices, event["onset"] + event["duration"]))
    stack = np.stack(subjects)
    rho = pairwise_isc(stack)
    if stack.shape != (10, 266, len(vertices)) or not np.isfinite(rho).all():
        raise ValueError("Unexpected cache shape or nonfinite ISC")
    np.testing.assert_allclose(stack.mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(stack.std(axis=1), 1, atol=1e-5)
    print(json.dumps({"check": "Real cached Pie Man LH input/ISC; NOT encoding or a primary gate",
                      "boundary_rings": 0, "primary_eligible": False,
                      "shape": list(stack.shape), "mean_MD_pairwise_ISC": float(rho.mean()),
                      "finite": True}, indent=2))
    inventory = inventory_without_config(root)
    print(json.dumps({key: inventory[key] for key in (
        "status", "primary_eligible", "selected_scans", "excluded_scans", "local_clean_files", "missing_clean_files")}, indent=2))


if __name__ == "__main__":
    main()
