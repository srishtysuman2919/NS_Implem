"""Prepare a Kriegeskorte-92-style dataset.

Tries to use `rsatoolbox`'s bundled 92-image RDM data (fastest path). If
that's not available, synthesises a 92-condition multi-run dataset with the
well-known animate/inanimate + face/body/artificial/natural structure so
every method has something meaningful to compute on.

Outputs
-------
data/cache/kriegeskorte92.npz
    patterns : (n_runs, 92, n_voxels)  synthetic multi-run response patterns
    labels   : (92,) int super-category label (0..3)
    label_names : (4,) str
    it_rdm   : (92, 92) IT-fMRI RDM (either real or synthesised from patterns)
    source   : str
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable so `methods` and `data._utils` both resolve.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import numpy as np

from _utils import banner, save_dataset


LABEL_NAMES = np.array(["animate_face", "animate_body", "inanimate_natural", "inanimate_artificial"])
N_COND = 92


def _try_local_mat() -> np.ndarray | None:
    """Load and average the original 4-subject, 2-session human IT RDMs."""
    path = _THIS.parent / "92_brainRDMs.mat"
    if not path.exists():
        return None
    try:
        from scipy.io import loadmat

        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        entries = np.asarray(mat["RDMs"]).ravel()
        rdms = np.stack([entry.RDM for entry in entries])
        if rdms.shape != (8, N_COND, N_COND):
            raise ValueError(f"unexpected RDM shape {rdms.shape}")
        return rdms.mean(axis=0).astype(np.float32)
    except Exception as exc:
        banner(f"could not load {path.name}: {exc}")
        return None


def _try_rsatoolbox() -> np.ndarray | None:
    """Return the real 92-condition IT RDM if rsatoolbox happens to ship it."""
    try:
        import rsatoolbox                             # noqa: F401
    except Exception:
        return None
    try:
        # Different rsatoolbox versions have shipped this dataset under
        # different names; we probe a few.
        from rsatoolbox.data import load_92           # type: ignore
        ds = load_92()
        # ds.get_measurements() -> (n_channels, n_conditions) in some versions
        # We just want the group-averaged IT RDM.
        from rsatoolbox.rdm import calc_rdm
        rdms = calc_rdm(ds, method="correlation")
        return rdms.get_matrices()[0]
    except Exception:
        pass
    return None


def _synthesise(rng: np.random.Generator, n_voxels: int = 500, n_runs: int = 6,
                noise_scale: float = 1.2) -> dict:
    """Synthesise 92 conditions with 4 hidden super-categories."""
    # 4 super-categories, roughly 23 exemplars each
    n_per_cat = [23, 23, 23, 23]
    labels = np.concatenate([[c] * n_per_cat[c] for c in range(4)])
    assert len(labels) == N_COND

    # Category-level prototypes in a low-dim latent space
    latent_dim = 8
    proto = rng.standard_normal((4, latent_dim)) * 2.0
    # Exemplar-level variation
    latent = np.zeros((N_COND, latent_dim))
    for i in range(N_COND):
        latent[i] = proto[labels[i]] + 0.7 * rng.standard_normal(latent_dim)

    # Random projection to voxel space
    W = rng.standard_normal((latent_dim, n_voxels)) / np.sqrt(latent_dim)
    true_patterns = latent @ W                        # (92, n_voxels)

    # multi-run noisy measurements
    patterns = (
        true_patterns[None] + noise_scale * rng.standard_normal(
            (n_runs, N_COND, n_voxels)
        )
    )

    # "IT" RDM: 1 - Pearson of clean patterns (this is what a hypothetical
    # noise-free brain would produce). In the real dataset this is the human
    # IT fMRI RDM from Kriegeskorte 2008.
    from methods.classic_rsa import rdm

    it_rdm = rdm(true_patterns, metric="correlation")

    return dict(
        patterns=patterns.astype(np.float32),
        labels=labels.astype(np.int64),
        label_names=LABEL_NAMES,
        it_rdm=it_rdm.astype(np.float32),
        source=np.array("synthesised (4 super-categories, 6 runs, 500 voxels)"),
    )


def main() -> None:
    rng = np.random.default_rng(0)

    real_rdm = _try_local_mat()
    real_source = "Kriegeskorte 2008 hIT RDM (4 subjects x 2 sessions)"
    if real_rdm is None:
        real_rdm = _try_rsatoolbox()
        real_source = "rsatoolbox bundled IT RDM"
    data = _synthesise(rng)
    if real_rdm is not None and real_rdm.shape == (N_COND, N_COND):
        banner(f"using {real_source}")
        data["it_rdm"] = real_rdm.astype(np.float32)
        data["source"] = np.array(
            f"{real_source} + synthesised runs for crossnobis"
        )
    else:
        banner("rsatoolbox bundled data not found; synthesising")
        banner(f"    -> 92 conditions, {data['patterns'].shape[0]} runs, "
               f"{data['patterns'].shape[2]} voxels")
        banner(f"    -> structure: {len(LABEL_NAMES)} super-categories")

    p = save_dataset("kriegeskorte92", **data)
    banner(f"wrote {p}")


if __name__ == "__main__":
    main()
