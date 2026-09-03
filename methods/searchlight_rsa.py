"""Searchlight RSA -- Kriegeskorte, Goebel & Bandettini (2006).

Paper goal
----------
Prior fMRI analyses were either (a) univariate GLM contrasts, which look at
each voxel independently and miss distributed information; or (b)
multivariate pattern analysis in a *pre-committed* region of interest, which
only produces one number per ROI and cannot localise information within the
brain. Kriegeskorte 2006 proposed the **searchlight**: slide a small sphere
of radius r (measured in voxels) through the volume, and at each voxel
compute a local multivariate statistic (originally a Mahalanobis-based
information score; RSA-style local RDM comparison came shortly after). The
per-centre statistic gives a **whole-volume map of information content**
without requiring an ROI.

Contents of this module
-----------------------
* :func:`spherical_neighborhood_indices` -- precompute the voxel offsets that
  fall inside a sphere of radius ``r``. Written explicitly so the geometry is
  obvious.
* :func:`searchlight_rdm_map`            -- for a 4-D ``(nx, ny, nz, n_cond)``
  volume, produce a 4-D ``(nx, ny, nz, n_cond, n_cond)`` RDM map or a
  3-D map of RDM-vs-reference correlations.
* :func:`searchlight_similarity_map`     -- convenience wrapper: given a
  reference RDM, produce a scalar map of Spearman ρ per centre voxel.

Only volumetric (3-D grid) inputs are supported here. Cortical-surface
searchlights are a natural extension (see e.g. Oosterhof et al. 2011) but
require a mesh + geodesic distance and are out of scope for this study repo.

What shortcoming of the previous method does this address?
---------------------------------------------------------
Classic RSA gives one RDM per ROI, so you can only ask "is there
representational structure in *this* pre-chosen region?". The searchlight
turns RSA into a whole-brain, spatially resolved question: "*where* in the
brain does representational structure X show up?" -- without needing to
guess the region first.
"""
from __future__ import annotations

import numpy as np

from .classic_rsa import rdm
from .rdm_utils import compare_rdms


def spherical_neighborhood_indices(radius: float) -> np.ndarray:
    """Return integer offsets `(dx, dy, dz)` of voxels within a sphere.

    Parameters
    ----------
    radius : float
        Sphere radius, in voxel units.

    Returns
    -------
    (k, 3) ndarray of int
        Offsets relative to the centre voxel, including (0, 0, 0).
    """
    r = int(np.ceil(radius))
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=int)


def _extract_patch(
    volume: np.ndarray,
    centre: tuple[int, int, int],
    offsets: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    """Extract a ``(n_conditions, n_voxels_in_patch)`` matrix at ``centre``.

    Returns ``None`` if the patch is empty (all offsets fall outside the mask
    or the volume). ``volume`` has shape ``(nx, ny, nz, n_conditions)``.
    """
    nx, ny, nz, _ = volume.shape
    cx, cy, cz = centre
    coords = offsets + np.array([cx, cy, cz])
    in_bounds = (
        (coords[:, 0] >= 0) & (coords[:, 0] < nx) &
        (coords[:, 1] >= 0) & (coords[:, 1] < ny) &
        (coords[:, 2] >= 0) & (coords[:, 2] < nz)
    )
    coords = coords[in_bounds]
    if mask is not None:
        keep = mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        coords = coords[keep]
    if len(coords) < 3:                # need enough voxels for a meaningful RDM
        return None
    # (n_voxels, n_conditions) -> transpose to (n_conditions, n_voxels)
    patch = volume[coords[:, 0], coords[:, 1], coords[:, 2], :]
    return patch.T


def searchlight_similarity_map(
    volume: np.ndarray,
    reference_rdm: np.ndarray,
    radius: float = 3.0,
    mask: np.ndarray | None = None,
    metric: str = "correlation",
    compare: str = "spearman",
) -> np.ndarray:
    """Per-voxel map of Spearman ρ between local RDM and ``reference_rdm``.

    Parameters
    ----------
    volume : (nx, ny, nz, n_conditions) ndarray
        Response amplitude per condition per voxel. In fMRI this is usually a
        stack of beta maps from a first-level GLM.
    reference_rdm : (n_conditions, n_conditions) ndarray
        The target RDM the local RDMs are compared against (e.g. a model RDM,
        or a behavioural RDM).
    radius : float
        Searchlight sphere radius in voxels. Kriegeskorte 2006 used 3--4.
    mask : (nx, ny, nz) bool ndarray or None
        Only compute at centres where ``mask`` is True and only include
        voxels inside ``mask`` in each sphere. Recommended: a brain / grey-
        matter mask.

    Returns
    -------
    (nx, ny, nz) ndarray of float, NaN where no valid RDM could be built.
    """
    nx, ny, nz, n_cond = volume.shape
    if reference_rdm.shape != (n_cond, n_cond):
        raise ValueError(
            f"reference_rdm shape {reference_rdm.shape} does not match "
            f"n_conditions {n_cond}"
        )
    offsets = spherical_neighborhood_indices(radius)
    out = np.full((nx, ny, nz), np.nan, dtype=float)

    if mask is None:
        centres = np.ndindex(nx, ny, nz)
    else:
        centres = zip(*np.where(mask))

    for cx, cy, cz in centres:
        patch = _extract_patch(volume, (cx, cy, cz), offsets, mask)
        if patch is None:
            continue
        try:
            local_rdm = rdm(patch, metric=metric)
        except Exception:
            continue
        # Compare local RDM to reference
        r = compare_rdms(local_rdm, reference_rdm, method=compare)
        out[cx, cy, cz] = r
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    # Build a tiny synthetic volume where a hidden 4-voxel cluster carries the
    # "true" representation and the rest is noise.
    nx, ny, nz, n_cond = 8, 8, 8, 10
    true_patterns = rng.standard_normal((n_cond, 4))
    volume = rng.standard_normal((nx, ny, nz, n_cond)) * 0.1
    hotspot = (3, 3, 3)
    idx = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                if idx >= 4:
                    break
                volume[hotspot[0] + dx, hotspot[1] + dy, hotspot[2] + dz, :] += (
                    true_patterns[:, idx]
                )
                idx += 1

    ref_rdm = rdm(true_patterns, metric="correlation")
    r_map = searchlight_similarity_map(volume, ref_rdm, radius=2.0)

    # Peak of the map should be near the hotspot.
    peak = np.unravel_index(np.nanargmax(r_map), r_map.shape)
    d = np.linalg.norm(np.array(peak) - np.array(hotspot))
    assert d < 2.5, f"peak {peak} too far from hotspot {hotspot}"
    print("searchlight_rsa self-test OK")
    print(f"  peak voxel     : {peak}   (hotspot was {hotspot})")
    print(f"  peak ρ         : {np.nanmax(r_map):+.3f}")
    print(f"  fraction NaN   : {np.isnan(r_map).mean():.2f}")


if __name__ == "__main__":
    _selftest()
