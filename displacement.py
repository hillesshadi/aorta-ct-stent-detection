"""
displacement.py
===============
Phase 4 – Centroid Tracking & Displacement Computation

Computes the 3-D physical position of the stent centroid for each
time-point and derives displacement vectors and statistics.

All positions are expressed in millimetres using the image spacing and
origin from the DICOM metadata.

Functions
---------
centroid_physical        – centroid (x,y,z) in mm
voxel_coords_physical    – all stent voxel positions in mm
compute_displacements    – pairwise displacement vectors (ΔX, ΔY, ΔZ)
displacement_statistics  – magnitude, mean, max, std per vector
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def centroid_physical(
    mask: np.ndarray,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
) -> Optional[np.ndarray]:
    """
    Centroid of binary mask in physical (mm) space.

    FIX: Explicit axis labelling to prevent Z/X confusion.
    SimpleITK spacing = (sx, sy, sz) → X, Y, Z physical order.
    np.argwhere on (Z,Y,X) array → columns are [z_idx, y_idx, x_idx].
    """
    """Compute the geometric centroid of a binary mask in physical (mm) space.

    Parameters
    ----------
    mask : np.ndarray, shape (Z, Y, X)
        Binary stent mask.
    spacing : tuple of float
        Voxel spacing (sx, sy, sz) in mm.
    origin : tuple of float
        Physical origin (ox, oy, oz) in mm.

    Returns
    -------
    np.ndarray of shape (3,) – [x_mm, y_mm, z_mm] or **None** if mask is empty.
    """
    pos = np.argwhere(mask > 0)   # (N, 3) array of [z_idx, y_idx, x_idx]

    if pos.size == 0:
        return None

    # pos columns: [0]=z_idx, [1]=y_idx, [2]=x_idx
    z_idx_mean = pos[:, 0].mean()
    y_idx_mean = pos[:, 1].mean()
    x_idx_mean = pos[:, 2].mean()

    # SimpleITK spacing order: (sx=X, sy=Y, sz=Z)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(origin[0]),  float(origin[1]),  float(origin[2])

    x_mm = ox + x_idx_mean * sx
    y_mm = oy + y_idx_mean * sy
    z_mm = oz + z_idx_mean * sz

    return np.array([x_mm, y_mm, z_mm], dtype=np.float64)


# ---------------------------------------------------------------------------
# All voxel positions in physical space (for histogram plots)
# ---------------------------------------------------------------------------

def voxel_coords_physical(
    mask: np.ndarray,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
    max_voxels: int = 5_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Physical coordinates of all foreground voxels.

    FIX: Explicit column indexing to match centroid_physical axis convention.
    Returns (xs, ys, zs) consistent with centroid_physical output order.
    """
    """Return physical (x, y, z) coordinates for all foreground voxels.

    Randomly sub-samples to ``max_voxels`` when the mask is very large.

    Returns
    -------
    xs, ys, zs : np.ndarray  – 1-D arrays of physical coordinates in mm.
    """
    pos = np.argwhere(mask > 0)
    if pos.size == 0:
        return np.array([]), np.array([]), np.array([])

    if pos.shape[0] > max_voxels:
        idx = np.random.choice(pos.shape[0], max_voxels, replace=False)
        pos = pos[idx]

    # pos columns: [0]=z_idx, [1]=y_idx, [2]=x_idx  (matches centroid_physical)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(origin[0]),  float(origin[1]),  float(origin[2])

    xs = ox + pos[:, 2] * sx   # x_idx column
    ys = oy + pos[:, 1] * sy   # y_idx column
    zs = oz + pos[:, 0] * sz   # z_idx column
    return xs, ys, zs

# ADD this function before compute_displacements:

def validate_spacing_consistency(
    spacings: Dict[str, Tuple[float, ...]],
    tolerance_mm: float = 0.1,
) -> None:
    """
    Assert all spacings are consistent before computing displacements.

    If volumes were correctly resampled in preprocessing, all spacings
    should be identical. If not, displacement mm values are meaningless.

    Raises ValueError if any spacing differs by more than tolerance_mm.
    """
    reference_name = next(iter(spacings))
    ref_sp = np.array(spacings[reference_name], dtype=np.float64)

    for name, sp in spacings.items():
        diff = np.abs(np.array(sp, dtype=np.float64) - ref_sp)
        if diff.max() > tolerance_mm:
            raise ValueError(
                f"Spacing inconsistency detected!\n"
                f"  Reference ({reference_name}): {ref_sp}\n"
                f"  {name}: {sp}\n"
                f"  Max diff: {diff.max():.3f} mm (tolerance: {tolerance_mm} mm)\n"
                f"  → Displacement values will be physically incorrect.\n"
                f"  → Re-run preprocessing with normalize_slice_spacing() first."
            )
    print(f"  Spacing validation passed: all datasets consistent within {tolerance_mm} mm")


# ADD after validate_spacing_consistency:

def centroid_per_slice(
    mask: np.ndarray,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
) -> Dict[int, Optional[np.ndarray]]:
    """
    Compute stent centroid for each axial slice independently.

    This is essential for longitudinal analysis — the global centroid averages
    out regional displacement, missing focal migration (e.g., proximal neck
    displacement vs. distal limb migration). Per-slice centroids reveal where
    along the aorta the stent is actually moving.

    Returns
    -------
    dict[z_index → np.ndarray([x_mm, y_mm, z_mm]) or None]
    """
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(origin[0]),  float(origin[1]),  float(origin[2])

    result = {}
    for z in range(mask.shape[0]):
        sl = mask[z]
        pos = np.argwhere(sl > 0)
        if pos.size == 0:
            result[z] = None
        else:
            y_mean = pos[:, 0].mean()
            x_mean = pos[:, 1].mean()
            result[z] = np.array([
                ox + x_mean * sx,
                oy + y_mean * sy,
                oz + z * sz,
            ], dtype=np.float64)
    return result


# ---------------------------------------------------------------------------
# Pairwise displacement vectors
# ---------------------------------------------------------------------------

def compute_displacements(
    centroids: Dict[str, Optional[np.ndarray]],
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[Tuple[str, str], Optional[np.ndarray]]:
    """Compute displacement vectors between time-point centroids.

    displacement(a → b) = centroid_b − centroid_a

    Parameters
    ----------
    centroids : dict[name → np.ndarray or None]
        Physical centroid per dataset.
    pairs : list of (name_a, name_b) or None
        If None, all ordered pairs are used.

    Returns
    -------
    dict[(a, b) → np.ndarray([Δx, Δy, Δz]) or None]
    """
    names = list(centroids.keys())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    displacements: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    for a, b in pairs:
        ca, cb = centroids.get(a), centroids.get(b)
        if ca is None or cb is None:
            displacements[(a, b)] = None
        else:
            displacements[(a, b)] = (cb - ca).astype(np.float64)

    return displacements


# ---------------------------------------------------------------------------
# Displacement statistics
# ---------------------------------------------------------------------------

def displacement_statistics(
    displacements: Dict[Tuple[str, str], Optional[np.ndarray]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute per-pair displacement statistics.

    For each displacement vector d = [Δx, Δy, Δz]:
      - magnitude  = ‖d‖  (Euclidean distance in mm)
      - dx, dy, dz = individual components

    Parameters
    ----------
    displacements : dict[(a, b) → np.ndarray or None]

    Returns
    -------
    dict[(a, b) → {"dx_mm", "dy_mm", "dz_mm", "magnitude_mm"}]
    """
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for pair, vec in displacements.items():
        if vec is None:
            stats[pair] = {
                "dx_mm": float("nan"),
                "dy_mm": float("nan"),
                "dz_mm": float("nan"),
                "magnitude_mm": float("nan"),
            }
        else:
            stats[pair] = {
                "dx_mm":        float(vec[0]),
                "dy_mm":        float(vec[1]),
                "dz_mm":        float(vec[2]),
                "magnitude_mm": float(np.linalg.norm(vec)),
            }
    return stats


# ---------------------------------------------------------------------------
# Multi-model centroid extraction + displacement
# ---------------------------------------------------------------------------

def compute_all_displacements(
    preds_by_model: Dict[str, Dict[str, np.ndarray]],
    spacings: Dict[str, Tuple[float, ...]],
    origins:  Dict[str, Tuple[float, ...]],
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[
    Dict[str, Dict[str, Optional[np.ndarray]]],   # centroids_by_model
    Dict[str, Dict[Tuple[str, str], Optional[np.ndarray]]],  # displacements_by_model
    Dict[str, Dict[Tuple[str, str], Dict[str, float]]],      # stats_by_model
]:
    """Compute centroids, displacements, and stats for all models.

    Returns
    -------
    centroids_by_model, displacements_by_model, stats_by_model
    """
    centroids_by_model:     Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    displacements_by_model: Dict[str, Dict] = {}
    stats_by_model:         Dict[str, Dict] = {}

    for mname, pred_by_ds in preds_by_model.items():
        cent = {}
        for ds_name, pred_vol in pred_by_ds.items():
            c = centroid_physical(pred_vol, spacings[ds_name], origins[ds_name])
            cent[ds_name] = c
            if c is not None:
                print(
                    f"  {mname} {ds_name}: centroid "
                    f"(x={c[0]:.1f}, y={c[1]:.1f}, z={c[2]:.1f}) mm",
                    flush=True,
                )
        centroids_by_model[mname] = cent

        disp = compute_displacements(cent, pairs=pairs)
        displacements_by_model[mname] = disp

        st = displacement_statistics(disp)
        stats_by_model[mname] = st
        for (a, b), s in st.items():
            print(
                f"  {mname} {a}→{b}: "
                f"Δx={s['dx_mm']:.2f} Δy={s['dy_mm']:.2f} "
                f"Δz={s['dz_mm']:.2f} |d|={s['magnitude_mm']:.2f} mm",
                flush=True,
            )

    return centroids_by_model, displacements_by_model, stats_by_model
