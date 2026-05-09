"""
postprocessing.py
=================
Phase 3 – Post-Processing & Refinement

Applies morphological cleaning and connected-component analysis to convert
raw segmentation masks into clean, clinically plausible binary stent masks.

Pipeline (applied in order)
---------------------------
1. Binary morphological **closing**  (bridge small gaps in stent struts)
2. Binary morphological **opening**  (remove isolated noise pixels)
3. Connected Component Analysis      (remove components too small to be a stent)
4. Optional tubularity filter        (reject blobs that are far too round/compact)

Pre-3D-mesh helpers (called by visualization.py)
-------------------------------------------------
5. ``gaussian_smooth_volume``  – anisotropic 3-D Gaussian smoothing of the
   float field before marching cubes.  Using a larger sigma along Z eliminates
   the staircase (slice-to-slice) artefacts that arise from voxel anisotropy.
6. ``smooth_mask_for_3d``      – convenience wrapper: converts sigma from mm to
   voxels using the scan spacing, calls ``gaussian_smooth_volume``, and
   returns the smoothed *float* field (NOT re-thresholded) so marching cubes
   sees a proper gradient isocontour rather than a hard jump.

Binary-domain mask refinement
------------------------------
7. ``refine_aorta_mask``  – final 3-D binary polish for the aorta mask:
   Z-elongated closing bridges any remaining inter-slice gaps then fill_holes
   removes internal cavities.  Output is uint8, same shape as input.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label, regionprops
from typing import Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Pre-3D-mesh volume smoothing  (eliminates axial / staircase artefacts)
# ---------------------------------------------------------------------------

def gaussian_smooth_volume(
    mask: np.ndarray,
    sigma: Union[float, Tuple[float, float, float]] = (1.5, 1.0, 1.0),
) -> np.ndarray:
    """Apply anisotropic 3-D Gaussian blurring to a binary mask volume.

    The result is a *float* field in [0, 1] suitable for use as the scalar
    field fed to marching cubes (iso-level 0.5).  This is intentionally NOT
    re-thresholded so that the mesh extractor sees smooth gradients.

    Parameters
    ----------
    mask : np.ndarray, shape (Z, Y, X)
        Binary input mask (bool or uint8).
    sigma : float or 3-tuple
        Gaussian sigma in voxels per axis ``(σ_z, σ_y, σ_x)``.
        A larger σ_z relative to σ_xy corrects slice-to-slice discontinuities.

    Returns
    -------
    np.ndarray, float32, same shape as input  – values in [0, 1].
    """
    float_mask = mask.astype(np.float32)
    smoothed   = ndi.gaussian_filter(float_mask, sigma=sigma)
    # Clip for safety (Gaussian may push slightly outside [0,1] for uint masks)
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def smooth_mask_for_3d(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_mm: float = 1.2,
    axial_boost: float = 1.5,
) -> np.ndarray:
    """Convert a binary mask to a smooth float field ready for marching cubes.

    Computes per-axis sigma in *voxels* from a target smoothing radius in mm,
    then applies an **anisotropic** Gaussian that boosts the Z-axis sigma by
    ``axial_boost`` to compensate for the typically larger axial slice spacing
    in CT and eliminate staircase artefacts along the Z direction.

    Parameters
    ----------
    mask : np.ndarray, shape (Z, Y, X)
        Binary mask.
    spacing : 3-tuple of float
        Voxel spacing in mm ``(sz, sy, sx)``  (default: isotropic 1 mm).
    sigma_mm : float
        Target smoothing radius in mm (applied uniformly in XY).  Default 1.2 mm
        is a safe value that removes staircase artefacts without blurring thin
        stent struts.
    axial_boost : float
        Multiplicative factor applied to σ_z.  Default 1.5 compensates for
        typical axial anisotropy (e.g. 1.5–3 mm slice thickness vs ~0.7 mm
        in-plane resolution).

    Returns
    -------
    np.ndarray, float32, shape (Z, Y, X)  – smooth scalar field in [0, 1].
    """
    sz, sy, sx = (float(s) for s in spacing)
    # Convert sigma from mm to voxels per axis
    sig_z = (sigma_mm / sz) * axial_boost
    sig_y = sigma_mm / sy
    sig_x = sigma_mm / sx
    return gaussian_smooth_volume(mask, sigma=(sig_z, sig_y, sig_x))


def refine_aorta_mask(
    mask: np.ndarray,
    z_close_radius: int = 2,
    xy_close_radius: int = 1,
) -> np.ndarray:
    """Final binary-domain 3-D refinement for the aorta mask.

    Applies a Z-elongated morphological closing to bridge any remaining
    inter-slice gaps, followed by a 3-D ``binary_fill_holes`` to close
    internal cavities.  This is the last binary-domain clean-up step before
    the Gaussian pre-smooth + marching-cubes mesh pipeline.

    Parameters
    ----------
    mask : np.ndarray, uint8 / bool, shape (Z, Y, X)
        Binary aorta (or any vessel) mask.
    z_close_radius : int
        Half-height of the Z-elongated closing element.  Each unit spans
        one slice (~1–2 mm), so 2 bridges a ~3-slice gap.
    xy_close_radius : int
        In-plane half-width of the closing element.

    Returns
    -------
    np.ndarray, uint8, same shape as input.
    """
    binary = (mask > 0).astype(bool)

    # Z-elongated ellipsoidal structuring element
    dz = 2 * z_close_radius + 1
    dy = 2 * xy_close_radius + 1
    dx = 2 * xy_close_radius + 1
    gz, gy, gx = np.mgrid[0:dz, 0:dy, 0:dx]
    se = (
        ((gz - z_close_radius)  / z_close_radius)**2
        + ((gy - xy_close_radius) / max(xy_close_radius, 1))**2
        + ((gx - xy_close_radius) / max(xy_close_radius, 1))**2
    ) <= 1.0

    closed = ndi.binary_closing(binary, structure=se)
    filled = ndi.binary_fill_holes(closed)
    return filled.astype(np.uint8)


# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

def morphological_clean(
    mask: np.ndarray,
    closing_radius: int = 2,
    opening_radius: int = 1,
) -> np.ndarray:
    """Apply 3-D morphological closing then opening to a binary mask.

    Parameters
    ----------
    mask : np.ndarray, shape (Z, Y, X)
        Binary input mask (any integer / bool dtype).
    closing_radius : int
        Structuring element radius for closing (fills gaps ≤ 2×radius).
    opening_radius : int
        Structuring element radius for opening (removes blobs ≤ radius).

    Returns
    -------
    np.ndarray, uint8, same shape as input.
    """
    binary = (mask > 0).astype(bool)

    # Build spherical structuring elements
    def _ball(r):
        d = 2 * r + 1
        c = r
        z, y, x = np.mgrid[0:d, 0:d, 0:d]
        return ((z - c)**2 + (y - c)**2 + (x - c)**2) <= r**2

    if closing_radius > 0:
        se = _ball(closing_radius)
        binary = ndi.binary_closing(binary, structure=se)

    if opening_radius > 0:
        se = _ball(opening_radius)
        binary = ndi.binary_opening(binary, structure=se)

    return binary.astype(np.uint8)


# ---------------------------------------------------------------------------
# Connected Component Filtering
# ---------------------------------------------------------------------------

def connected_component_filter(
    mask: np.ndarray,
    min_voxels: int = 20,
) -> np.ndarray:
    """Keep only 3-D connected components with at least ``min_voxels`` voxels.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    min_voxels : int
        Minimum component size (voxels) to retain.

    Returns
    -------
    np.ndarray, uint8.
    """
    labeled, n = sk_label(mask > 0, return_num=True)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_voxels:
            out[comp] = 1
    return out


# ---------------------------------------------------------------------------
# Tubularity / shape filter
# ---------------------------------------------------------------------------

def filter_tubular_structure(
    mask: np.ndarray,
    min_voxels: int = 20,
    min_extent: float = 0.02,
    max_extent: float = 0.65,
) -> np.ndarray:
    """Remove connected components that are too compact (blob-like) to be a stent.

    Extent = foreground_voxels / bounding_box_volume.  A real stent is an
    elongated tubular mesh, so it has a low filling percentage (many voids)
    while blob-like noise has a high extent.  We use a *minimum* extent
    threshold to discard truly tiny, perfectly-filled blobs.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    min_voxels : int
        Minimum component size.
    min_extent : float
        Minimum extent ratio to keep (very low → keep almost everything).

    Returns
    -------
    np.ndarray, uint8.
    """
    labeled, n = sk_label(mask > 0, return_num=True)
    out = np.zeros_like(mask, dtype=np.uint8)

    for prop in regionprops(labeled):
        if prop.area < min_voxels:
            continue
        bbox = prop.bbox  # (z0,y0,x0,z1,y1,x1)
        bb_vol = max(
            (bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2]), 1
        )
        extent = prop.area / bb_vol
        # Keep only tubular-range extent — rejects both tiny noise AND solid blobs
        if min_extent <= extent <= max_extent:
            out[labeled == prop.label] = 1

    return out


# ---------------------------------------------------------------------------
# Full post-processing pipeline
# ---------------------------------------------------------------------------

def postprocess_mask(
    mask: np.ndarray,
    closing_radius: int = 2,
    opening_radius: int = 1,
    min_voxels: int = 20,
    apply_tubularity_filter: bool = True,
) -> np.ndarray:
    """Apply the full post-processing pipeline to a raw segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Raw binary mask (output of shape_filter / thresholded inference).
    closing_radius, opening_radius : int
        Morphological operation radii.
    min_voxels : int
        Minimum CCA component size.
    apply_tubularity_filter : bool
        Whether to apply the tubularity / extent filter.

    Returns
    -------
    np.ndarray, uint8.
    """
    out = morphological_clean(mask, closing_radius, opening_radius)
    out = connected_component_filter(out, min_voxels)
    if apply_tubularity_filter:
        out = filter_tubular_structure(out, min_voxels)
    return out

# ADD after postprocess_mask:

def postprocess_aorta_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Full postprocessing pipeline for the aorta mask.
    Uses larger morphological radii appropriate for a large tubular structure.
    Ends with Gaussian smoothing field for marching cubes.
    """
    # Step 1: Binary morphological clean with aorta-appropriate radii
    out = morphological_clean(mask, closing_radius=3, opening_radius=1)
    # Step 2: Remove small components
    out = connected_component_filter(out, min_voxels=500)
    # Step 3: Final 3D binary refinement (bridges slice gaps)
    out = refine_aorta_mask(out, z_close_radius=3, xy_close_radius=1)
    return out


def postprocess_stent_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Full postprocessing pipeline for the stent mask.

    Key differences from aorta pipeline:
    - closing_radius=1 (NOT 2): stent struts are 1-2 voxels — radius 2 merges them
    - opening_radius=0: no opening — it removes the thin struts entirely
    - min_voxels=5: stent CCs are legitimately tiny
    - tubularity filter WITH max_extent: removes solid calcification blobs
    - NO Gaussian smoothing: stent is visualised as skeleton, not smooth surface
    """
    # Step 1: Minimal closing only — do NOT open stent mask
    out = morphological_clean(mask, closing_radius=1, opening_radius=0)
    # Step 2: Remove truly isolated noise specks
    out = connected_component_filter(out, min_voxels=5)
    # Step 3: Tubularity filter — removes solid blobs (calcification, bone fragments)
    out = filter_tubular_structure(out, min_voxels=5, min_extent=0.02, max_extent=0.65)
    return out