"""
mask_generation.py
==================
Dual-stage automatic mask generation using HU thresholding + morphology.

Stage A – Aorta mask
  Threshold raw HU volume in [HU_MIN_AORTA, HU_MAX_AORTA], then:
    1. Per-slice morphological closing  (fill small intra-luminal gaps)
    2. Per-slice binary fill_holes      (close any residual hollow)
    3. Per-slice CC area filter         (remove lung, fat, small vessels)
    4. 3D binary closing along Z        (bridges slice-to-slice gaps → no staircase)
    5. 3D binary fill_holes             (removes internal cavities)
    6. 3D CCA                           (keeps largest anatomically plausible component)

Stage B – Stent mask
  Inside the dilated aorta ROI, threshold at HU ≥ HU_MIN_STENT (metal),
  then apply 3D CCA to remove isolated noise specks.

All outputs are uint8 binary masks with the same (Z, Y, X) shape as input.
Compatible with centerline, displacement, evaluation, and report modules.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _largest_cc_3d(binary_vol: np.ndarray) -> np.ndarray:
    """Return only the largest 3-D connected component of a binary volume."""
    labeled, n = sk_label(binary_vol, return_num=True)
    if n == 0:
        return binary_vol.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0          # ignore background
    largest = int(counts.argmax())
    return (labeled == largest).astype(np.uint8)


def _keep_large_ccs_2d(
    binary_slice: np.ndarray,
    min_area: int = 200,
    max_area: int = 25_000,
) -> np.ndarray:
    """Keep connected components whose area is in [min_area, max_area]."""
    labeled, _ = sk_label(binary_slice, return_num=True)
    out = np.zeros_like(binary_slice, dtype=np.uint8)
    for i in range(1, int(labeled.max()) + 1):
        area = int((labeled == i).sum())
        if min_area <= area <= max_area:
            out[labeled == i] = 1
    return out


def _ball(r: int) -> np.ndarray:
    """Spherical binary structuring element of radius r."""
    d = 2 * r + 1
    z, y, x = np.mgrid[0:d, 0:d, 0:d]
    return ((z - r)**2 + (y - r)**2 + (x - r)**2) <= r**2


def _zhat_struct(z_radius: int = 3, xy_radius: int = 1) -> np.ndarray:
    """Z-elongated structuring element for closing along the axial direction.

    Using a structuring element that is tall in Z relative to XY forces
    the morphological closing to bridge gaps between adjacent slices first,
    which is exactly what produces the staircase artefact.
    """
    dz = 2 * z_radius + 1
    dy = 2 * xy_radius + 1
    dx = 2 * xy_radius + 1
    cz, cy, cx = z_radius, xy_radius, xy_radius
    z, y, x = np.mgrid[0:dz, 0:dy, 0:dx]
    return (
        ((z - cz) / z_radius)**2
        + ((y - cy) / max(xy_radius, 1))**2
        + ((x - cx) / max(xy_radius, 1))**2
    ) <= 1.0


# ---------------------------------------------------------------------------
# Aorta mask generation
# ---------------------------------------------------------------------------

def generate_aorta_mask(
    volume_hu: np.ndarray,
    hu_min: float = 150.0,
    hu_max: float = 550.0,
    min_area: int = 200,
    max_area: int = 25_000,
    closing_iters: int = 3,
    z_close_radius: int = 2,
) -> np.ndarray:
    """Generate a 3-D binary aorta mask via HU thresholding + 3D refinement.

    Pipeline
    --------
    Per-slice (Z-axis loop):
      1. HU threshold [hu_min, hu_max]
      2. 2-D binary closing (fill intra-luminal gaps)
      3. 2-D fill_holes (close hollow regions)
      4. Area filter (removes lung, fat, tiny vessels)

    3-D refinement (after all slices):
      5. Z-elongated binary closing → bridges slice-to-slice gaps, the main
         source of staircase/jagged artefacts in 3D surface rendering
      6. 3-D fill_holes → closes internal cavities left by threshold drop-outs
      7. Keep only largest 3-D connected component (aorta is one tube)

    Parameters
    ----------
    volume_hu : np.ndarray, shape (Z, Y, X), float32
        Raw Hounsfield-Unit volume.
    hu_min, hu_max : float
        HU window for blood-pool segmentation.
    min_area, max_area : int
        Slice-level area filter (voxels).
    closing_iters : int
        Iterations for 2-D binary closing.
    z_close_radius : int
        Z-radius of the 3-D Z-elongated closing structuring element.
        Larger = more inter-slice bridging (default 2 ≈ 3–4 mm at 1.5 mm slices).

    Returns
    -------
    np.ndarray, uint8, shape (Z, Y, X)  – binary aorta mask.
    """
    struct2 = ndi.generate_binary_structure(2, 1)
    mask = np.zeros(volume_hu.shape, dtype=np.uint8)

    # ── Step 1-4: per-slice processing ────────────────────────────────────────
    for z in range(volume_hu.shape[0]):
        sl = ((volume_hu[z] >= hu_min) & (volume_hu[z] <= hu_max)).astype(np.uint8)
        sl = ndi.binary_closing(sl, structure=struct2, iterations=closing_iters).astype(np.uint8)
        sl = ndi.binary_fill_holes(sl).astype(np.uint8)
        sl = _keep_large_ccs_2d(sl, min_area=min_area, max_area=max_area)
        mask[z] = sl

    # ── Step 5: 3-D inter-slice closing (eliminates staircase) ────────────────
    if z_close_radius > 0:
        se3 = _zhat_struct(z_radius=z_close_radius, xy_radius=1)
        mask = ndi.binary_closing(mask.astype(bool), structure=se3).astype(np.uint8)

    # ── Step 6: 3-D fill holes ────────────────────────────────────────────────
    mask = ndi.binary_fill_holes(mask.astype(bool)).astype(np.uint8)

    # ── Step 7: keep only largest 3-D CC ─────────────────────────────────────
    if mask.any():
        mask = _largest_cc_3d(mask)

    return mask


# ---------------------------------------------------------------------------
# Stent mask generation
# ---------------------------------------------------------------------------

def generate_stent_mask(
    volume_hu: np.ndarray,
    aorta_mask: np.ndarray,
    hu_min_stent: float = 1000.0,
    dilate_iters: int = 3,          # FIXED: was 8 — now 3 to avoid bone inclusion
    min_cc_voxels: int = 5,
    closing_radius: int = 1,        # NEW: morphological closing to bridge strut gaps
) -> np.ndarray:
    """Generate a 3-D binary stent (metal) mask inside the aortic ROI.

    Parameters
    ----------
    volume_hu : np.ndarray
        Raw HU volume.
    aorta_mask : np.ndarray, uint8
        Binary aorta mask (from ``generate_aorta_mask``).
    hu_min_stent : float
        HU threshold for metallic structures (≥ 1 000 HU).
    dilate_iters : int
        Extra dilation of the aorta mask before applying it as an ROI.
        Ensures that stent wires touching the vessel wall are included.
    min_cc_voxels : int
        3-D connected components smaller than this are discarded as noise.

    Returns
    -------
    np.ndarray, uint8, shape (Z, Y, X) – binary stent mask.
    """
    struct3 = ndi.generate_binary_structure(3, 1)

    # Step 1: Dilate aorta mask to capture periwall stent struts
    roi = ndi.binary_dilation(
        aorta_mask.astype(bool), structure=struct3, iterations=dilate_iters
    ).astype(np.uint8)

    # Step 2: High-HU threshold inside aorta ROI
    stent_raw = ((volume_hu >= hu_min_stent) & (roi > 0)).astype(np.uint8)
    if not stent_raw.any():
            return stent_raw
    
    # Step 3: NEW — morphological closing to bridge inter-strut gaps
    # Stent wires are 1-2 voxels wide; partial volume drops HU below
    # threshold at boundaries, fragmenting the mask. Closing reconnects.
    se_close = _ball(closing_radius)
    stent_closed = ndi.binary_closing(
        stent_raw.astype(bool), structure=se_close
    ).astype(np.uint8)

    # Step 4: CCA — now applied AFTER closing, so struts survive
    labeled3d, n = sk_label(stent_closed, return_num=True)
    stent_out = np.zeros_like(stent_raw, dtype=np.uint8)
    for i in range(1, n + 1):
        cc = labeled3d == i
        if int(cc.sum()) >= min_cc_voxels:
            stent_out[cc] = 1

    return stent_out

# ---------------------------------------------------------------------------
# Bounding-box extraction (ROI crops for Stage B dataset)
# ---------------------------------------------------------------------------

def get_bbox_2d_per_slice(
    aorta_mask: np.ndarray,
    padding: int = 20,
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Compute a 2-D bounding box for the aorta on each axial slice.

    Parameters
    ----------
    aorta_mask : np.ndarray, shape (Z, Y, X)
        Binary aorta mask.
    padding : int
        Extra padding (pixels) around the tight bounding box.

    Returns
    -------
    list of (y_min, y_max, x_min, x_max) or None (if slice has no aorta).
        Bounding box per axial slice with stent-aware padding.

    Padding increased from 8→20 pixels because:
    - Stent struts sit on/outside vessel wall
    - HU thresholding slightly under-segments wall boundary
    - 20px ≈ 7–10mm depending on pixel spacing — safely captures all struts

    """
    H, W = aorta_mask.shape[1], aorta_mask.shape[2]
    bboxes = []
    for z in range(aorta_mask.shape[0]):
        pos = np.argwhere(aorta_mask[z] > 0)
        if pos.size == 0:
            bboxes.append(None)
        else:
            y0, x0 = pos.min(axis=0)
            y1, x1 = pos.max(axis=0)
            bboxes.append((
                max(0, y0 - padding),
                min(H, y1 + padding + 1),
                max(0, x0 - padding),
                min(W, x1 + padding + 1),
            ))
    return bboxes


# ---------------------------------------------------------------------------
# Batch mask generation
# ---------------------------------------------------------------------------

def generate_all_masks(
    raw_volumes: Dict[str, np.ndarray],
    hu_min_aorta: float = 150.0,
    hu_max_aorta: float = 550.0,
    hu_min_stent: float = 1000.0,
    stent_dilate_iters: int = 3,         # FIXED: was 8
    stent_closing_radius: int = 1,       # NEW
    aorta_min_area: int = 200,
    aorta_max_area: int = 25_000,
    stent_min_cc_voxels: int = 5,
    aorta_z_close_radius: int = 3,       # INCREASED: was default 2 → 3 for smoother surface
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate aorta and stent masks for all datasets.

    Returns
    -------
    aorta_masks : dict[name → np.ndarray]
    stent_masks : dict[name → np.ndarray]
    """
    aorta_masks: Dict[str, np.ndarray] = {}
    stent_masks: Dict[str, np.ndarray] = {}

    for name, vol in raw_volumes.items():
        print(f"  Generating masks for {name} …", flush=True)

        am = generate_aorta_mask(
            vol,
            hu_min=hu_min_aorta, hu_max=hu_max_aorta,
            min_area=aorta_min_area, max_area=aorta_max_area,
            z_close_radius=aorta_z_close_radius,   # larger Z closing → smoother 3D surface

        )
        sm = generate_stent_mask(
            vol, am,
            hu_min_stent=hu_min_stent,
            dilate_iters=stent_dilate_iters,
            min_cc_voxels=stent_min_cc_voxels,
            closing_radius=stent_closing_radius,    # NEW
        )
        aorta_masks[name] = am
        stent_masks[name] = sm

        # Sanity check: warn if stent mask is suspiciously large
        stent_pct = 100.0 * sm.sum() / max(am.sum(), 1)
        if stent_pct > 15.0:
            print(
                f"    ⚠ WARNING: stent={sm.sum():,} voxels ({stent_pct:.1f}% of aorta) "
                f"— possible bone/calcification contamination. Reduce dilate_iters.",
                flush=True,
            )
        else:
            print(
                f"    → aorta={am.sum():,}  stent={sm.sum():,} ({stent_pct:.1f}%)",
                flush=True,
            )

    return aorta_masks, stent_masks
