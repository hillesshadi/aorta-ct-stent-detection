"""
preprocessing.py
================
Phase 1: Data Preparation & Preprocessing

Handles:
  - Loading DICOM series (flat directory, no sub-folders) via SimpleITK
  - Hounsfield Unit (HU) windowing and clipping
  - Intensity normalisation (min-max or Z-score)
  - Convenience wrappers to process all three time-point datasets at once

All volumes are returned as float32 NumPy arrays with axis order (Z, Y, X).
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, Tuple, Optional

try:
    import pydicom  # optional; used only for metadata inspection
except ImportError:
    pydicom = None


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def load_dicom_series(
    folder: str,
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """Load a flat DICOM directory into a 3-D Hounsfield-Unit volume.

    Parameters
    ----------
    folder : str
        Path to a directory containing .dcm files (no sub-folders required).

    Returns
    -------
    volume : np.ndarray, shape (Z, Y, X), dtype float32
        Raw HU values (includes negative air, ~-1000 HU).
    spacing : tuple of float
        Voxel spacing (sx, sy, sz) in millimetres.
    origin : tuple of float
        Physical origin (ox, oy, oz) in the patient coordinate system (mm).
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"DICOM folder not found: {folder}")

    reader = sitk.ImageSeriesReader()
    # GetGDCMSeriesFileNames handles flat directories natively
    series_ids = reader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        raise RuntimeError(
            f"No valid DICOM series detected in '{folder}'. "
            "Ensure the folder contains .dcm files."
        )

    # Use the first (and usually only) series
    file_names = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()   # expose per-slice tags
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image).astype(np.float32)   # (Z, Y, X)
    spacing = image.GetSpacing()   # (sx, sy, sz)  – metres in ITK convention → mm
    origin  = image.GetOrigin()    # (ox, oy, oz)

    return volume, spacing, origin


# ---------------------------------------------------------------------------
# HU Windowing
# ---------------------------------------------------------------------------

def hu_window_and_clip(
    volume: np.ndarray,
    center: float = 300.0,
    width: float = 1500.0,
) -> np.ndarray:
    """Apply a Hounsfield-Unit display window and clip outside values.

    Parameters
    ----------
    volume : np.ndarray
        Raw HU volume (float32).
    center : float
        Window centre (Level), e.g. 300 HU for aorta/stent.
    width : float
        Window width, e.g. 1500 HU.

    Returns
    -------
    np.ndarray
        Clipped float32 volume in the range [center - width/2, center + width/2].
    """
    low  = center - width / 2.0
    high = center + width / 2.0
    return np.clip(volume, low, high).astype(np.float32)


# ---------------------------------------------------------------------------
# Intensity Normalisation
# ---------------------------------------------------------------------------
def preprocess_dual_channel(
    volume_hu: np.ndarray,
    aorta_center: float = 350.0,
    aorta_width: float  = 600.0,
    stent_center: float = 1500.0,
    stent_width: float  = 2000.0,
    norm_mode: str = "minmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce two independent normalised channels from a raw HU volume.

    Returns
    -------
    aorta_channel : np.ndarray (Z, H, W) float32, normalised [0,1]
        Windowed for soft tissue / blood pool — input to aorta nnUNet.
    stent_channel : np.ndarray (Z, H, W) float32, normalised [0,1]
        Windowed for metal / stent struts — input to stent nnUNet.

    Why two channels?
        A single window cannot represent both structures. At center=300/width=1500,
        metal above 1050 HU is clipped flat — the stent becomes invisible.
        The stent channel uses center=1500/width=2000, giving [500, 2500] HU range
        which fully captures stent strut density.
    """
    aorta_ch = preprocess_volume(volume_hu, aorta_center, aorta_width, norm_mode)
    stent_ch  = preprocess_volume(volume_hu, stent_center, stent_width, norm_mode)
    return aorta_ch, stent_ch
def extract_metal_mask(
    volume_hu: np.ndarray,
    hu_threshold: float = 800.0,
) -> np.ndarray:
    """
    Binary mask of metal/stent voxels from raw HU volume.

    This mask is used by EADTV to EXCLUDE metal regions from TV denoising,
    preventing the denoiser from erasing stent strut edges.

    Parameters
    ----------
    volume_hu : np.ndarray (Z, H, W)
        Raw HU volume (before any windowing).
    hu_threshold : float
        Voxels above this value are classified as metal.
        800 HU adds a safety margin below the 1000 HU stent threshold.

    Returns
    -------
    np.ndarray, bool, same shape as input.
    """
    return volume_hu > hu_threshold
def normalize_slice_spacing(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing_z: float = 1.0,
) -> np.ndarray:
    """
    Resample volume along Z axis to achieve uniform 1 mm slice spacing.

    Your 3 datasets have different numbers of slices (1198, 1442, 696)
    which implies different slice thicknesses. Without Z-resampling,
    displacement measurements in mm will be wrong because voxel-to-mm
    conversion assumes uniform spacing.

    Parameters
    ----------
    volume : np.ndarray (Z, H, W)
        Preprocessed volume.
    spacing : tuple (sx, sy, sz)
        Voxel spacing in mm from load_dicom_series().
    target_spacing_z : float
        Target slice spacing in mm (default 1.0 mm).

    Returns
    -------
    np.ndarray float32 with resampled Z dimension.
    """
    from scipy.ndimage import zoom
    sz = spacing[2]   # current Z spacing in mm
    if abs(sz - target_spacing_z) < 0.01:
        return volume   # already at target, skip resampling
    zoom_factor = sz / target_spacing_z
    return zoom(volume, (zoom_factor, 1.0, 1.0), order=1).astype(np.float32)

def minmax_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale a volume to the range [0, 1].

    Parameters
    ----------
    volume : np.ndarray
        Input float array (any range).
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    np.ndarray, dtype float32, values in [0, 1].
    """
    low  = float(volume.min())
    high = float(volume.max())
    return ((volume - low) / (high - low + eps)).astype(np.float32)


def zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance normalisation.

    Parameters
    ----------
    volume : np.ndarray
        Input float array.

    Returns
    -------
    np.ndarray, dtype float32.
    """
    mu  = float(volume.mean())
    std = float(volume.std())
    return ((volume - mu) / (std + eps)).astype(np.float32)


def normalize(volume: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """Dispatch to min-max or Z-score normalisation.

    Parameters
    ----------
    volume : np.ndarray
        Windowed HU volume.
    mode : str
        ``"minmax"`` (default) or ``"zscore"``.
    """
    if mode == "zscore":
        return zscore_normalize(volume)
    return minmax_normalize(volume)


# ---------------------------------------------------------------------------
# Combined preprocessing step
# ---------------------------------------------------------------------------

def preprocess_volume(
    volume_hu: np.ndarray,
    center: float = 350.0,
    width: float = 600.0,
    norm_mode: str = "minmax",
) -> np.ndarray:
    """Window → clip → normalise a single HU volume.

    Returns
    -------
    np.ndarray, float32, normalised to [0, 1] (minmax) or approx [-3, 3] (zscore).
    """
    windowed = hu_window_and_clip(volume_hu, center, width)
    return normalize(windowed, norm_mode)


# ---------------------------------------------------------------------------
# Batch preprocessing for all datasets
# ---------------------------------------------------------------------------

def preprocess_all_datasets(
    datasets: Dict[str, str],
    aorta_center: float = 350.0,
    aorta_width:  float = 600.0,
    stent_center: float = 1500.0,
    stent_width:  float = 2000.0,
    norm_mode: str = "minmax",
    target_spacing_z: float = 1.0,
):
    raw_volumes   = {}
    aorta_volumes = {}   # soft-tissue channel
    stent_volumes = {}   # metal channel
    metal_masks   = {}   # boolean metal mask (for EADTV exclusion)
    spacings      = {}
    origins       = {}
    """Load and preprocess all time-point datasets.

    Parameters
    ----------
    datasets : dict
        Mapping from dataset name to DICOM folder path, e.g.::

            {"aorta_p1_ds1": "/path/ds1", "aorta_p1_ds2": "/path/ds2", ...}

    center, width : float
        HU window parameters.
    norm_mode : str
        ``"minmax"`` or ``"zscore"``.

    Returns
    -------
    raw_volumes : dict[name → np.ndarray]
        Original float32 HU arrays, shape (Z, Y, X).
    pre_volumes : dict[name → np.ndarray]
        Windowed + normalised arrays.
    spacings : dict[name → tuple]
        Per-dataset voxel spacing in mm.
    origins : dict[name → tuple]
        Per-dataset physical origin in mm.
    """

    for name, folder in datasets.items():
            print(f"  Loading {name} …", flush=True)
            vol, sp, orig = load_dicom_series(folder)
            raw_volumes[name] = vol
            spacings[name]    = sp
            origins[name]     = orig

            # Dual-channel preprocessing
            aorta_ch, stent_ch = preprocess_dual_channel(
                vol, aorta_center, aorta_width,
                    stent_center, stent_width, norm_mode
            )

            # Metal mask from raw HU (before any windowing)
            metal_masks[name] = extract_metal_mask(vol, hu_threshold=800.0)

            # Z-spacing normalisation (critical for displacement accuracy)
            aorta_ch = normalize_slice_spacing(aorta_ch, sp, target_spacing_z)
            stent_ch = normalize_slice_spacing(stent_ch, sp, target_spacing_z)

            aorta_volumes[name] = aorta_ch
            stent_volumes[name] = stent_ch

            print(
                f"    shape={vol.shape} → resampled_z={aorta_ch.shape[0]}  "
                f"metal_voxels={metal_masks[name].sum()}",
                flush=True,
            )

    return raw_volumes, aorta_volumes, stent_volumes, metal_masks, spacings, origins
