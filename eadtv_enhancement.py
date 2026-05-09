"""
eadtv_enhancement.py
====================
Phase 1 – Enhancement Validation

Implements Edge-Aware Discrete Total Variation (EADTV)-style denoising on
pre-processed CT volumes using the Chambolle algorithm (via scikit-image).
TV regularisation preserves vessel/stent edges while suppressing noise.

Also computes image quality metrics (MSE, PSNR, SSIM) to validate that
enhancement improves image quality without losing diagnostic information.
"""

import numpy as np
from typing import Dict, Tuple

from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import (
    structural_similarity as ssim_metric,
    peak_signal_noise_ratio as psnr_metric,
    mean_squared_error as mse_metric,
)


# ---------------------------------------------------------------------------
# Slice-level TV denoising
# ---------------------------------------------------------------------------

def denoise_slice(
    slice_2d: np.ndarray,
    weight: float = 0.08,
    max_num_iter: int = 30,
) -> np.ndarray:
    """Apply Chambolle Total Variation denoising to a single 2-D slice.

    The slice must be in [0, 1] (float32/float64).  Values outside this
    range are clipped before denoising to satisfy skimage's assumptions.

    Parameters
    ----------
    slice_2d : np.ndarray, shape (H, W)
        Normalised CT slice.
    weight : float
        Regularisation weight (higher → smoother output).
    max_num_iter : int
        Maximum number of Chambolle iterations.

    Returns
    -------
    np.ndarray, float32, same shape as input.
    """
    s = np.clip(slice_2d.astype(np.float64), 0.0, 1.0)
    denoised = denoise_tv_chambolle(s, weight=weight, max_num_iter=max_num_iter)
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Volume-level TV denoising
# ---------------------------------------------------------------------------

def enhance_volume_tv(
    volume: np.ndarray,
    metal_mask: np.ndarray = None,
    weight: float = 0.05,
    max_iter: int = 20,
) -> np.ndarray:
    """
    Slice-by-slice TV denoising with metal region exclusion.

    Metal/stent voxels are excluded from denoising and their original
    intensities are restored after processing. This prevents TV from
    treating stent wire edges as 'noise' and smoothing them away.

    Parameters
    ----------
    volume : np.ndarray (Z, H, W) float32
        Aorta-windowed, normalised volume (NOT the stent channel).
        The stent channel should NEVER be TV-denoised.
    metal_mask : np.ndarray (Z, H, W) bool, optional
        True where voxels are metal. If None, no exclusion is applied.
    weight : float
        TV weight — reduced from 0.08 to 0.05 to preserve vessel edges.
    rray, float32, same shape as input.
    """
    out = np.empty_like(volume, dtype=np.float32)

    for z in range(volume.shape[0]):
        sl         = volume[z].copy()
        sl_orig    = sl.copy()   # keep original for metal restoration

        # Normalise slice to [0,1] for Chambolle
        sl_min, sl_max = float(sl.min()), float(sl.max())
        rng = sl_max - sl_min
        sl_norm = (sl - sl_min) / rng if rng > 1e-8 else sl.copy()

        # --- Metal exclusion: replace metal voxels with local median ---
        # so TV does not mistake the hard stent edge for noise to remove
        if metal_mask is not None:
            metal_z = metal_mask[z]
            if metal_z.any():
                from scipy.ndimage import median_filter
                smoothed = median_filter(sl_norm, size=3)
                sl_norm[metal_z] = smoothed[metal_z]

        denoised = denoise_slice(sl_norm, weight=weight, max_num_iter=max_iter)

        # Rescale back to original intensity range
        if rng > 1e-8:
            denoised = denoised * rng + sl_min

        # Restore exact original values in metal regions
        if metal_mask is not None and metal_mask[z].any():
            denoised[metal_mask[z]] = sl_orig[metal_mask[z]]

        out[z] = denoised.astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Enhancement quality metrics
# ---------------------------------------------------------------------------

def enhancement_metrics(
    original: np.ndarray,
    enhanced: np.ndarray,
) -> Dict[str, float]:
    """Compute image quality metrics between original and enhanced volumes.

    Parameters
    ----------
    original : np.ndarray, float
        Volume before enhancement (reference).
    enhanced : np.ndarray, float
        Volume after enhancement.

    Returns
    -------
    dict with keys ``"MSE"``, ``"PSNR"``, ``"SSIM"``.
    """
    orig = original.astype(np.float64)
    enh  = enhanced.astype(np.float64)
    data_range = max(float(orig.max() - orig.min()), 1e-8)

    # Mean Squared Error
    mse_val = float(np.mean((orig - enh) ** 2))

    # Peak Signal-to-Noise Ratio
    try:
        psnr_val = float(psnr_metric(orig, enh, data_range=data_range))
    except Exception:
        psnr_val = 0.0

    # Structural Similarity Index (averaged over slices for efficiency)
    try:
        ssim_vals = [
            float(ssim_metric(orig[z], enh[z], data_range=data_range))
            for z in range(orig.shape[0])
        ]
        ssim_val = float(np.mean(ssim_vals))
    except Exception:
        ssim_val = 0.0

    return {"MSE": mse_val, "PSNR": psnr_val, "SSIM": ssim_val}


# ---------------------------------------------------------------------------
# Batch enhancement
# ---------------------------------------------------------------------------

# REPLACE enhance_all_datasets signature and body WITH:

def enhance_all_datasets(
    aorta_volumes: Dict[str, np.ndarray],   # renamed from pre_volumes
    metal_masks:   Dict[str, np.ndarray],   # NEW — from preprocessing
    weight: float = 0.05,
    max_iter: int = 20,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Enhance only the aorta channel of all datasets.
    The stent channel is NEVER TV-denoised — pass it through unchanged.
    """
    enhanced_volumes    = {}
    enhancement_results = {}

    for name, vol in aorta_volumes.items():
        print(f"  Enhancing {name} (weight={weight}, iters={max_iter}) …",
              flush=True)
        mask = metal_masks.get(name, None)
        enh  = enhance_volume_tv(vol, metal_mask=mask,
                                 weight=weight, max_iter=max_iter)
        enhanced_volumes[name]    = enh
        enhancement_results[name] = enhancement_metrics(vol, enh)
        m = enhancement_results[name]
        print(
            f"    MSE={m['MSE']:.6f}  PSNR={m['PSNR']:.2f} dB  "
            f"SSIM={m['SSIM']:.4f}",
            flush=True,
        )

    return enhanced_volumes, enhancement_results