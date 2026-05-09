"""
registration.py
===============
Phase 3 – Inter-Scan Registration

Aligns T2 and T3 datasets to the T1 baseline using SimpleITK rigid
(Euler 3D) registration with Mattes Mutual Information as the similarity
metric.  An optional BSpline deformable step can be enabled via config.

All registration is performed on the *normalised* (preprocessed) volumes
so that the metric is computed on comparable intensity ranges.  The
computed transforms are then used to resample both the image volume AND
all associated masks so that every downstream computation is in the T1
coordinate frame.

Key functions
-------------
- ``rigid_registration``           – register one moving image to a fixed image
- ``deformable_registration``      – optional BSpline refinement
- ``register_all_to_baseline``     – align T2, T3 → T1, return resampled arrays
- ``apply_transform_to_mask``      – resample a binary mask with the same transform
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers: numpy ↔ SimpleITK
# ---------------------------------------------------------------------------

def _to_sitk(
    volume: np.ndarray,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
) -> sitk.Image:
    """Convert a (Z, Y, X) float32 NumPy array to a SimpleITK Image."""
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in origin))
    return img


def _from_sitk(img: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK Image back to a float32 NumPy array (Z, Y, X)."""
    return sitk.GetArrayFromImage(img).astype(np.float32)


# ---------------------------------------------------------------------------
# Rigid registration
# ---------------------------------------------------------------------------

def rigid_registration(
    fixed_vol: np.ndarray,
    moving_vol: np.ndarray,
    fixed_spacing: Tuple[float, ...],
    moving_spacing: Tuple[float, ...],
    fixed_origin: Tuple[float, ...] = (0.0, 0.0, 0.0),
    moving_origin: Tuple[float, ...] = (0.0, 0.0, 0.0),
    num_iterations: int = 200,
    learning_rate: float = 1.0,
    shrink_factors: Tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: Tuple[float, ...] = (2.0, 1.0, 0.0),
    verbose: bool = False,
) -> Tuple[np.ndarray, sitk.Transform]:
    """Register *moving* to *fixed* using a 3-D Euler (rigid) transform.

    Uses Mattes Mutual Information with a multi-resolution gradient-descent
    optimiser — robust for multi-temporal CT scans that may differ in
    contrast phase and patient positioning.

    Parameters
    ----------
    fixed_vol, moving_vol : np.ndarray, shape (Z, Y, X)
        Normalised (windowed + clipped) volumes.
    fixed_spacing, moving_spacing : tuple of float
        Voxel spacings in mm.
    fixed_origin, moving_origin : tuple of float
        Physical origins in mm.
    num_iterations : int
        Optimiser iteration limit per resolution level.
    learning_rate : float
        Gradient-descent step size.
    shrink_factors : tuple of int
        Down-sampling factors per resolution (coarse → fine).
    smoothing_sigmas : tuple of float
        Gaussian smoothing sigmas in voxels per resolution level.
    verbose : bool
        Print iteration callbacks if True.

    Returns
    -------
    registered_vol : np.ndarray, float32
        Moving volume resampled into the fixed image space.
    transform : sitk.Transform
        Optimised Euler3DTransform (can be applied to masks).
    """
    fixed  = _to_sitk(fixed_vol,  fixed_spacing,  fixed_origin)
    moving = _to_sitk(moving_vol, moving_spacing, moving_origin)

    # Initialise transform using image centres of mass
    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Build registration method
    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(init_tx, inPlace=False)

    # Similarity metric: Mattes Mutual Information
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.30)

    # Interpolator
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimiser: gradient descent with adaptive learning rate
    reg.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    reg.SetShrinkFactorsPerLevel(shrinkFactors=list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smoothing_sigmas))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if verbose:
        reg.AddCommand(
            sitk.sitkIterationEvent,
            lambda: print(
                f"    iter {reg.GetOptimizerIteration():4d}  "
                f"metric={reg.GetMetricValue():.6f}",
                flush=True,
            ),
        )

    final_transform = reg.Execute(
        sitk.Cast(fixed,  sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )

    # Resample moving into fixed space
    resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    return _from_sitk(resampled), final_transform


# ---------------------------------------------------------------------------
# Optional: deformable (BSpline) registration
# ---------------------------------------------------------------------------

def deformable_registration(
    fixed_vol: np.ndarray,
    moving_vol: np.ndarray,
    fixed_spacing: Tuple[float, ...],
    moving_spacing: Tuple[float, ...],
    fixed_origin: Tuple[float, ...] = (0.0, 0.0, 0.0),
    moving_origin: Tuple[float, ...] = (0.0, 0.0, 0.0),
    initial_transform: Optional[sitk.Transform] = None,
    mesh_size: Tuple[int, ...] = (8, 8, 8),
    num_iterations: int = 100,
) -> Tuple[np.ndarray, sitk.Transform]:
    """Refine a rigid registration with a BSpline deformable transform.

    Parameters
    ----------
    initial_transform : sitk.Transform or None
        Result of ``rigid_registration``; used as initialisation.
    mesh_size : tuple of int
        BSpline control-point grid per dimension.
    num_iterations : int
        Optimiser iterations.

    Returns
    -------
    registered_vol : np.ndarray
    transform : sitk.BSplineTransform
    """
    fixed  = _to_sitk(fixed_vol,  fixed_spacing,  fixed_origin)
    moving = _to_sitk(moving_vol, moving_spacing, moving_origin)

    # If a rigid transform was provided, pre-warp moving
    if initial_transform is not None:
        moving = sitk.Resample(moving, fixed, initial_transform,
                               sitk.sitkLinear, 0.0, moving.GetPixelID())

    tx_init = sitk.BSplineTransformInitializer(
        image1=fixed,
        transformDomainMeshSize=list(mesh_size),
        order=3,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(tx_init)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    final_tx = reg.Execute(
        sitk.Cast(fixed,  sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )

    resampled = sitk.Resample(
        moving, fixed, final_tx, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return _from_sitk(resampled), final_tx


# ---------------------------------------------------------------------------
# Apply transform to a binary mask
# ---------------------------------------------------------------------------

def apply_transform_to_mask(
    mask: np.ndarray,
    transform: sitk.Transform,
    reference_image: sitk.Image,
) -> np.ndarray:
    """Resample a binary mask with the given transform.

    Nearest-neighbour interpolation preserves binary values.

    Parameters
    ----------
    mask : np.ndarray, shape (Z, Y, X), uint8 or bool
        Binary segmentation mask.
    transform : sitk.Transform
        Rigid or deformable transform from registration.
    reference_image : sitk.Image
        The fixed image (defines output grid).

    Returns
    -------
    np.ndarray, uint8, same spatial extent as reference_image.
    """
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_sitk.CopyInformation(reference_image)

    resampled = sitk.Resample(
        mask_sitk,
        reference_image,
        transform,
        sitk.sitkNearestNeighbor,   # ← preserves binary values
        0,
        mask_sitk.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled).astype(np.uint8)


# ---------------------------------------------------------------------------
# Batch registration: align T2, T3 → T1 baseline
# ---------------------------------------------------------------------------

def register_all_to_baseline(
    aorta_volumes: Dict[str, np.ndarray],   # RENAMED from pre_volumes
    stent_volumes: Dict[str, np.ndarray],   # NEW — used to weight metric
    raw_volumes: Dict[str, np.ndarray],
    aorta_masks:   Dict[str, np.ndarray],   # NEW — defines metric computation region
    spacings: Dict[str, Tuple[float, ...]],
    origins: Dict[str, Tuple[float, ...]],
    baseline: str = "aorta_p1_ds1",
    do_deformable: bool = False,
    verbose: bool = False,
) -> Tuple[
    Dict[str, np.ndarray],   # registered preprocessed volumes
    Dict[str, np.ndarray],   # registered raw HU volumes
    Dict[str, np.ndarray],   # registered stent volumes
    Dict[str, sitk.Transform],
]:
    """Align all non-baseline volumes to the T1 baseline.

    Parameters
    ----------
    pre_volumes : dict[name → np.ndarray]
        Normalised volumes (used for computing the registration metric).
    raw_volumes : dict[name → np.ndarray]
        Raw HU volumes (resampled with the same transform for mask generation).
    spacings, origins : dict
        Per-dataset metadata.
    baseline : str
        Key identifying the T1 (fixed) dataset.
    do_deformable : bool
        If True, refine the rigid result with a BSpline step.
    verbose : bool
        Print per-iteration metric values.

    Returns
    -------
    reg_pre : dict  – registered normalised volumes
    reg_raw : dict  – registered raw HU volumes
    transforms : dict – per-dataset transforms (identity for baseline)
    """
    reg_aorta:  Dict[str, np.ndarray]      = {}
    reg_stent:  Dict[str, np.ndarray]      = {}
    reg_raw:    Dict[str, np.ndarray]      = {}
    transforms: Dict[str, sitk.Transform] = {}

    fixed_aorta = aorta_volumes[baseline]
    fixed_sp    = spacings[baseline]
    fixed_or    = origins[baseline]
    fixed_img   = _to_sitk(fixed_aorta, fixed_sp, fixed_or)

    # Build fixed-image aorta mask for metric weighting
    # Dilate slightly so the aorta wall and periwall stent struts are included
    from scipy import ndimage as ndi
    fixed_mask_np = aorta_masks[baseline].astype(np.uint8)
    fixed_mask_np = ndi.binary_dilation(
        fixed_mask_np.astype(bool),
        iterations=5
    ).astype(np.uint8)
    fixed_mask_sitk = sitk.GetImageFromArray(fixed_mask_np)
    fixed_mask_sitk.CopyInformation(fixed_img)

    for name in aorta_volumes:
        if name == baseline:
            reg_aorta[name]  = fixed_aorta
            reg_stent[name]  = stent_volumes[baseline]
            reg_raw[name]    = raw_volumes[baseline]
            transforms[name] = sitk.Transform()
            print(f"  {name}: baseline — no registration needed", flush=True)
            continue

        print(f"  Registering {name} → {baseline} …", flush=True)
        moving_img = _to_sitk(aorta_volumes[name], spacings[name], origins[name])

        init_tx = sitk.CenteredTransformInitializer(
            fixed_img, moving_img,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetInitialTransform(init_tx, inPlace=False)
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.30)   # FIXED from 0.10

        # CRITICAL FIX: restrict metric to aorta region only
        reg.SetMetricFixedMask(fixed_mask_sitk)

        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        if verbose:
            reg.AddCommand(
                sitk.sitkIterationEvent,
                lambda: print(
                    f"    iter {reg.GetOptimizerIteration():4d}  "
                    f"metric={reg.GetMetricValue():.6f}", flush=True
                ),
            )

        tx = reg.Execute(
            sitk.Cast(fixed_img,   sitk.sitkFloat32),
            sitk.Cast(moving_img,  sitk.sitkFloat32),
        )

        # Apply the SAME transform to all three volumes
        def _resample_vol(vol_np, sp, orig, interp=sitk.sitkLinear):
            sitk_vol = _to_sitk(vol_np, sp, orig)
            return _from_sitk(sitk.Resample(
                sitk_vol, fixed_img, tx, interp, 0.0, sitk_vol.GetPixelID()
            ))

        reg_aorta[name]  = _resample_vol(aorta_volumes[name], spacings[name], origins[name])
        reg_stent[name]  = _resample_vol(stent_volumes[name], spacings[name], origins[name])
        reg_raw[name]    = _resample_vol(raw_volumes[name],   spacings[name], origins[name])
        transforms[name] = tx

        print(f"    → done.", flush=True)

    return reg_aorta, reg_stent, reg_raw, transforms