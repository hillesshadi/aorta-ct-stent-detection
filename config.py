"""
config.py
=========
Central configuration file for the Longitudinal Aortic Stent Displacement
Analysis Pipeline. Edit the path constants and hyperparameters here; all
other modules import from this file.

Usage
-----
    from config import DATASETS, HU_WINDOW_CENTER, ...
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory of the source datasets (DICOM folders)
DATA_ROOT = r"C:\applications\pytorch\aorta_project\code\data"

# The three longitudinal timepoint datasets (T1=baseline, T2, T3)
DATASETS = {
    "aorta_p1_ds1": os.path.join(DATA_ROOT, "aorta_p1_ds1"),   # T1 – 1 198 slices
    "aorta_p1_ds2": os.path.join(DATA_ROOT, "aorta_p1_ds2"),   # T2 – 1 442 slices
    "aorta_p1_ds3": os.path.join(DATA_ROOT, "aorta_p1_ds3"),   # T3 –   696 slices
}

# T1 is the registration baseline; T2 and T3 will be aligned to it
BASELINE_DATASET = "aorta_p1_ds1"

# Output directories (created automatically)
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR        = os.path.join(OUTPUT_DIR, "_cache")
CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# HU windowing – captures aorta lumen (blood + contrast) and dense metal stent
HU_WINDOW_CENTER = 350.0    # (L)  centre of the display window (HU)
HU_WINDOW_WIDTH  = 600.0   # (W)  width of the display window  (HU)
# Resulting clip range:  [-450, 1050] HU

# Window A: Aorta/soft-tissue (blood pool, vessel wall)
# Clip range: [50, 650] HU — captures contrast-enhanced lumen
HU_AORTA_CENTER = 350.0
HU_AORTA_WIDTH  = 600.0

# Window B: Metal/stent
# Clip range: [500, 2500] HU — captures stent struts and dense metal
HU_STENT_CENTER = 1500.0
HU_STENT_WIDTH  = 2000.0

# Legacy single-window (kept only for EADTV enhancement of soft tissue)
# Do NOT use this for stent preprocessing
HU_WINDOW_CENTER = 350.0
HU_WINDOW_WIDTH  = 600.0


# Normalisation mode: "minmax" → [0,1]  |  "zscore" → zero-mean unit-variance
NORM_MODE = "minmax"        # "minmax" | "zscore"
# HU above this value is classified as metal — excluded from TV denoising
HU_METAL_THRESHOLD = 800.0   # stent struts typically >1000; 800 adds safety margi

# ---------------------------------------------------------------------------
# EADTV Enhancement (Edge-Aware Discrete Total Variation)
# ---------------------------------------------------------------------------

# EADTV_WEIGHT   = 0.05     # TV regularisation weight (lower = less smoothing)
# EADTV_MAX_ITER = 20       # Chambolle iteration limit

# EADTV Enhancement — SOFT TISSUE ONLY (metal regions are masked out)
EADTV_WEIGHT_SOFT  = 0.05    # reduced from 0.08 — preserves vessel wall edges
EADTV_WEIGHT_METAL = 0.0     # metal/stent regions: NO TV denoising applied
EADTV_MAX_ITER     = 20      # reduced from 30 — less aggressive smoothing

# ---------------------------------------------------------------------------
# Inter-scan Registration
# ---------------------------------------------------------------------------

REG_RIGID_ITERS    = 200           # gradient-descent iterations (rigid)
REG_RIGID_LR       = 1.0           # learning rate (SimpleITK optimizer)
REG_RIGID_SHRINK   = [4, 2, 1]     # multi-resolution shrink factors
REG_RIGID_SMOOTH   = [2, 1, 0]     # Gaussian smoothing sigmas (voxels)
REG_DEFORMABLE     = False         # set True to add BSpline deformable step

# ---------------------------------------------------------------------------
# Aorta Mask (auto-generation via HU thresholding)
# ---------------------------------------------------------------------------

HU_MIN_AORTA    = 150.0    # lower HU bound for blood-pool / contrast
HU_MAX_AORTA    = 550.0    # upper HU bound (excludes cortical bone)
AORTA_MIN_AREA  = 200      # minimum slice area (voxels) to keep a CC
AORTA_MAX_AREA  = 25_000   # maximum slice area (voxels) – removes lung fields

# ---------------------------------------------------------------------------
# Stent Mask (auto-generation via HU thresholding inside aorta ROI)
# ---------------------------------------------------------------------------

HU_MIN_STENT           = 800.0   # metal / dense stent material threshold (HU)
STENT_DILATE_AORTA_ITERS = 8      # extra dilation of aorta mask before stent search
STENT_MIN_CC_VOXELS    = 5        # minimum connected-component size for stent

# ---------------------------------------------------------------------------
# Deep Learning – Segmentation Models
# ---------------------------------------------------------------------------

BATCH_SIZE     = 2
EPOCHS         = 100               # set higher for production runs
LEARNING_RATE  = 3e-4
IMAGE_SIZE     = (512, 512)       # each 2-D slice is resized to this
BASE_CHANNELS  = 32               # base feature maps for all U-Net variants

# Device: auto-select GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# MORPH_CLOSING_RADIUS  = 3    # binary morphological closing (fill gaps)
# MORPH_OPENING_RADIUS  = 1    # binary morphological opening (remove noise)
# CC_MIN_VOXELS         = 30   # minimum voxels for a kept connected component
MORPH_CLOSING_RADIUS_AORTA  = 3    # larger for aorta wall continuity
MORPH_CLOSING_RADIUS_STENT  = 1    # small — stent wires are thin; over-closing merges struts
MORPH_OPENING_RADIUS        = 1
CC_MIN_VOXELS_AORTA         = 50   # raised from 20
CC_MIN_VOXELS_STENT         = 3    # very small — stent CCs are legitimately tiny


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

INCLUDE_HAUSDORFF = True     # compute HD95 (slow for large masks)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

VIZ_VIEWS       = ["isometric", "sagittal", "coronal", "axial"]
VIZ_OFFSCREEN   = True       # False → open interactive PyVista window
VIZ_WINDOW_SIZE = (1600, 1200)

# Colour palette
COLOR_AORTA      = "#ff6347"   # tomato-red
COLOR_STENT_ORIG = "silver"    # original stent position
COLOR_STENT_DISP = "#ffd700"   # displaced stent position (gold)
COLOR_ARROW      = "lime"      # displacement arrow

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

REPORT_CSV  = os.path.join(OUTPUT_DIR, "displacement_report.csv")
REPORT_PDF  = os.path.join(OUTPUT_DIR, "clinical_report.pdf")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")
