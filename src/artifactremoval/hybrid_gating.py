"""
Hybrid voxel-shell gating for combining Tumor CNN and Normal-Brain CNN expert predictions.

Extracted from notebooks/45_hybrid_model.ipynb.
Hardcoded configuration: GATE_MODE="voxel", FLAIR_GATE_DILATE_VOX=2.
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from pathlib import Path

# ── Hardcoded configuration (baked per user request) ─────────────────────
FLAIR_GATE_DILATE_VOX = 2
CEREB_DILATE_VOX = 1
SEG_CLEAN_MIN_CC = 100
SEG_CLEAN_CLOSE_RAD = 1
SEG_CLEAN_OPEN_RAD = 1


# ── Helper functions ─────────────────────────────────────────────────────

def _np_zyx(img: sitk.Image) -> np.ndarray:
    """SimpleITK image to numpy array (z,y,x ordering)."""
    return sitk.GetArrayFromImage(img)


def _sitk_from_np(arr_zyx: np.ndarray, ref: sitk.Image,
                  pixel_type=sitk.sitkFloat32) -> sitk.Image:
    """Create SimpleITK image from numpy array, copying geometry from ref."""
    out = sitk.GetImageFromArray(arr_zyx)
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    return sitk.Cast(out, pixel_type)


# ── Registration ─────────────────────────────────────────────────────────

def register_to_ref(fixed_image, moving_image, fixed_mask=None):
    """
    Rigid registration (Euler3D) of moving image to fixed (REF) space.
    Uses Mattes Mutual Information, multi-resolution.

    Returns
    -------
    sitk.Transform
        The rigid transform mapping moving → fixed.
    """
    fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving = sitk.Cast(moving_image, sitk.sitkFloat32)

    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=42)

    if fixed_mask is not None:
        fm = sitk.Cast(fixed_mask > 0, sitk.sitkUInt8)
        fm = sitk.Resample(fm, fixed, sitk.Transform(),
                           sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        R.SetMetricFixedMask(fm)

    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3,
        numberOfIterations=300, gradientMagnitudeTolerance=1e-6,
        relaxationFactor=0.5,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(sitk.Euler3DTransform(init_tx), inPlace=False)

    return R.Execute(fixed, moving)


# ── Segmentation cleaning ────────────────────────────────────────────────

def clean_flair_seg(seg_in_ref, ref_mask,
                    min_cc_vox=SEG_CLEAN_MIN_CC,
                    close_rad_vox=SEG_CLEAN_CLOSE_RAD,
                    open_rad_vox=SEG_CLEAN_OPEN_RAD):
    """
    Clean a binary FLAIR seg mask in REF space:
      1) Constrain to brain mask
      2) Remove connected components < min_cc_vox
      3) Morphological closing then opening
    """
    seg = sitk.Cast(seg_in_ref > 0, sitk.sitkUInt8)
    bm = sitk.Cast(ref_mask > 0, sitk.sitkUInt8)
    bm = sitk.Resample(bm, seg, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    seg = sitk.And(seg, bm)

    cc = sitk.ConnectedComponent(seg)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if stats.GetNumberOfLabels() == 0:
        return seg

    out = sitk.Image(seg.GetSize(), sitk.sitkUInt8)
    out.CopyInformation(seg)
    for L in stats.GetLabels():
        if stats.GetNumberOfPixels(L) >= int(min_cc_vox):
            out = sitk.Or(out, sitk.Cast(cc == L, sitk.sitkUInt8))
    seg = sitk.And(out, bm)

    if close_rad_vox > 0:
        seg = sitk.BinaryMorphologicalClosing(seg, (close_rad_vox,) * 3, sitk.sitkBall)
        seg = sitk.And(seg, bm)

    if open_rad_vox > 0:
        seg = sitk.BinaryMorphologicalOpening(seg, (open_rad_vox,) * 3, sitk.sitkBall)
        seg = sitk.And(seg, bm)

    return sitk.Cast(seg, sitk.sitkUInt8)


# ── Gate building ────────────────────────────────────────────────────────

def dilate_within_brain(seg, brain_mask, radius_vox):
    """Dilate seg by radius_vox within brain_mask. Returns UInt8 image."""
    if isinstance(radius_vox, int):
        rad = (radius_vox, radius_vox, radius_vox)
    else:
        rad = tuple(int(r) for r in radius_vox)

    seg_u8 = sitk.Cast(seg > 0, sitk.sitkUInt8)
    brain = sitk.Cast(brain_mask > 0, sitk.sitkUInt8)
    brain = sitk.Resample(brain, seg_u8, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    if rad == (0, 0, 0):
        return sitk.And(seg_u8, brain)

    dil = sitk.BinaryDilate(seg_u8, rad, sitk.sitkBall)
    return sitk.And(dil, brain)


def build_gate(seg_clean, ref_mask, dilate_vox=FLAIR_GATE_DILATE_VOX):
    """
    Build the gate: dilate cleaned FLAIR seg within brain mask.

    Returns
    -------
    seg_gate : sitk.Image
        Dilated gate as UInt8 mask.
    gate_bool : np.ndarray
        Boolean gate array (z,y,x).
    """
    seg_gate = dilate_within_brain(seg_clean, ref_mask,
                                   radius_vox=(dilate_vox,) * 3)
    bm = (_np_zyx(ref_mask) > 0)
    gate_bool = (_np_zyx(seg_gate) > 0) & bm
    return seg_gate, gate_bool


# ── Voxel-shell weight computation ──────────────────────────────────────

def compute_voxel_shell_weights(seg_gate, brain_mask):
    """
    Compute Tumor CNN weight map using voxel-distance shells.

    Inside gate:        w = 1.0
    Shell 0-1 voxels:   w = 0.75
    Shell 1-2 voxels:   w = 0.50
    Shell 2-3 voxels:   w = 0.25
    Beyond 3 voxels:    w = 0.0

    Returns
    -------
    w_tumor : np.ndarray
        Weight map (z,y,x), float32.
    gate_bool : np.ndarray
        Boolean gate (z,y,x).
    """
    bm = (_np_zyx(brain_mask) > 0)
    seg_bool = (_np_zyx(seg_gate) > 0)
    gate_bool = seg_bool & bm

    # Distance outside gate in voxel units
    outside = ~seg_bool
    d_out_vox = distance_transform_edt(outside).astype(np.float32)
    d_out_vox[seg_bool] = 0.0
    d = np.where(bm, d_out_vox, 0.0).astype(np.float32)

    # Piecewise shell weights
    w_tumor = np.zeros_like(d, dtype=np.float32)
    w_tumor[gate_bool] = 1.0
    w_tumor[(~gate_bool) & (d > 0.0) & (d <= 1.0)] = 0.75
    w_tumor[(~gate_bool) & (d > 1.0) & (d <= 2.0)] = 0.50
    w_tumor[(~gate_bool) & (d > 2.0) & (d <= 3.0)] = 0.25

    return w_tumor, gate_bool


# ── Cerebellum override ──────────────────────────────────────────────────

def apply_cerebellum_override(w_tumor, gate_bool, cereb_mask_sitk,
                              ref_sitk, brain_mask):
    """
    Force w_tumor=0 in cerebellum/brainstem voxels outside the gate.

    Parameters
    ----------
    cereb_mask_sitk : sitk.Image or None
        Cerebellum/brainstem mask (labels 5+6 from ANTs segmentation).
        If None, no override is applied.
    """
    if cereb_mask_sitk is None:
        return w_tumor

    bm = (_np_zyx(brain_mask) > 0)

    # Optionally dilate cerebellum mask
    cereb_dil = dilate_within_brain(cereb_mask_sitk, brain_mask,
                                     radius_vox=(CEREB_DILATE_VOX,) * 3)
    cereb_bool = (_np_zyx(cereb_dil) > 0) & bm

    # In cerebellum but outside gate → force Normal-Brain CNN (w=0)
    w_tumor = np.where(cereb_bool & (~gate_bool), 0.0, w_tumor).astype(np.float32)
    return w_tumor


# ── Expert blending ──────────────────────────────────────────────────────

def blend_experts(tumor_cnn_art, normal_brain_art, w_tumor):
    """
    Blend two expert predictions using weight map.

    hybrid = w * P_tumor + (1-w) * P_normal_brain
    """
    hybrid = (w_tumor * tumor_cnn_art + (1.0 - w_tumor) * normal_brain_art).astype(np.float32)
    return hybrid


# ── QMAP override ────────────────────────────────────────────────────────

def apply_qmap_override(hybrid, w_tumor, qmap_sitk, flair_seg_bool):
    """
    Force artifact_prob=0.0 where QMAP==4 within the FLAIR segmentation.

    Parameters
    ----------
    qmap_sitk : sitk.Image or None
        QMAP label image in REF space. If None, no override applied.
    flair_seg_bool : np.ndarray
        Boolean mask of the raw (undilated) FLAIR segmentation in REF space.

    Returns
    -------
    hybrid : np.ndarray
        Updated hybrid probabilities.
    w_tumor : np.ndarray
        Updated weight map.
    n_changed : int
        Number of voxels overridden.
    """
    if qmap_sitk is None:
        return hybrid, w_tumor, 0

    qmap4_bool = (_np_zyx(qmap_sitk) == 4) & flair_seg_bool

    n_changed = int(np.count_nonzero(qmap4_bool & (hybrid >= 0.5)))
    hybrid = np.where(qmap4_bool, 0.0, hybrid).astype(np.float32)
    w_tumor = np.where(qmap4_bool, 0.0, w_tumor).astype(np.float32)

    return hybrid, w_tumor, n_changed


# ── Full hybrid pipeline ────────────────────────────────────────────────

def run_hybrid_gating(
    *,
    tumor_cnn_art,
    normal_brain_art,
    flair_sitk,
    flair_seg_sitk,
    ref_sitk,
    brain_mask_sitk,
    cereb_mask_sitk=None,
    qmap_sitk=None,
):
    """
    Run the full hybrid voxel-shell gating pipeline.

    Parameters
    ----------
    tumor_cnn_art : np.ndarray
        Tumor CNN P(artifact), shape (Z,Y,X).
    normal_brain_art : np.ndarray
        Normal-Brain CNN P(artifact), same shape.
    flair_sitk : sitk.Image
        FLAIR image (native space).
    flair_seg_sitk : sitk.Image
        FLAIR hyperintensity segmentation (native space).
    ref_sitk : sitk.Image
        Reference image (REF/water-ref space).
    brain_mask_sitk : sitk.Image
        Brain mask in REF space.
    cereb_mask_sitk : sitk.Image or None
        Cerebellum mask in REF space (optional).
    qmap_sitk : sitk.Image or None
        QMAP in REF space (optional).

    Returns
    -------
    dict with keys:
        hybrid, w_tumor, tumor_cnn_art, normal_brain_art (all np.ndarray),
        ref_sitk (for saving NIfTIs), n_qmap_overridden (int).
    """
    # Ensure brain mask is on ref grid
    brain_mask = sitk.Cast(brain_mask_sitk > 0, sitk.sitkUInt8)
    brain_mask = sitk.Resample(brain_mask, ref_sitk, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    bm = (_np_zyx(brain_mask) > 0)

    # Register FLAIR to REF space
    tx = register_to_ref(ref_sitk, flair_sitk, fixed_mask=brain_mask)

    # Resample segmentation into REF space (nearest-neighbor)
    seg_in_ref = sitk.Resample(flair_seg_sitk, ref_sitk, tx,
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # Clean segmentation
    seg_clean = clean_flair_seg(seg_in_ref, brain_mask)

    # Build dilated gate
    seg_gate, gate_bool = build_gate(seg_clean, brain_mask)

    # Compute voxel-shell weights
    w_tumor, gate_bool = compute_voxel_shell_weights(seg_gate, brain_mask)

    # Cerebellum override
    if cereb_mask_sitk is not None:
        cereb_ref = sitk.Resample(cereb_mask_sitk, ref_sitk, sitk.Transform(),
                                   sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        w_tumor = apply_cerebellum_override(w_tumor, gate_bool, cereb_ref,
                                            ref_sitk, brain_mask)

    # Apply brain mask to expert predictions
    tumor_cnn_art = np.where(bm, np.clip(tumor_cnn_art, 0.0, 1.0), 0.0).astype(np.float32)
    normal_brain_art = np.where(bm, np.clip(normal_brain_art, 0.0, 1.0), 0.0).astype(np.float32)

    # Blend
    hybrid = blend_experts(tumor_cnn_art, normal_brain_art, w_tumor)

    # QMAP override
    n_changed = 0
    if qmap_sitk is not None:
        qmap_ref = sitk.Resample(qmap_sitk, ref_sitk, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        flair_seg_bool = (_np_zyx(seg_clean) > 0)
        hybrid, w_tumor, n_changed = apply_qmap_override(hybrid, w_tumor,
                                                         qmap_ref, flair_seg_bool)
        if n_changed > 0:
            print(f"QMAP4 override: {n_changed} artifact voxels reclassified as good")

    return dict(
        hybrid=hybrid,
        w_tumor=w_tumor,
        tumor_cnn_art=tumor_cnn_art,
        normal_brain_art=normal_brain_art,
        ref_sitk=ref_sitk,
        seg_clean=seg_clean,
        n_qmap_overridden=n_changed,
    )
