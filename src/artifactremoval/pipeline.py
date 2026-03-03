"""
Core inference pipeline for a single (subject, study) pair.

Shared by inference.py (single-subject CLI) and batch_inference.py (project-wide batch).
"""

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from artifactremoval.hybrid_gating import (
    FLAIR_GATE_DILATE_VOX,
    _sitk_from_np,
    run_hybrid_gating,
)
from artifactremoval.model_inference import (
    run_normal_brain_cnn_inference,
    run_tumor_cnn_inference,
)
from artifactremoval.update_xml import update_subject_xml

logger = logging.getLogger(__name__)


# ── ITK / SimpleITK helpers ──────────────────────────────────────────────────

def itk_to_sitk(itk_img):
    """Convert an ITK image to a SimpleITK image."""
    import itk
    arr = itk.GetArrayFromImage(itk_img)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    sitk_img.SetDirection(
        tuple(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    )
    return sitk_img


# ── File-discovery helpers ───────────────────────────────────────────────────

def find_flair_seg(subj_dir, date_str_dot):
    """
    Locate FLAIR segmentation in the subject directory.
    Tries multiple known filenames.
    """
    autoseg_dir = subj_dir / "mri" / f"{date_str_dot}_autoseg"
    candidates = [
        autoseg_dir / "binary_segmentation_flair_128.nii.gz",
        autoseg_dir / "bricsflairseg_cleaned.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            logger.info(f"Found FLAIR seg: {path}")
            return path
    raise FileNotFoundError(
        f"No FLAIR segmentation found in {autoseg_dir}. "
        f"Looked for: {[c.name for c in candidates]}"
    )


def find_flair_image(subj_dir, date_str_dot):
    """
    Locate FLAIR image in the subject directory.
    Tries multiple known filenames.
    """
    autoseg_dir = subj_dir / "mri" / f"{date_str_dot}_autoseg"
    candidates = [
        autoseg_dir / "preprocessed_flair.nii.gz",
        autoseg_dir / "bricsreft1_cleaned.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            logger.info(f"Found FLAIR image: {path}")
            return path
    raise FileNotFoundError(
        f"No FLAIR image found in {autoseg_dir}. "
        f"Looked for: {[c.name for c in candidates]}"
    )


def find_cereb_seg(subj_dir, study_date_slash):
    """
    Try to find ANTs cerebellum/brainstem segmentation.
    Returns sitk.Image or None.
    """
    date_us = study_date_slash.replace("/", "_")
    path = subj_dir / "ants" / f"segmentations_{date_us}.nii.gz"
    if not path.exists():
        return None
    try:
        seg = sitk.ReadImage(str(path))
        mask = sitk.Cast((seg == 5) | (seg == 6), sitk.sitkUInt8)
        logger.info(f"Loaded cerebellum/brainstem mask from {path}")
        return mask
    except Exception as e:
        logger.warning(f"Could not load ANTs segmentation: {e}")
        return None


# ── Core pipeline ────────────────────────────────────────────────────────────

def run_subject_study(
    subject_dir,
    study_date,
    tumor_cnn_model,
    normal_brain_model,
    output_dir=None,
    cerebellum=None,
    no_qmap=False,
    batch_size=4096,
    mc_passes=20,
):
    """
    Run the full NN-Artifact pipeline for one subject/study pair.

    Parameters
    ----------
    subject_dir : Path
        Path to MIDAS subject folder (must contain subject.xml).
    study_date : str
        Study date, e.g. '01.31.2018' or '01/31/2018'.
    tumor_cnn_model : tf.keras.Model
        Pre-loaded Tumor CNN model.
    normal_brain_model : tf.saved_model
        Pre-loaded Normal-Brain CNN SavedModel.
    output_dir : Path, optional
        Output directory. Defaults to
        {subject_dir}/artifactremoval/nn_artifact_output/.
    cerebellum : Path, optional
        Explicit path to cerebellum segmentation NIfTI.
    no_qmap : bool
        Disable QMAP==4 override.
    batch_size : int
        Inference batch size.
    mc_passes : int
        MC dropout passes for Tumor CNN (0 = deterministic).

    Returns
    -------
    dict
        Keys: subject_dir, study_date, output_dir, n_brain, n_hybrid,
              artifact_pct, n_tumor_cnn, n_normal_brain, n_qmap.

    Raises
    ------
    Exception
        On any fatal error (missing data, failed registration, etc.).
    """
    from artifactremoval.midas import MidasSubject, NNFitDataset

    subject_dir = Path(subject_dir)
    date_str_dot = study_date.replace("/", ".")
    date_str_slash = study_date.replace(".", "/")
    tag = f"{subject_dir.name} | {date_str_dot}"

    out_dir = output_dir or (
        subject_dir / "artifactremoval" / "nn_artifact_output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{tag}] Output directory: {out_dir}")

    # ── Load MIDAS subject ────────────────────────────────────────────────
    subject_xml = subject_dir / "subject.xml"
    if not subject_xml.exists():
        raise FileNotFoundError(f"subject.xml not found in {subject_dir}")

    subject = MidasSubject(subject_xml)

    study = None
    for s in subject.all_study():
        if s.date.replace("/", ".") == date_str_dot:
            study = s
            break

    if study is None:
        available = [s.date for s in subject.all_study()]
        raise ValueError(
            f"No study found for date '{date_str_dot}'. "
            f"Available dates: {available}"
        )

    logger.info(f"[{tag}] Found study")

    # ── Load spectral data ────────────────────────────────────────────────
    spec = fit = water = None

    try:
        nnfit_ds = NNFitDataset(study, og=False)
        spec = nnfit_ds.load_spectra()
        fit = nnfit_ds.load_baseline() + nnfit_ds.load_peaks()
        logger.info(f"[{tag}] Loaded spectra via NNFitDataset(og=False)")
    except Exception as e:
        logger.warning(f"[{tag}] NNFitDataset(og=False) failed: {e}")

    if spec is None or fit is None:
        try:
            nnfit_ds = NNFitDataset(study, og=True)
            spec = spec if spec is not None else nnfit_ds.load_spectra()
            if fit is None:
                fit = nnfit_ds.load_baseline() + nnfit_ds.load_peaks()
            logger.info(f"[{tag}] Loaded spectra via NNFitDataset(og=True)")
        except Exception as e:
            logger.warning(f"[{tag}] NNFitDataset(og=True) failed: {e}")

    if fit is None:
        try:
            fit = study.fitt()
            logger.info(f"[{tag}] Loaded fit via study.fitt()")
        except Exception as e:
            logger.warning(f"[{tag}] study.fitt() failed: {e}")

    if water is None:
        try:
            water = study.siref()
            logger.info(f"[{tag}] Loaded water via study.siref()")
        except Exception as e:
            logger.warning(f"[{tag}] study.siref() failed: {e}")

    if spec is None:
        try:
            spec = study.si()
            logger.info(f"[{tag}] Loaded spectra via study.si()")
        except Exception as e:
            logger.warning(f"[{tag}] study.si() failed: {e}")

    if spec is None or water is None or fit is None:
        raise RuntimeError(
            f"[{tag}] Could not load required spectral data (spec, water, fit)"
        )

    Z, X, Y, S = spec.shape
    logger.info(f"[{tag}] Spectral shape: Z={Z}, X={X}, Y={Y}, S={S}")

    # ── Load spatial data ─────────────────────────────────────────────────
    try:
        ref_itk = study.ref()[1]
        brain_mask_itk = study.brain_mask()[1]
        ref_sitk = itk_to_sitk(ref_itk)
        brain_mask_sitk = itk_to_sitk(brain_mask_itk)
    except Exception as e:
        raise RuntimeError(f"[{tag}] Could not load REF/brain_mask: {e}") from e

    # ── FLAIR ─────────────────────────────────────────────────────────────
    try:
        flair_path = find_flair_image(subject_dir, date_str_dot)
        flair_seg_path = find_flair_seg(subject_dir, date_str_dot)
        flair_sitk = sitk.ReadImage(str(flair_path))
        flair_seg_sitk = sitk.ReadImage(str(flair_seg_path))
    except Exception as e:
        raise RuntimeError(f"[{tag}] Could not load FLAIR data: {e}") from e

    # ── Cerebellum (optional) ─────────────────────────────────────────────
    cereb_mask_sitk = None
    if cerebellum:
        try:
            cereb_mask_sitk = sitk.ReadImage(str(cerebellum))
            cereb_mask_sitk = sitk.Cast(cereb_mask_sitk > 0, sitk.sitkUInt8)
            logger.info(f"[{tag}] Loaded cerebellum mask from {cerebellum}")
        except Exception as e:
            logger.warning(f"[{tag}] Could not load cerebellum mask: {e}")
    else:
        cereb_mask_sitk = find_cereb_seg(subject_dir, date_str_slash)

    # ── QMAP (optional) ───────────────────────────────────────────────────
    qmap_sitk = None
    if not no_qmap:
        try:
            qmap_itk = study.qmap()[1]
            qmap_sitk = itk_to_sitk(qmap_itk)
            qmap_sitk = sitk.Cast(qmap_sitk, sitk.sitkUInt8)
            logger.info(f"[{tag}] Loaded QMAP")
        except Exception as e:
            logger.warning(f"[{tag}] Could not load QMAP (continuing without): {e}")

    # ── Inference ─────────────────────────────────────────────────────────
    logger.info(f"[{tag}] Running Normal-Brain CNN inference...")
    normal_brain_art = run_normal_brain_cnn_inference(
        normal_brain_model, spec, batch_size=batch_size
    )

    mc_label = f"MC dropout (K={mc_passes})" if mc_passes > 0 else "single pass"
    logger.info(f"[{tag}] Running Tumor CNN inference ({mc_label})...")
    tumor_cnn_art = run_tumor_cnn_inference(
        tumor_cnn_model, spec, water, fit,
        batch_size=batch_size,
        mc_passes=mc_passes,
    )

    # ── Hybrid gating ─────────────────────────────────────────────────────
    logger.info(f"[{tag}] Running hybrid voxel-shell gating...")
    gating_results = run_hybrid_gating(
        tumor_cnn_art=tumor_cnn_art,
        normal_brain_art=normal_brain_art,
        flair_sitk=flair_sitk,
        flair_seg_sitk=flair_seg_sitk,
        ref_sitk=ref_sitk,
        brain_mask_sitk=brain_mask_sitk,
        cereb_mask_sitk=cereb_mask_sitk,
        qmap_sitk=qmap_sitk,
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    ref = gating_results["ref_sitk"]
    prefix = date_str_dot

    outputs = {
        f"{prefix}_hybrid_artifact_prob.nii.gz":       gating_results["hybrid"],
        f"{prefix}_w_tumor.nii.gz":                    gating_results["w_tumor"],
        f"{prefix}_tumor_cnn_artifact_prob.nii.gz":    gating_results["tumor_cnn_art"],
        f"{prefix}_normal_brain_artifact_prob.nii.gz": gating_results["normal_brain_art"],
    }

    for filename, arr in outputs.items():
        path = out_dir / filename
        img = _sitk_from_np(arr.astype(np.float32), ref, sitk.sitkFloat32)
        sitk.WriteImage(img, str(path))
        logger.info(f"[{tag}] Saved: {path}")

    # Save FLAIR segmentation registered to REF space (used by create_chonaanorm.py)
    seg_in_ref_path = out_dir / f"{prefix}_flair_seg_in_ref.nii.gz"
    sitk.WriteImage(
        sitk.Cast(gating_results["seg_clean"], sitk.sitkUInt8),
        str(seg_in_ref_path),
    )
    logger.info(f"[{tag}] Saved: {seg_in_ref_path}")

    # ── Statistics ────────────────────────────────────────────────────────
    bm_ref = sitk.Cast(brain_mask_sitk > 0, sitk.sitkUInt8)
    bm_ref = sitk.Resample(
        bm_ref, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )
    bm = sitk.GetArrayFromImage(bm_ref).astype(bool)

    n_brain = int(bm.sum())
    n_hybrid = int((gating_results["hybrid"][bm] >= 0.5).sum())
    n_tumor_cnn = int((gating_results["tumor_cnn_art"][bm] >= 0.5).sum())
    n_normal_brain = int((gating_results["normal_brain_art"][bm] >= 0.5).sum())
    n_qmap = gating_results.get("n_qmap_overridden", 0)

    # ── Update subject.xml ────────────────────────────────────────────────
    try:
        data_id = update_subject_xml(
            subject_xml_path=subject_xml,
            study_date_dot=date_str_dot,
            output_dir=out_dir,
            date_prefix=date_str_dot,
            mc_passes=mc_passes,
            batch_size=batch_size,
            flair_gate_dilation=FLAIR_GATE_DILATE_VOX,
            qmap_override_enabled=not no_qmap,
            n_brain_voxels=n_brain,
            n_artifact_hybrid=n_hybrid,
            n_artifact_tumor_cnn=n_tumor_cnn,
            n_artifact_normal_brain=n_normal_brain,
            n_qmap_overridden=n_qmap,
        )
        logger.info(f"[{tag}] Updated subject.xml (data_id={data_id})")
    except Exception as e:
        logger.warning(f"[{tag}] Could not update subject.xml: {e}")

    logger.info(f"[{tag}] Done")

    return {
        "subject_dir": subject_dir,
        "study_date": date_str_dot,
        "output_dir": out_dir,
        "n_brain": n_brain,
        "n_hybrid": n_hybrid,
        "artifact_pct": round(100.0 * n_hybrid / max(n_brain, 1), 1),
        "n_tumor_cnn": n_tumor_cnn,
        "n_normal_brain": n_normal_brain,
        "n_qmap": n_qmap,
    }
