#!/usr/bin/env python
"""
NN-Artifact: Hybrid artifact detection for MR spectroscopy.

Combines two expert models (Tumor CNN and Normal-Brain CNN) with spatial
voxel-shell gating based on FLAIR hyperintensity segmentation to produce
artifact probability maps.

Usage:
    python inference.py --subject-dir /data/subject --study-date 01.31.2018

Docker:
    docker run -v /path/to/subject:/data/subject nn-artifact \
        --subject-dir /data/subject --study-date 01.31.2018
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so artifactremoval package is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from artifactremoval.model_inference import (
    load_tumor_cnn_model,
    load_normal_brain_cnn_model,
    run_tumor_cnn_inference,
    run_normal_brain_cnn_inference,
)
from artifactremoval.hybrid_gating import run_hybrid_gating, _sitk_from_np, FLAIR_GATE_DILATE_VOX
from artifactremoval.update_xml import update_subject_xml

# Resolve model paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
TUMOR_CNN_MODEL_PATH = SCRIPT_DIR / "models" / "RTNNARTIFACT_best.keras"
NORMAL_BRAIN_MODEL_PATH = SCRIPT_DIR / "models" / "NNArtifact_tf2"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("nn-artifact")


def parse_args():
    p = argparse.ArgumentParser(
        description="NN-Artifact: hybrid artifact detection for MR spectroscopy",
    )
    p.add_argument(
        "--subject-dir", required=True, type=Path,
        help="Path to MIDAS subject folder (contains subject.xml)",
    )
    p.add_argument(
        "--study-date", required=True,
        help="Study date string, e.g. '01.31.2018' or '01/31/2018'",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: {subject-dir}/artifactremoval/nn_artifact_output/)",
    )
    p.add_argument(
        "--cerebellum", type=Path, default=None,
        help="Path to cerebellum/brainstem segmentation NIfTI (optional)",
    )
    p.add_argument(
        "--no-qmap", action="store_true",
        help="Disable QMAP==4 override",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="Inference batch size (default: 4096)",
    )
    p.add_argument(
        "--mc-passes", type=int, default=20,
        help="Number of MC dropout forward passes for Tumor CNN (default: 20). "
             "Set to 0 for single deterministic pass.",
    )
    return p.parse_args()


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
        # Labels 5+6 = brainstem + cerebellum
        mask = sitk.Cast((seg == 5) | (seg == 6), sitk.sitkUInt8)
        logger.info(f"Loaded cerebellum/brainstem mask from {path}")
        return mask
    except Exception as e:
        logger.warning(f"Could not load ANTs segmentation: {e}")
        return None


def itk_to_sitk(itk_img):
    """Convert ITK image to SimpleITK image."""
    import itk
    arr = itk.GetArrayFromImage(itk_img)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    sitk_img.SetDirection(tuple(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten()))
    return sitk_img


def main():
    args = parse_args()

    subj_dir = args.subject_dir.resolve()
    if not subj_dir.exists():
        logger.error(f"Subject directory does not exist: {subj_dir}")
        sys.exit(1)

    # Normalize date format
    date_str_dot = args.study_date.replace("/", ".")
    date_str_slash = args.study_date.replace(".", "/")

    # Output directory
    out_dir = args.output_dir or (subj_dir / "artifactremoval" / "nn_artifact_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # ── Load MIDAS subject data ──────────────────────────────────────────
    from artifactremoval.midas import MidasSubject, NNFitDataset

    subject_xml = subj_dir / "subject.xml"
    if not subject_xml.exists():
        logger.error(f"subject.xml not found in {subj_dir}")
        sys.exit(1)

    logger.info(f"Loading MIDAS subject from {subject_xml}")
    subject = MidasSubject(subject_xml)

    # Find the study by date
    study = None
    for s in subject.all_study():
        s_date_dot = s.date.replace("/", ".")
        if s_date_dot == date_str_dot:
            study = s
            break

    if study is None:
        available = [s.date for s in subject.all_study()]
        logger.error(
            f"No study found for date '{date_str_dot}'. "
            f"Available dates: {available}"
        )
        sys.exit(1)

    logger.info(f"Processing study: {study.date}")

    # ── Load spectral data ───────────────────────────────────────────────
    spec = fit = water = None

    try:
        nnfit_ds = NNFitDataset(study, og=False)
        spec = nnfit_ds.load_spectra()
        fit = nnfit_ds.load_baseline() + nnfit_ds.load_peaks()
        logger.info("Loaded spectra via NNFitDataset")
    except Exception as e:
        logger.warning(f"NNFitDataset(og=False) failed: {e}")

    if spec is None or fit is None:
        try:
            nnfit_ds = NNFitDataset(study, og=True)
            spec = spec if spec is not None else nnfit_ds.load_spectra()
            if fit is None:
                fit = nnfit_ds.load_baseline() + nnfit_ds.load_peaks()
            logger.info("Loaded spectra via NNFitDataset(og=True)")
        except Exception as e:
            logger.warning(f"NNFitDataset(og=True) failed: {e}")

    if fit is None:
        try:
            fit = study.fitt()
            logger.info("Loaded fit via study.fitt()")
        except Exception as e:
            logger.warning(f"study.fitt() failed: {e}")

    if water is None:
        try:
            water = study.siref()
            logger.info("Loaded water via study.siref()")
        except Exception as e:
            logger.warning(f"study.siref() failed: {e}")

    if spec is None:
        try:
            spec = study.si()
            logger.info("Loaded spectra via study.si()")
        except Exception as e:
            logger.warning(f"study.si() failed: {e}")

    # Validate
    if spec is None or water is None or fit is None:
        logger.error("Could not load required spectral data (spec, water, fit)")
        sys.exit(1)

    Z, X, Y, S = spec.shape
    logger.info(f"Spectral data shape: Z={Z}, X={X}, Y={Y}, S={S}")

    # ── Load spatial data (REF, brain mask, FLAIR) ───────────────────────
    try:
        ref_itk = study.ref()[1]
        brain_mask_itk = study.brain_mask()[1]
        ref_sitk = itk_to_sitk(ref_itk)
        brain_mask_sitk = itk_to_sitk(brain_mask_itk)
    except Exception as e:
        logger.error(f"Could not load REF/brain_mask from MIDAS: {e}")
        sys.exit(1)

    # FLAIR image and segmentation
    try:
        flair_path = find_flair_image(subj_dir, date_str_dot)
        flair_seg_path = find_flair_seg(subj_dir, date_str_dot)
        flair_sitk = sitk.ReadImage(str(flair_path))
        flair_seg_sitk = sitk.ReadImage(str(flair_seg_path))
    except Exception as e:
        logger.error(f"Could not load FLAIR data: {e}")
        sys.exit(1)

    # Cerebellum mask (optional)
    cereb_mask_sitk = None
    if args.cerebellum:
        try:
            cereb_mask_sitk = sitk.ReadImage(str(args.cerebellum))
            cereb_mask_sitk = sitk.Cast(cereb_mask_sitk > 0, sitk.sitkUInt8)
            logger.info(f"Loaded cerebellum mask from {args.cerebellum}")
        except Exception as e:
            logger.warning(f"Could not load cerebellum mask: {e}")
    else:
        cereb_mask_sitk = find_cereb_seg(subj_dir, study.date)

    # QMAP (optional)
    qmap_sitk = None
    if not args.no_qmap:
        try:
            qmap_itk = study.qmap()[1]
            qmap_sitk = itk_to_sitk(qmap_itk)
            qmap_sitk = sitk.Cast(qmap_sitk, sitk.sitkUInt8)
            logger.info("Loaded QMAP from MIDAS")
        except Exception as e:
            logger.warning(f"Could not load QMAP (continuing without): {e}")

    # ── Load models ──────────────────────────────────────────────────────
    normal_brain_model = load_normal_brain_cnn_model(NORMAL_BRAIN_MODEL_PATH)
    tumor_cnn_model = load_tumor_cnn_model(TUMOR_CNN_MODEL_PATH)

    # ── Run inference ────────────────────────────────────────────────────
    logger.info("Running Normal-Brain CNN inference...")
    normal_brain_art = run_normal_brain_cnn_inference(normal_brain_model, spec,
                                                      batch_size=args.batch_size)
    logger.info("Normal-Brain CNN inference complete")

    mc_label = f"MC dropout (K={args.mc_passes})" if args.mc_passes > 0 else "single pass"
    logger.info(f"Running Tumor CNN inference ({mc_label})...")
    tumor_cnn_art = run_tumor_cnn_inference(tumor_cnn_model, spec, water, fit,
                                            batch_size=args.batch_size,
                                            mc_passes=args.mc_passes)
    logger.info("Tumor CNN inference complete")

    # Free model memory
    del normal_brain_model, tumor_cnn_model
    tf.keras.backend.clear_session()

    # ── Run hybrid gating ────────────────────────────────────────────────
    logger.info("Running hybrid voxel-shell gating...")
    results = run_hybrid_gating(
        tumor_cnn_art=tumor_cnn_art,
        normal_brain_art=normal_brain_art,
        flair_sitk=flair_sitk,
        flair_seg_sitk=flair_seg_sitk,
        ref_sitk=ref_sitk,
        brain_mask_sitk=brain_mask_sitk,
        cereb_mask_sitk=cereb_mask_sitk,
        qmap_sitk=qmap_sitk,
    )
    logger.info("Hybrid gating complete")

    # ── Save outputs ─────────────────────────────────────────────────────
    ref = results["ref_sitk"]
    prefix = date_str_dot

    outputs = {
        f"{prefix}_hybrid_artifact_prob.nii.gz": results["hybrid"],
        f"{prefix}_w_tumor.nii.gz": results["w_tumor"],
        f"{prefix}_tumor_cnn_artifact_prob.nii.gz": results["tumor_cnn_art"],
        f"{prefix}_normal_brain_artifact_prob.nii.gz": results["normal_brain_art"],
    }

    for filename, arr in outputs.items():
        path = out_dir / filename
        img = _sitk_from_np(arr.astype(np.float32), ref, sitk.sitkFloat32)
        sitk.WriteImage(img, str(path))
        logger.info(f"Saved: {path}")

    # ── Compute statistics for XML node ────────────────────────────────
    bm_ref = sitk.Cast(brain_mask_sitk > 0, sitk.sitkUInt8)
    bm_ref = sitk.Resample(bm_ref, ref, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    bm = sitk.GetArrayFromImage(bm_ref).astype(bool)

    n_brain = int(bm.sum())
    n_hybrid = int((results["hybrid"][bm] >= 0.5).sum())
    n_tumor_cnn = int((results["tumor_cnn_art"][bm] >= 0.5).sum())
    n_normal_brain = int((results["normal_brain_art"][bm] >= 0.5).sum())
    n_qmap = results.get("n_qmap_overridden", 0)

    # ── Update subject.xml ─────────────────────────────────────────────
    try:
        data_id = update_subject_xml(
            subject_xml_path=subject_xml,
            study_date_dot=date_str_dot,
            output_dir=out_dir,
            date_prefix=date_str_dot,
            mc_passes=args.mc_passes,
            batch_size=args.batch_size,
            flair_gate_dilation=FLAIR_GATE_DILATE_VOX,
            qmap_override_enabled=not args.no_qmap,
            n_brain_voxels=n_brain,
            n_artifact_hybrid=n_hybrid,
            n_artifact_tumor_cnn=n_tumor_cnn,
            n_artifact_normal_brain=n_normal_brain,
            n_qmap_overridden=n_qmap,
        )
        logger.info(f"Updated subject.xml (data_id={data_id})")
    except Exception as e:
        logger.warning(f"Could not update subject.xml: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
