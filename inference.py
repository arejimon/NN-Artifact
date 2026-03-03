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

import tensorflow as tf

from artifactremoval.model_inference import load_normal_brain_cnn_model, load_tumor_cnn_model
from artifactremoval.pipeline import run_subject_study

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


def main():
    args = parse_args()

    subj_dir = args.subject_dir.resolve()
    if not subj_dir.exists():
        logger.error(f"Subject directory does not exist: {subj_dir}")
        sys.exit(1)

    # Load models
    normal_brain_model = load_normal_brain_cnn_model(NORMAL_BRAIN_MODEL_PATH)
    tumor_cnn_model = load_tumor_cnn_model(TUMOR_CNN_MODEL_PATH)

    try:
        run_subject_study(
            subject_dir=subj_dir,
            study_date=args.study_date,
            tumor_cnn_model=tumor_cnn_model,
            normal_brain_model=normal_brain_model,
            output_dir=args.output_dir,
            cerebellum=args.cerebellum,
            no_qmap=args.no_qmap,
            batch_size=args.batch_size,
            mc_passes=args.mc_passes,
        )
    finally:
        del tumor_cnn_model, normal_brain_model
        tf.keras.backend.clear_session()

    logger.info("Done!")


if __name__ == "__main__":
    main()
