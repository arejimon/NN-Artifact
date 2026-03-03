#!/usr/bin/env python
"""
NN-Artifact batch processing: run inference for all subjects and studies
in a MIDAS project.

Models are loaded once per worker process (not once per study), so the
overhead of loading large model files is amortised across many studies.

Usage:
    python batch_inference.py --project-xml /path/to/project.xml --mc-passes 5
"""

import argparse
import concurrent.futures
import csv
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add src/ to path so artifactremoval package is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Resolve model paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
TUMOR_CNN_MODEL_PATH = SCRIPT_DIR / "models" / "RTNNARTIFACT_best.keras"
NORMAL_BRAIN_MODEL_PATH = SCRIPT_DIR / "models" / "NNArtifact_tf2"

# ── Worker-process globals (populated by _worker_init) ───────────────────────
_tumor_model = None
_normal_brain_model = None


def _worker_init(tumor_path, normal_brain_path):
    """Load both models once in each worker process."""
    global _tumor_model, _normal_brain_model

    # Allow TF to grow GPU memory incrementally rather than claiming all
    # VRAM upfront — required when multiple worker processes share one GPU.
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    from artifactremoval.model_inference import (
        load_normal_brain_cnn_model,
        load_tumor_cnn_model,
    )
    _normal_brain_model = load_normal_brain_cnn_model(normal_brain_path)
    _tumor_model = load_tumor_cnn_model(tumor_path)


def _run_task(task):
    """
    Process one (subject_dir, study_date) pair using the worker-loaded models.
    Never raises — on error writes a .txt file and returns an error result dict.
    """
    subject_dir, study_date, batch_size, mc_passes, no_qmap = task
    subject_dir = Path(subject_dir)
    date_dot = study_date.replace("/", ".")
    start = time.time()

    from artifactremoval.pipeline import run_subject_study

    try:
        result = run_subject_study(
            subject_dir=subject_dir,
            study_date=study_date,
            tumor_cnn_model=_tumor_model,
            normal_brain_model=_normal_brain_model,
            batch_size=batch_size,
            mc_passes=mc_passes,
            no_qmap=no_qmap,
        )
        result["status"] = "success"
        result["error"] = ""
        result["duration_s"] = round(time.time() - start, 1)
        return result

    except Exception as e:
        err_text = traceback.format_exc()

        # Write error file into subject's artifactremoval folder
        err_dir = subject_dir / "artifactremoval"
        err_dir.mkdir(parents=True, exist_ok=True)
        err_file = err_dir / f"nn_artifact_error_{date_dot}.txt"
        err_file.write_text(
            f"NN-Artifact error\n"
            f"Subject : {subject_dir}\n"
            f"Study   : {date_dot}\n"
            f"Time    : {datetime.now().isoformat()}\n\n"
            f"{err_text}"
        )

        return {
            "subject_dir": str(subject_dir),
            "study_date": date_dot,
            "status": "error",
            "error": str(e),
            "n_brain": "",
            "n_hybrid": "",
            "artifact_pct": "",
            "n_tumor_cnn": "",
            "n_normal_brain": "",
            "n_qmap": "",
            "duration_s": round(time.time() - start, 1),
        }


def _write_summary(results, summary_path):
    fieldnames = [
        "subject_dir", "study_date", "status",
        "n_brain", "n_hybrid", "artifact_pct",
        "n_tumor_cnn", "n_normal_brain", "n_qmap",
        "duration_s", "error",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    p = argparse.ArgumentParser(
        description="NN-Artifact batch: process all subjects/studies in a MIDAS project",
    )
    p.add_argument(
        "--project-xml", required=True, type=Path,
        help="Path to MIDAS project.xml",
    )
    p.add_argument(
        "--mc-passes", type=int, default=5,
        help="MC dropout passes for Tumor CNN (default: 5). Use 0 for deterministic.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="Inference batch size (default: 4096)",
    )
    p.add_argument(
        "--no-qmap", action="store_true",
        help="Disable QMAP==4 override",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel worker processes (default: 2). "
             "Each worker loads the models independently, so RAM usage scales with workers.",
    )
    p.add_argument(
        "--output-summary", type=Path, default=None,
        help="Path for the summary CSV (default: nn_artifact_summary_YYYYMMDD_HHMMSS.csv "
             "next to project.xml)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("nn-artifact-batch")

    if not args.project_xml.exists():
        logger.error(f"project.xml not found: {args.project_xml}")
        sys.exit(1)

    # ── Collect all (subject, study) tasks ───────────────────────────────
    from artifactremoval.midas import MidasProject

    logger.info(f"Loading project: {args.project_xml}")
    project = MidasProject(args.project_xml)
    subjects = project.all_subject()

    tasks = []
    for subject in subjects:
        for study in subject.all_study():
            tasks.append((
                subject.subject_path,
                study.date,
                args.batch_size,
                args.mc_passes,
                args.no_qmap,
            ))

    if not tasks:
        logger.error("No subjects/studies found in project.")
        sys.exit(1)

    n_workers = min(args.workers, len(tasks))
    logger.info(
        f"Found {len(tasks)} stud{'y' if len(tasks) == 1 else 'ies'} "
        f"across {len(subjects)} subject(s)"
    )
    logger.info(
        f"Running with {n_workers} worker(s), mc-passes={args.mc_passes}"
    )

    # ── Run in parallel ───────────────────────────────────────────────────
    results = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(TUMOR_CNN_MODEL_PATH), str(NORMAL_BRAIN_MODEL_PATH)),
    ) as executor:
        future_to_task = {
            executor.submit(_run_task, task): task for task in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1

            subj = Path(result["subject_dir"]).name
            date = result["study_date"]

            if result["status"] == "success":
                pct = result.get("artifact_pct", "?")
                logger.info(
                    f"[{completed}/{len(tasks)}] {subj} | {date} "
                    f"— {pct}% artifact  ({result['duration_s']}s)"
                )
            else:
                logger.error(
                    f"[{completed}/{len(tasks)}] {subj} | {date} "
                    f"— FAILED: {result['error']}"
                )

    # ── Write summary CSV ─────────────────────────────────────────────────
    results.sort(key=lambda r: (str(r["subject_dir"]), r["study_date"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_summary or (
        args.project_xml.parent / f"nn_artifact_summary_{timestamp}.csv"
    )
    _write_summary(results, summary_path)

    n_ok = sum(1 for r in results if r["status"] == "success")
    n_err = sum(1 for r in results if r["status"] == "error")
    logger.info(f"Done — {n_ok} succeeded, {n_err} failed")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
