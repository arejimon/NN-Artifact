#!/usr/bin/env python
"""
NN-Artifact: NAWM-normalized CHO/NAA with hybrid artifact masking.

Creates a spatially-varying threshold map and applies it to CHO/NAA, producing
NAWM-normalized CHO/NAA volumes with artifacts removed.

Requires inference.py (or batch_inference.py) to have been run first, as it
reads the hybrid artifact probability and FLAIR segmentation outputs.

Usage:
    python create_chonaanorm.py --subject-dir /path/to/subject --study-date 01.31.2018
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

# Add src/ to path so artifactremoval package is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ── ANTs tissue label IDs ─────────────────────────────────────────────────────
LBL_GM  = 2
LBL_WM  = 3
LBL_DGM = 4
LBL_BS  = 5
LBL_CER = 6

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("nn-artifact-chonaa")


# ── Grid / resampling helpers ─────────────────────────────────────────────────

def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def _resample(moving: sitk.Image, ref: sitk.Image, is_mask: bool = False) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    out_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    return sitk.Resample(moving, ref, sitk.Transform(), interp, 0.0, out_type)


# ── CHO/NAA loading ───────────────────────────────────────────────────────────

def _fuzzy_match(target, candidates):
    """Case-insensitive alphanumeric match with substring fallback."""
    t = "".join(c for c in str(target).lower() if c.isalnum())
    for c in candidates:
        if "".join(ch for ch in str(c).lower() if ch.isalnum()) == t:
            return c
    for c in candidates:
        norm = "".join(ch for ch in str(c).lower() if ch.isalnum())
        if t in norm or norm in t:
            return c
    return None


def _itk_to_sitk(itk_img):
    import itk
    arr = itk.GetArrayFromImage(itk_img)
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(tuple(itk_img.GetOrigin()))
    img.SetSpacing(tuple(itk_img.GetSpacing()))
    img.SetDirection(
        tuple(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    )
    return img


def load_chonaa(study) -> sitk.Image:
    """
    Load CHO/NAA ratio map from NNFit xarray dataset.

    Searches all data variables for a dimension coordinate matching 'cho/naa'
    using case-insensitive fuzzy matching.
    """
    from artifactremoval.midas import NNFitDataset

    for og in (False, True):
        try:
            nnfit = NNFitDataset(study, og=og)
            xr_ds = nnfit.open_ds()

            for var_name in xr_ds.data_vars:
                var = xr_ds[var_name]
                for dim in var.dims:
                    if dim not in var.coords:
                        continue
                    labels = list(map(str, var.coords[dim].values))
                    key = _fuzzy_match("cho/naa", labels)
                    if key is None:
                        continue
                    arr = np.nan_to_num(
                        var.sel({dim: key}).values, copy=False
                    ).astype(np.float32)
                    img = _itk_to_sitk(nnfit.ndarray_to_itk(arr))
                    logger.info(f"Loaded CHO/NAA: {var_name}[{dim}={key!r}]")
                    return img
        except Exception as e:
            logger.debug(f"NNFit (og={og}) failed: {e}")

    raise RuntimeError(
        "Could not load CHO/NAA ratio from NNFit. "
        "Ensure the nnfit xarray dataset exists for this study."
    )


# ── NAWM normalization ────────────────────────────────────────────────────────

def normalize_to_nawm(chonaa: sitk.Image, nawm_mask: sitk.Image) -> sitk.Image:
    """Divide CHO/NAA by mean value within the NAWM mask."""
    nawm = sitk.Cast(nawm_mask > 0, sitk.sitkUInt8)
    if not _same_grid(nawm, chonaa):
        nawm = _resample(nawm, chonaa, is_mask=True)

    nawm_arr = sitk.GetArrayFromImage(nawm).astype(bool)
    chonaa_arr = sitk.GetArrayFromImage(sitk.Cast(chonaa, sitk.sitkFloat32))
    vals = chonaa_arr[nawm_arr & np.isfinite(chonaa_arr) & (chonaa_arr > 0)]

    if vals.size == 0:
        raise RuntimeError(
            "No valid NAWM voxels found. "
            "Check that the NAWM mask overlaps the CHO/NAA volume."
        )

    mean_val = float(vals.mean())
    logger.info(f"NAWM mean CHO/NAA = {mean_val:.4f} ({vals.size} voxels)")

    out = sitk.Cast(chonaa, sitk.sitkFloat32) / (mean_val + 1e-6)
    out.CopyInformation(chonaa)
    return out


# ── Threshold map ─────────────────────────────────────────────────────────────

def _outside_distance_vox(seg_bool: np.ndarray) -> np.ndarray:
    """Euclidean distance transform (voxel units) from outside a binary mask."""
    d = distance_transform_edt(~seg_bool).astype(np.float32)
    d[seg_bool] = 0.0
    return d


def _apply_cereb_inferior_penalty(
    thrA: np.ndarray,
    segA: np.ndarray,
    cer_drop: float = 0.19,
    min_vox: int = 200,
) -> np.ndarray:
    """
    Reduce threshold on all intracranial slices at or below the most superior
    slice that contains cerebellum/brainstem (prevents false-negative artifacts
    in posterior fossa).
    """
    intracranial = segA > 0
    cer_mask = (segA == LBL_CER) | (segA == LBL_BS)
    z_counts = cer_mask.reshape(segA.shape[0], -1).sum(axis=1)
    z_has_cer = z_counts >= int(min_vox)

    if not np.any(z_has_cer):
        return thrA

    z_top = int(np.where(z_has_cer)[0].max())  # most superior cereb slice
    thrA[:z_top + 1][intracranial[:z_top + 1]] -= float(cer_drop)
    return thrA


def make_threshold_map(
    hybrid_sitk: sitk.Image,
    ants_seg_sitk: sitk.Image,
    seg_in_ref_sitk: sitk.Image,
) -> sitk.Image:
    """
    Build a spatially-varying Float32 threshold map on the hybrid artifact
    probability grid.

    The map encodes: keep voxel if hybrid_prob <= threshold.
    Lower threshold = stricter acceptance (fewer accepted voxels).

    Strategy:
    - Inside FLAIR seg:          threshold = 0.50
    - Within 2 voxels outside:   threshold = 0.40
    - Beyond 2 voxels outside:   threshold = 0.10
    - Top/bottom 20% of brain:   threshold = ~0.00001 (very strict)
    - Cerebellum/inferior:       threshold reduced by 0.19
    - Tissue adjustments:        brainstem/cereb get -0.02
    """
    extent_labels = np.array([LBL_GM, LBL_WM, LBL_DGM, LBL_BS, LBL_CER], dtype=np.int16)

    # Resample ANTs tissue seg to hybrid grid (label-safe)
    seg = sitk.Cast(ants_seg_sitk, sitk.sitkUInt16)
    if not _same_grid(seg, hybrid_sitk):
        seg = sitk.Resample(
            seg, hybrid_sitk, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
        )
    segA = sitk.GetArrayFromImage(seg).astype(np.int16)   # (Z,Y,X)
    intracranial = segA > 0

    # Resample FLAIR seg in REF to hybrid grid (label-safe)
    segf = sitk.Cast(seg_in_ref_sitk, sitk.sitkUInt8)
    if not _same_grid(segf, hybrid_sitk):
        segf = sitk.Resample(
            segf, hybrid_sitk, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
    segF = (sitk.GetArrayFromImage(segf) > 0) & intracranial  # (Z,Y,X) bool

    # ── FLAIR-proximity thresholds ──────────────────────────────────────────
    d_out_vox = _outside_distance_vox(segF)

    thr_flair = np.full(segA.shape, 0.10, dtype=np.float32)   # beyond 2 vox
    thr_flair[segF] = 0.50                                      # inside FLAIR
    ring1 = (~segF) & (d_out_vox > 0.0) & (d_out_vox <= 2.0)
    thr_flair[ring1] = 0.40                                     # 0-2 vox outside
    thr_flair = np.where(intracranial, thr_flair, 0.0).astype(np.float32)

    # ── Z-extreme strictness (top/bottom 20% of brain extent) ──────────────
    extent = np.isin(segA, extent_labels)
    if np.any(extent):
        z_any = extent.reshape(extent.shape[0], -1).any(axis=1)
        z_idx = np.where(z_any)[0]
        zmin, zmax = int(z_idx.min()), int(z_idx.max())
        zlen = max(1, zmax - zmin)

        z = np.arange(extent.shape[0], dtype=np.float32)
        zfrac_inf = (z - zmin) / float(zlen)   # 0=bottom, 1=top
        zfrac_sup = (zmax - z) / float(zlen)   # 0=top, 1=bottom

        p = np.zeros_like(z, dtype=np.float32)
        p[zfrac_inf <= 0.2] = 1.0   # bottom 20%
        p[zfrac_sup <= 0.2] = 1.0   # top 20%

        thr_z = (0.2 + p * (0.00001 - 0.2)).astype(np.float32)  # thr_mid=0.2, thr_extreme≈0
        thr_z_map = (
            np.broadcast_to(thr_z[:, None, None], segA.shape)
            .astype(np.float32).copy()
        )
        thr_z_map = np.where(intracranial, thr_z_map, 0.0).astype(np.float32)
    else:
        thr_z_map = np.where(intracranial, 0.2, 0.0).astype(np.float32)

    # ── Combine: min of the two, but protect FLAIR regions ─────────────────
    thrA = np.minimum(thr_z_map, thr_flair).astype(np.float32)
    thrA[ring1] = thr_flair[ring1]   # protect ring around FLAIR
    thrA[segF]  = thr_flair[segF]    # protect inside FLAIR

    # ── Tissue adjustments ──────────────────────────────────────────────────
    for lbl, delta in {LBL_BS: -0.02, LBL_CER: -0.02}.items():
        thrA[segA == lbl] += float(delta)

    # ── Cerebellum inferior penalty ─────────────────────────────────────────
    thrA = _apply_cereb_inferior_penalty(thrA, segA)

    # ── Clamp and wrap ──────────────────────────────────────────────────────
    thrA = np.clip(thrA, 0.0, 0.975).astype(np.float32)
    out = sitk.GetImageFromArray(thrA)
    out.CopyInformation(hybrid_sitk)
    return sitk.Cast(out, sitk.sitkFloat32)


# ── File discovery ────────────────────────────────────────────────────────────

def _find_flair_seg_in_ref(
    subject_dir: Path,
    date_str_dot: str,
    nn_output_dir: Path,
) -> Path:
    """
    Locate the FLAIR segmentation registered to REF space.
    Checks the nn_artifact_output directory first, then the legacy notebook cache.
    """
    primary = nn_output_dir / f"{date_str_dot}_flair_seg_in_ref.nii.gz"
    if primary.exists():
        return primary

    legacy = (
        subject_dir / "artifactremoval" / "_cache_hybrid"
        / date_str_dot / "seg_in_ref.nii.gz"
    )
    if legacy.exists():
        logger.info(f"Using legacy cache: {legacy}")
        return legacy

    raise FileNotFoundError(
        f"FLAIR seg in REF not found. Run inference.py first.\n"
        f"  Expected: {primary}\n"
        f"  Legacy:   {legacy}"
    )


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_chonaanorm(
    subject_dir: Path,
    study_date: str,
    output_dir: Path = None,
    no_qmap: bool = False,
) -> dict:
    """
    Create NAWM-normalized CHO/NAA map with hybrid artifact masking.

    Parameters
    ----------
    subject_dir : Path
        MIDAS subject folder (must contain subject.xml).
    study_date : str
        Study date, e.g. '01.31.2018' or '01/31/2018'.
    output_dir : Path, optional
        Where to save outputs. Defaults to
        {subject_dir}/artifactremoval/nn_artifact_output/.
    no_qmap : bool
        Skip the QMAP>=4 filtered output.

    Returns
    -------
    dict
        Keys: subject_dir, study_date, output_dir.
    """
    from artifactremoval.midas import MidasSubject

    subject_dir = Path(subject_dir)
    date_str_dot = study_date.replace("/", ".")
    date_str_us  = date_str_dot.replace(".", "_")
    tag = f"{subject_dir.name} | {date_str_dot}"

    nn_output_dir = subject_dir / "artifactremoval" / "nn_artifact_output"
    out_dir = output_dir or nn_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load hybrid artifact prob ──────────────────────────────────────────
    hybrid_nii = nn_output_dir / f"{date_str_dot}_hybrid_artifact_prob.nii.gz"
    if not hybrid_nii.exists():
        raise FileNotFoundError(
            f"Hybrid artifact prob not found: {hybrid_nii}\n"
            "Run inference.py first."
        )
    hybrid_sitk = sitk.Cast(sitk.ReadImage(str(hybrid_nii)), sitk.sitkFloat32)
    logger.info(f"[{tag}] Loaded hybrid artifact probability")

    # ── Load FLAIR seg in REF space ────────────────────────────────────────
    flair_seg_path = _find_flair_seg_in_ref(subject_dir, date_str_dot, nn_output_dir)
    seg_in_ref_sitk = sitk.ReadImage(str(flair_seg_path))
    logger.info(f"[{tag}] Loaded FLAIR seg in REF: {flair_seg_path.name}")

    # ── Load ANTs tissue segmentation ─────────────────────────────────────
    ants_nii = subject_dir / "ants" / f"segmentations_{date_str_us}.nii.gz"
    if not ants_nii.exists():
        raise FileNotFoundError(f"ANTs segmentation not found: {ants_nii}")
    ants_seg_sitk = sitk.ReadImage(str(ants_nii))
    logger.info(f"[{tag}] Loaded ANTs segmentation")

    # ── Load NAWM mask ─────────────────────────────────────────────────────
    nawm_path = subject_dir / "mri" / f"{date_str_dot}_nawm_mask.nii.gz"
    if not nawm_path.exists():
        raise FileNotFoundError(
            f"NAWM mask not found: {nawm_path}\n"
            "The NAWM mask is required for normalization."
        )
    nawm_sitk = sitk.ReadImage(str(nawm_path))
    logger.info(f"[{tag}] Loaded NAWM mask")

    # ── Load CHO/NAA from MIDAS/NNFit ─────────────────────────────────────
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
            f"Available: {available}"
        )

    logger.info(f"[{tag}] Loading CHO/NAA map...")
    chonaa_sitk = load_chonaa(study)

    # ── Cerebellum exclusion mask (on CHO/NAA grid) ────────────────────────
    cereb_bin = sitk.Cast(
        (ants_seg_sitk == LBL_CER) | (ants_seg_sitk == LBL_BS),
        sitk.sitkUInt8,
    )
    outside_cereb = sitk.Cast(cereb_bin == 0, sitk.sitkUInt8)
    if not _same_grid(outside_cereb, chonaa_sitk):
        outside_cereb = _resample(outside_cereb, chonaa_sitk, is_mask=True)

    # ── Normalize CHO/NAA to NAWM mean ────────────────────────────────────
    logger.info(f"[{tag}] Normalizing CHO/NAA to NAWM mean...")
    chonaa_norm = normalize_to_nawm(chonaa_sitk, nawm_sitk)

    # Remove cerebellum from normalized map
    chonaa_norm = sitk.Mask(sitk.Cast(chonaa_norm, sitk.sitkFloat32), outside_cereb)

    # ── Build threshold map ────────────────────────────────────────────────
    logger.info(f"[{tag}] Building spatially-varying threshold map...")

    # Resample all spatial inputs to CHO/NAA spectral grid
    hybrid_on_chonaa = _resample(hybrid_sitk, chonaa_norm, is_mask=False)
    segref_on_chonaa = _resample(
        sitk.Cast(seg_in_ref_sitk, sitk.sitkUInt8), chonaa_norm, is_mask=True
    )
    ants_on_chonaa = sitk.Resample(
        sitk.Cast(ants_seg_sitk, sitk.sitkUInt16), chonaa_norm, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
    )

    thr_map = make_threshold_map(hybrid_on_chonaa, ants_on_chonaa, segref_on_chonaa)

    # ── Apply threshold mask to normalized CHO/NAA ─────────────────────────
    hybrid_mask = sitk.Cast(hybrid_on_chonaa <= thr_map, sitk.sitkUInt8)
    hybrid_mask = sitk.And(hybrid_mask, outside_cereb)
    chonaa_hybrid = sitk.Mask(sitk.Cast(chonaa_norm, sitk.sitkFloat32), hybrid_mask)

    # ── Save outputs ───────────────────────────────────────────────────────
    thr_path    = out_dir / f"{date_str_dot}_hybrid_thrmap.nii.gz"
    hybrid_path = out_dir / f"{date_str_dot}_chonaa_hybridthr_norm.nii.gz"

    sitk.WriteImage(thr_map, str(thr_path))
    sitk.WriteImage(sitk.Cast(chonaa_hybrid, sitk.sitkFloat32), str(hybrid_path))
    logger.info(f"[{tag}] Saved: {thr_path.name}")
    logger.info(f"[{tag}] Saved: {hybrid_path.name}")

    # ── QMAP>=4 filtered output (optional) ────────────────────────────────
    if not no_qmap:
        try:
            import itk
            qmap_itk = study.qmap()[1]
            qmap_sitk = _itk_to_sitk(qmap_itk)
            qmap_on_chonaa = sitk.Resample(
                sitk.Cast(qmap_sitk, sitk.sitkUInt8), chonaa_norm, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
            qmap4_mask = sitk.And(
                sitk.Cast(qmap_on_chonaa >= 4, sitk.sitkUInt8),
                outside_cereb,
            )
            chonaa_q4 = sitk.Mask(
                sitk.Cast(chonaa_norm, sitk.sitkFloat32), qmap4_mask
            )
            q4_path = out_dir / f"{date_str_dot}_chonaa_qmap4_norm.nii.gz"
            sitk.WriteImage(sitk.Cast(chonaa_q4, sitk.sitkFloat32), str(q4_path))
            logger.info(f"[{tag}] Saved: {q4_path.name}")
        except Exception as e:
            logger.warning(f"[{tag}] Could not create QMAP4 map: {e}")

    logger.info(f"[{tag}] Done")
    return {
        "subject_dir": subject_dir,
        "study_date": date_str_dot,
        "output_dir": out_dir,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "NN-Artifact: create NAWM-normalized CHO/NAA filtered by hybrid artifact mask.\n"
            "Requires inference.py to have been run first."
        ),
    )
    p.add_argument(
        "--subject-dir", required=True, type=Path,
        help="Path to MIDAS subject folder (contains subject.xml)",
    )
    p.add_argument(
        "--study-date", required=True,
        help="Study date, e.g. '01.31.2018' or '01/31/2018'",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output directory "
            "(default: {subject-dir}/artifactremoval/nn_artifact_output/)"
        ),
    )
    p.add_argument(
        "--no-qmap", action="store_true",
        help="Skip the QMAP>=4 filtered CHO/NAA output",
    )
    return p.parse_args()


def main():
    args = parse_args()

    subj_dir = args.subject_dir.resolve()
    if not subj_dir.exists():
        logger.error(f"Subject directory does not exist: {subj_dir}")
        sys.exit(1)

    try:
        run_chonaanorm(
            subject_dir=subj_dir,
            study_date=args.study_date,
            output_dir=args.output_dir,
            no_qmap=args.no_qmap,
        )
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
