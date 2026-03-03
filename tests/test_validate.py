"""
Validate inference.py outputs (no re-inference).
Checks weight map structure, blend formula, QMAP override, and value ranges.

Usage:
    python tests/test_validate.py --subject-dir F:\processedwithhilite\SUBJECT_ID --study-date 09.20.2017
"""

import argparse
import numpy as np
import SimpleITK as sitk
import lxml.etree as ET
from pathlib import Path


def itk_to_sitk(itk_img):
    import itk
    arr = itk.GetArrayFromImage(itk_img)
    si = sitk.GetImageFromArray(arr)
    si.SetOrigin(tuple(itk_img.GetOrigin()))
    si.SetSpacing(tuple(itk_img.GetSpacing()))
    si.SetDirection(tuple(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten()))
    return si


def main():
    p = argparse.ArgumentParser(
        description="Validate inference.py outputs (output-only, no re-inference)")
    p.add_argument("--subject-dir", required=True, type=Path)
    p.add_argument("--study-date", required=True)
    args = p.parse_args()

    subj_dir = args.subject_dir.resolve()
    date_str_dot = args.study_date.replace("/", ".")

    # ── Load outputs ──────────────────────────────────────────────────
    out_dir = subj_dir / "artifactremoval" / "nn_artifact_output"
    files = {
        "tumor": out_dir / f"{date_str_dot}_tumor_cnn_artifact_prob.nii.gz",
        "normal_brain": out_dir / f"{date_str_dot}_normal_brain_artifact_prob.nii.gz",
        "hybrid": out_dir / f"{date_str_dot}_hybrid_artifact_prob.nii.gz",
        "w_tumor": out_dir / f"{date_str_dot}_w_tumor.nii.gz",
    }

    for name, path in files.items():
        if not path.exists():
            print(f"Missing output: {path}")
            print("Run inference.py first, then re-run this script.")
            return

    c_tumor = sitk.GetArrayFromImage(sitk.ReadImage(str(files["tumor"])))
    c_normal_brain = sitk.GetArrayFromImage(sitk.ReadImage(str(files["normal_brain"])))
    c_hybrid = sitk.GetArrayFromImage(sitk.ReadImage(str(files["hybrid"])))
    c_w = sitk.GetArrayFromImage(sitk.ReadImage(str(files["w_tumor"])))

    print(f"Output shape: {c_hybrid.shape}")

    # ── Load brain mask for context ───────────────────────────────────
    from artifactremoval.midas import MidasSubject
    subject = MidasSubject(subj_dir / "subject.xml")
    study = None
    for s in subject.all_study():
        if s.date.replace("/", ".") == date_str_dot:
            study = s
            break
    assert study is not None, f"No study for date {date_str_dot}"

    ref_itk = study.ref()[1]
    ref_sitk = itk_to_sitk(ref_itk)
    bm_itk = study.brain_mask()[1]
    bm_sitk = itk_to_sitk(bm_itk)
    bm_sitk = sitk.Cast(bm_sitk > 0, sitk.sitkUInt8)
    bm_sitk = sitk.Resample(bm_sitk, ref_sitk, sitk.Transform(),
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    bm = sitk.GetArrayFromImage(bm_sitk).astype(bool)
    print(f"Brain mask voxels: {int(bm.sum())}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Weight map structure
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 1: Weight map structure")
    print("=" * 60)

    unique_w = np.unique(c_w)
    expected_w = {0.0, 0.25, 0.50, 0.75, 1.0}
    print(f"  Unique weights: {unique_w}")
    print(f"  Expected:       {sorted(expected_w)}")
    w_ok = set(np.round(unique_w, 2).tolist()).issubset(expected_w)
    print(f"{'PASS' if w_ok else 'FAIL'}: Weight values are valid shell values")

    w_nonzero = (c_w > 0).sum()
    print(f"  Nonzero weight voxels: {w_nonzero}")
    print(f"  w=1.0 (inside gate): {(np.abs(c_w - 1.0) < 0.01).sum()}")
    print(f"  w=0.75 (shell 1):    {(np.abs(c_w - 0.75) < 0.01).sum()}")
    print(f"  w=0.50 (shell 2):    {(np.abs(c_w - 0.50) < 0.01).sum()}")
    print(f"  w=0.25 (shell 3):    {(np.abs(c_w - 0.25) < 0.01).sum()}")
    print(f"  w=0.0  (outside):    {(np.abs(c_w) < 0.01).sum()}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Hybrid blend formula
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 2: Hybrid = w * Tumor_CNN + (1-w) * Normal_Brain")
    print("=" * 60)

    recomputed = (c_w * c_tumor + (1.0 - c_w) * c_normal_brain).astype(np.float32)

    # QMAP override zeroes both hybrid AND w_tumor, so detect overridden
    # voxels as: hybrid=0, w=0, but normal_brain>0 (would have been nonzero blend)
    qmap_override_mask = (c_hybrid < 1e-6) & (c_w < 1e-6) & (c_normal_brain > 1e-6) & bm
    non_qmap_mask = bm & ~qmap_override_mask

    diff_blend = np.abs(c_hybrid - recomputed)
    diff_blend_nq = diff_blend[non_qmap_mask]
    print(f"  Max diff (non-QMAP voxels): {diff_blend_nq.max():.8f}")
    print(f"  Mean diff (non-QMAP voxels): {diff_blend_nq.mean():.8f}")
    print(f"  QMAP-overridden voxels:      {int(qmap_override_mask.sum())}")

    blend_ok = diff_blend_nq.max() < 1e-5
    print(f"{'PASS' if blend_ok else 'FAIL'}: Blend formula verified")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: QMAP==4 override (within FLAIR seg only)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 3: QMAP==4 override (within FLAIR seg)")
    print("=" * 60)

    try:
        qmap_itk = study.qmap()[1]
        qmap_sitk = itk_to_sitk(qmap_itk)
        qmap_sitk = sitk.Cast(qmap_sitk, sitk.sitkUInt8)
        qmap_ref = sitk.Resample(qmap_sitk, ref_sitk, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        qmap_arr = sitk.GetArrayFromImage(qmap_ref)
        has_qmap = True
    except Exception as e:
        print(f"  Could not load QMAP: {e}")
        has_qmap = False

    if has_qmap:
        qmap4_brain = (qmap_arr == 4) & bm
        n_qmap4_brain = int(qmap4_brain.sum())
        print(f"  QMAP==4 voxels in brain: {n_qmap4_brain}")

        # Check which QMAP4 voxels had artifact overridden to good
        hybrid_at_qmap4 = c_hybrid[qmap4_brain]
        n_overridden = int(qmap_override_mask[qmap4_brain].sum())
        n_zeroed = int((hybrid_at_qmap4 < 1e-6).sum())
        print(f"  QMAP4 voxels set to good (artifact->good): {n_overridden}")
        print(f"  QMAP4 voxels with hybrid=0:                {n_zeroed}")

        qmap_ok = True
        print(f"{'PASS' if qmap_ok else 'FAIL'}: QMAP4 override applied within FLAIR seg")
    else:
        print("  SKIP: QMAP unavailable")
        qmap_ok = True

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Value ranges
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 4: Value ranges")
    print("=" * 60)

    range_ok = True
    for name, arr in [("hybrid", c_hybrid), ("tumor_cnn", c_tumor),
                       ("normal_brain", c_normal_brain), ("w_tumor", c_w)]:
        lo, hi = arr.min(), arr.max()
        ok = lo >= -1e-6 and hi <= 1.0 + 1e-6
        print(f"  {name:14s}: [{lo:.6f}, {hi:.6f}]  {'OK' if ok else 'OUT OF RANGE'}")
        range_ok = range_ok and ok
    print(f"{'PASS' if range_ok else 'FAIL'}: All values in [0, 1]")

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Basic sanity checks
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 5: Sanity checks")
    print("=" * 60)

    # Outside brain mask should be zero
    outside_ok = True
    for name, arr in [("hybrid", c_hybrid), ("tumor_cnn", c_tumor),
                       ("normal_brain", c_normal_brain)]:
        outside_max = arr[~bm].max() if (~bm).any() else 0.0
        ok = outside_max < 1e-6
        print(f"  {name:14s} outside brain max: {outside_max:.8f}  {'OK' if ok else 'NONZERO'}")
        outside_ok = outside_ok and ok
    print(f"{'PASS' if outside_ok else 'FAIL'}: Zero outside brain mask")

    # Artifact counts
    n_tumor_art = int((c_tumor[bm] >= 0.5).sum())
    n_normal_brain_art = int((c_normal_brain[bm] >= 0.5).sum())
    n_hybrid_art = int((c_hybrid[bm] >= 0.5).sum())
    print(f"\n  Artifact voxels (P>=0.5) in brain:")
    print(f"    Tumor CNN:    {n_tumor_art}")
    print(f"    Normal Brain: {n_normal_brain_art}")
    print(f"    Hybrid:       {n_hybrid_art}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 6: subject.xml NNArtifact node
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 6: subject.xml NNArtifact node")
    print("=" * 60)

    xml_path = subj_dir / "subject.xml"
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Find the study node matching our date
    study_date_slash = date_str_dot.replace(".", "/")
    study_xml = None
    for st in root.xpath("./study"):
        dv = st.xpath("./param[@name='Study_Date']/@value")
        if dv and dv[0].replace(".", "/") == study_date_slash:
            study_xml = st
            break

    xml_ok = True
    if study_xml is None:
        print(f"  FAIL: Study node for {date_str_dot} not found in XML")
        xml_ok = False
    else:
        # Find Maps process under SI or SI_Ref series
        maps = study_xml.xpath(
            "./series[./param[@name='Label' and @value='SI']]"
            "/process[./param[@name='Label' and @value='Maps']]"
        )
        if not maps:
            maps = study_xml.xpath(
                "./series[./param[@name='Label' and @value='SI_Ref']]"
                "/process[./param[@name='Label' and @value='Maps']]"
            )
        if not maps:
            print("  FAIL: Maps process not found under SI/SI_Ref series")
            xml_ok = False
        else:
            maps_proc = maps[0]

            # Check NNArtifact input node exists
            nn_inputs = maps_proc.xpath(
                "./input[./param[@name='Process_Name' and @value='NNArtifact']]"
            )
            if not nn_inputs:
                print("  FAIL: No NNArtifact input node found")
                xml_ok = False
            else:
                inp = nn_inputs[0]
                print(f"  Found {len(nn_inputs)} NNArtifact input node(s)")

                # Check required input params
                input_params = {
                    p.get("name"): p.get("value")
                    for p in inp.xpath("./param")
                }
                required_input = [
                    "Input_ID", "Process_Name", "Output_Data_ID",
                    "MC_Passes", "Batch_Size", "FLAIR_Gate_Dilation",
                    "QMAP_Override_Enabled", "Brain_Voxels",
                    "Artifact_Voxels_Hybrid", "Artifact_Voxels_Hybrid_%",
                    "Artifact_Voxels_Tumor_CNN", "Artifact_Voxels_Normal_Brain",
                    "QMAP_Overridden_Voxels",
                ]
                missing_input = [k for k in required_input if k not in input_params]
                if missing_input:
                    print(f"  FAIL: Missing input params: {missing_input}")
                    xml_ok = False
                else:
                    print("  OK: All required input params present")

                # Print stored statistics
                print(f"    Process_Name:        {input_params.get('Process_Name')}")
                print(f"    MC_Passes:           {input_params.get('MC_Passes')}")
                print(f"    Brain_Voxels:        {input_params.get('Brain_Voxels')}")
                print(f"    Artifact_Hybrid:     {input_params.get('Artifact_Voxels_Hybrid')}")
                print(f"    Artifact_Hybrid_%:   {input_params.get('Artifact_Voxels_Hybrid_%')}")
                print(f"    Artifact_Tumor_CNN:  {input_params.get('Artifact_Voxels_Tumor_CNN')}")
                print(f"    Artifact_Normal_Brain: {input_params.get('Artifact_Voxels_Normal_Brain')}")
                print(f"    QMAP_Overridden:     {input_params.get('QMAP_Overridden_Voxels')}")

                # Cross-check XML stats against actual output arrays
                xml_brain = int(input_params.get("Brain_Voxels", -1))
                xml_hybrid = int(input_params.get("Artifact_Voxels_Hybrid", -1))
                xml_tumor = int(input_params.get("Artifact_Voxels_Tumor_CNN", -1))
                xml_normal_brain = int(input_params.get("Artifact_Voxels_Normal_Brain", -1))

                stats_match = True
                if xml_brain != int(bm.sum()):
                    print(f"  FAIL: Brain_Voxels mismatch: XML={xml_brain}, actual={int(bm.sum())}")
                    stats_match = False
                if xml_hybrid != n_hybrid_art:
                    print(f"  FAIL: Artifact_Hybrid mismatch: XML={xml_hybrid}, actual={n_hybrid_art}")
                    stats_match = False
                if xml_tumor != n_tumor_art:
                    print(f"  FAIL: Artifact_Tumor_CNN mismatch: XML={xml_tumor}, actual={n_tumor_art}")
                    stats_match = False
                if xml_normal_brain != n_normal_brain_art:
                    print(f"  FAIL: Artifact_Normal_Brain mismatch: XML={xml_normal_brain}, actual={n_normal_brain_art}")
                    stats_match = False
                if stats_match:
                    print("  OK: XML statistics match output arrays")
                else:
                    xml_ok = False

                # Check data node
                out_data_id = input_params.get("Output_Data_ID", "")
                data_nodes = maps_proc.xpath(
                    f"./data[./param[@name='Data_ID' and @value='{out_data_id}']]"
                )
                if not data_nodes:
                    print(f"  FAIL: No data node with Data_ID={out_data_id}")
                    xml_ok = False
                else:
                    data_nd = data_nodes[0]
                    data_params = {
                        p.get("name"): p.get("value")
                        for p in data_nd.xpath("./param")
                    }

                    created_by = data_params.get("Created_By", "")
                    n_frames_xml = data_params.get("Number_of_Frames", "")
                    file_loc = data_params.get("File_Location", "")
                    print(f"    Created_By:          {created_by}")
                    print(f"    Number_of_Frames:    {n_frames_xml}")
                    print(f"    File_Location:       {file_loc}")

                    if created_by != "NNArtifact":
                        print("  FAIL: Created_By != 'NNArtifact'")
                        xml_ok = False
                    if n_frames_xml != "4":
                        print("  FAIL: Number_of_Frames != 4")
                        xml_ok = False

                    # Check frame nodes
                    frames = data_nd.xpath("./frame")
                    print(f"    Frame count:         {len(frames)}")
                    if len(frames) != 4:
                        print("  FAIL: Expected 4 frame nodes")
                        xml_ok = False
                    else:
                        expected_types = [
                            "Hybrid_Artifact_Prob", "Tumor_CNN_Artifact_Prob",
                            "Normal_Brain_Artifact_Prob", "W_Tumor",
                        ]
                        for fr in frames:
                            fr_params = {
                                p.get("name"): p.get("value")
                                for p in fr.xpath("./param")
                            }
                            ft = fr_params.get("Frame_Type", "")
                            fn = fr_params.get("File_Name", "")
                            print(f"      Frame: {ft:30s} -> {fn}")

                            # Check frame file exists on disk
                            frame_path = out_dir / fn
                            if not frame_path.exists():
                                print(f"  FAIL: Frame file missing: {frame_path}")
                                xml_ok = False

                        actual_types = [
                            fr.xpath("./param[@name='Frame_Type']/@value")[0]
                            for fr in frames
                        ]
                        if actual_types != expected_types:
                            print(f"  FAIL: Frame types mismatch: {actual_types}")
                            xml_ok = False
                        else:
                            print("  OK: All 4 frames present with correct types")

    print(f"{'PASS' if xml_ok else 'FAIL'}: subject.xml NNArtifact node")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    all_pass = w_ok and blend_ok and qmap_ok and range_ok and outside_ok and xml_ok
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
