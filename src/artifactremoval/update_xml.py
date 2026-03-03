"""
Update subject.xml with NN-Artifact results node.

Creates input + data + frame nodes under the SI series Maps process,
similar to how QMAPS stores its results. Idempotent — replaces existing
NNArtifact nodes if re-run.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import lxml.etree as ET
from pymidas.common.libxml import SubjectXml

logger = logging.getLogger(__name__)

PROCESS_NAME = "NNArtifact"


def _find_maps_process(study_node):
    """Find the Maps process node under the SI series for a given study."""
    # Look for SI series → Maps process
    maps = study_node.xpath(
        "./series[./param[@name='Label' and @value='SI']]"
        "/process[./param[@name='Label' and @value='Maps']]"
    )
    if maps:
        return maps[0]

    # Fallback: try SI_Ref series → Maps process
    maps = study_node.xpath(
        "./series[./param[@name='Label' and @value='SI_Ref']]"
        "/process[./param[@name='Label' and @value='Maps']]"
    )
    if maps:
        return maps[0]

    return None


def _get_process_id(process_node):
    """Extract Process_ID from a process node."""
    ids = process_node.xpath("./param[@name='Process_ID']/@value")
    return ids[0] if ids else None


def _remove_existing_nnartifact(process_node):
    """Remove any existing NNArtifact input + data nodes."""
    # Find all NNArtifact input nodes
    inputs = process_node.xpath(
        f"./input[./param[@name='Process_Name' and @value='{PROCESS_NAME}']]"
    )
    for inp in inputs:
        # Find and remove the associated data node via Output_Data_ID
        out_ids = inp.xpath("./param[@name='Output_Data_ID']/@value")
        for out_id in out_ids:
            data_nodes = process_node.xpath(
                f"./data[./param[@name='Data_ID' and @value='{out_id}']]"
            )
            for d in data_nodes:
                process_node.remove(d)
                logger.info(f"Removed existing NNArtifact data node: {out_id}")
        process_node.remove(inp)
        logger.info("Removed existing NNArtifact input node")


def _get_next_subnode_id(process_node, process_id):
    """Get the next available sub-ID under this process node."""
    max_suffix = 0
    for node in process_node:
        for param in node.xpath("./param"):
            name = param.get("name", "")
            val = param.get("value", "")
            if name in ("Data_ID", "Input_ID") and val.startswith(process_id + "."):
                suffix = val[len(process_id) + 1:]
                try:
                    max_suffix = max(max_suffix, int(suffix))
                except ValueError:
                    pass
    return max_suffix + 1


def _make_param(parent, name, value):
    """Create a <param name="..." value="..."/> element."""
    ET.SubElement(parent, "param", name=str(name), value=str(value))


def _get_spatial_params(process_node):
    """Extract spatial parameters from the existing Maps data node (e.g. QMAP or REF)."""
    spatial_keys = [
        "Spatial_Points_1", "Spatial_Points_2", "Spatial_Points_3",
        "Slice_Thickness", "Pixel_Spacing_1", "Pixel_Spacing_2", "Pixel_Spacing_3",
        "FOV_1", "FOV_2", "FOV_3",
        "Image_Position_X", "Image_Position_Y", "Image_Position_Z",
        "Image_Orientation_Xr", "Image_Orientation_Yr", "Image_Orientation_Zr",
        "Image_Orientation_Xc", "Image_Orientation_Yc", "Image_Orientation_Zc",
    ]
    params = {}

    # Try to get spatial params from any existing data node in this process
    for data_node in process_node.xpath("./data"):
        for key in spatial_keys:
            if key not in params:
                vals = data_node.xpath(f"./param[@name='{key}']/@value")
                if vals:
                    params[key] = vals[0]

    return params


def update_subject_xml(
    subject_xml_path,
    study_date_dot,
    output_dir,
    date_prefix,
    mc_passes,
    batch_size,
    flair_gate_dilation,
    qmap_override_enabled,
    n_brain_voxels,
    n_artifact_hybrid,
    n_artifact_tumor_cnn,
    n_artifact_normal_brain,
    n_qmap_overridden,
):
    """
    Add NNArtifact results to subject.xml.

    Parameters
    ----------
    subject_xml_path : Path
        Path to subject.xml.
    study_date_dot : str
        Study date in dot format (e.g. "03.28.2018").
    output_dir : Path
        Directory containing the output NIfTI files.
    date_prefix : str
        Filename prefix (e.g. "03.28.2018").
    mc_passes : int
        Number of MC dropout passes used.
    batch_size : int
        Inference batch size.
    flair_gate_dilation : int
        FLAIR gate dilation in voxels.
    qmap_override_enabled : bool
        Whether QMAP==4 override was active.
    n_brain_voxels : int
        Number of brain mask voxels.
    n_artifact_hybrid : int
        Artifact voxels (P>=0.5) in hybrid output.
    n_artifact_tumor_cnn : int
        Artifact voxels in Tumor CNN output.
    n_artifact_normal_brain : int
        Artifact voxels in Normal-Brain CNN output.
    n_qmap_overridden : int
        Voxels where QMAP override reclassified artifact as good.
    """
    subject_xml_path = Path(subject_xml_path)
    output_dir = Path(output_dir)

    # Convert date format for matching (dot → slash)
    study_date_slash = study_date_dot.replace(".", "/")

    # Load subject.xml
    subj_xml = SubjectXml(str(subject_xml_path))
    tree = subj_xml.tree
    root = tree.getroot()

    # Find the correct study node by date
    study_node = None
    for study in root.xpath("./study"):
        date_vals = study.xpath("./param[@name='Study_Date']/@value")
        if date_vals:
            if date_vals[0].replace(".", "/") == study_date_slash:
                study_node = study
                break

    if study_node is None:
        raise ValueError(f"Study with date '{study_date_dot}' not found in {subject_xml_path}")

    # Find the Maps process under SI series
    maps_process = _find_maps_process(study_node)
    if maps_process is None:
        raise ValueError("Maps process not found under SI or SI_Ref series")

    process_id = _get_process_id(maps_process)
    logger.info(f"Found Maps process: {process_id}")

    # Remove existing NNArtifact nodes (idempotent)
    _remove_existing_nnartifact(maps_process)

    # Get spatial params from existing data nodes
    spatial_params = _get_spatial_params(maps_process)

    # Compute relative path from subject dir to output dir
    subj_dir = subject_xml_path.parent
    try:
        rel_path = output_dir.resolve().relative_to(subj_dir.resolve())
        file_location = ".\\" + str(rel_path)
    except ValueError:
        file_location = str(output_dir.resolve())

    # Timestamps
    now = datetime.now()
    date_str = now.strftime("%m/%d/%y")
    time_str = now.strftime("%I:%M:%S %p")

    # ── Create input node ─────────────────────────────────────────
    next_id = _get_next_subnode_id(maps_process, process_id)
    input_id = f"{process_id}.{next_id}"

    input_node = ET.SubElement(maps_process, "input")
    _make_param(input_node, "Input_ID", input_id)
    _make_param(input_node, "Process_Name", PROCESS_NAME)
    _make_param(input_node, "Creation_Date", date_str)
    _make_param(input_node, "Creation_Time", time_str)

    # Data node ID
    data_id = f"{process_id}.{next_id + 1}"
    _make_param(input_node, "Output_Data_ID", data_id)

    # Inference parameters
    _make_param(input_node, "MC_Passes", mc_passes)
    _make_param(input_node, "Batch_Size", batch_size)
    _make_param(input_node, "FLAIR_Gate_Dilation", flair_gate_dilation)
    _make_param(input_node, "QMAP_Override_Enabled", str(qmap_override_enabled))

    # Result statistics
    _make_param(input_node, "Brain_Voxels", n_brain_voxels)
    artifact_pct = round(100.0 * n_artifact_hybrid / max(n_brain_voxels, 1), 1)
    _make_param(input_node, "Artifact_Voxels_Hybrid", n_artifact_hybrid)
    _make_param(input_node, "Artifact_Voxels_Hybrid_%", artifact_pct)
    _make_param(input_node, "Artifact_Voxels_Tumor_CNN", n_artifact_tumor_cnn)
    _make_param(input_node, "Artifact_Voxels_Normal_Brain", n_artifact_normal_brain)
    _make_param(input_node, "QMAP_Overridden_Voxels", n_qmap_overridden)

    # ── Create data node ──────────────────────────────────────────
    data_node = ET.SubElement(maps_process, "data")
    _make_param(data_node, "Data_ID", data_id)
    _make_param(data_node, "Creation_Date", date_str)
    _make_param(data_node, "Creation_Time", time_str)
    _make_param(data_node, "File_Location", file_location)
    _make_param(data_node, "Data_Representation", "float")
    _make_param(data_node, "Byte_Order", "LittleEndian")
    _make_param(data_node, "Created_By", PROCESS_NAME)
    _make_param(data_node, "Number_of_Frames", "4")
    _make_param(data_node, "Input_Node_ID", input_id)

    # Copy spatial parameters from existing data
    for key, val in spatial_params.items():
        _make_param(data_node, key, val)

    # ── Create frame nodes ────────────────────────────────────────
    frames = [
        ("Hybrid_Artifact_Prob",       f"{date_prefix}_hybrid_artifact_prob.nii.gz",       "Top"),
        ("Tumor_CNN_Artifact_Prob",    f"{date_prefix}_tumor_cnn_artifact_prob.nii.gz",    "Other"),
        ("Normal_Brain_Artifact_Prob", f"{date_prefix}_normal_brain_artifact_prob.nii.gz", "Other"),
        ("W_Tumor",                    f"{date_prefix}_w_tumor.nii.gz",                   "Other"),
    ]

    for i, (frame_type, filename, viewer) in enumerate(frames):
        frame_id = f"{data_id}.{i + 1}"
        frame_node = ET.SubElement(data_node, "frame")
        _make_param(frame_node, "Frame_ID", frame_id)
        _make_param(frame_node, "Frame_Type", frame_type)
        _make_param(frame_node, "File_Name", filename)
        _make_param(frame_node, "Creation_Date", date_str)
        _make_param(frame_node, "Creation_Time", time_str)
        _make_param(frame_node, "Register_Viewer", viewer)

    # ── Save ──────────────────────────────────────────────────────
    subj_xml.write_file(str(subject_xml_path))
    logger.info(f"Updated subject.xml with NNArtifact node (data_id={data_id})")

    return data_id
