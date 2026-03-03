# NN-Artifact

Hybrid artifact detection for MR spectroscopy data. Combines two expert neural networks (Tumor CNN and Normal-Brain CNN) with spatial voxel-shell gating based on FLAIR hyperintensity segmentation.

## How It Works

1. **Tumor CNN** (3-channel CNN): processes raw spectra + water reference + fitted spectra → P(artifact)
2. **Normal-Brain CNN** (1-channel NN): processes raw spectra → P(artifact)
3. **Hybrid voxel-shell gating**: blends both predictions spatially using FLAIR segmentation:
   - Inside FLAIR hyperintensity (dilated 2 voxels): fully trust Tumor CNN (w=1.0)
   - 1 voxel outside: w=0.75
   - 2 voxels outside: w=0.50
   - 3 voxels outside: w=0.25
   - Beyond 3 voxels: fully trust Normal-Brain CNN (w=0.0)
   - Cerebellum outside gate: always use Normal-Brain CNN
   - Optional QMAP==4 override: force artifact probability to 0.0, applied **only within the FLAIR segmentation** (not the dilated gate)

## Requirements

- Docker
- A MIDAS-processed subject folder with:
  - `subject.xml` at the root
  - Spectral data (SI, SIREF, nnfit)
  - FLAIR hyperintensity segmentation in `mri/{date}_autoseg/`
  - (Optional) Cerebellum segmentation
  - (Optional) QMAP data

## Quick Start

### Build the Docker image

```bash
# CPU version
docker build -t nn-artifact .

# GPU version (requires nvidia-docker)
docker build -f Dockerfile.gpu -t nn-artifact-gpu .
```

### Run inference

```bash
docker run -v /path/to/subject:/data/subject nn-artifact \
    --subject-dir /data/subject \
    --study-date 01.31.2018
```

### With GPU

```bash
docker run --gpus all -v /path/to/subject:/data/subject nn-artifact-gpu \
    --subject-dir /data/subject \
    --study-date 01.31.2018
```

## CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--subject-dir` | Yes | Path to MIDAS subject folder (must contain `subject.xml`) |
| `--study-date` | Yes | Study date (e.g., `01.31.2018` or `01/31/2018`) |
| `--output-dir` | No | Output directory (default: `{subject-dir}/artifactremoval/nn_artifact_output/`) |
| `--cerebellum` | No | Path to cerebellum segmentation NIfTI |
| `--no-qmap` | No | Disable QMAP==4 override |
| `--batch-size` | No | Inference batch size (default: 4096) |

## Outputs

All outputs are saved to `{subject-dir}/artifactremoval/nn_artifact_output/`.

### inference.py outputs (REF/water reference space)

| File | Description |
|------|-------------|
| `{date}_hybrid_artifact_prob.nii.gz` | Final hybrid P(artifact) — **primary output** |
| `{date}_w_tumor.nii.gz` | Tumor CNN weight map showing the spatial gating pattern |
| `{date}_tumor_cnn_artifact_prob.nii.gz` | Tumor CNN P(artifact) alone |
| `{date}_normal_brain_artifact_prob.nii.gz` | Normal-Brain CNN P(artifact) alone |
| `{date}_flair_seg_in_ref.nii.gz` | FLAIR segmentation registered to REF space (used by `create_chonaanorm.py`) |

### create_chonaanorm.py outputs (spectral grid)

| File | Description |
|------|-------------|
| `{date}_hybrid_thrmap.nii.gz` | Spatially-varying artifact threshold map |
| `{date}_chonaa_hybridthr_norm.nii.gz` | CHO/NAA filtered by hybrid threshold, NAWM-normalized |
| `{date}_chonaa_qmap4_norm.nii.gz` | CHO/NAA filtered by QMAP≥4, NAWM-normalized |

## CHO/NAA Normalized Maps

After running inference, use `create_chonaanorm.py` to produce NAWM-normalized CHO/NAA maps
with artifacts removed:

```bash
python create_chonaanorm.py --subject-dir /path/to/subject --study-date 01.31.2018
```

### How it works

1. Loads the hybrid artifact probability map from inference output
2. Builds a **spatially-varying threshold map** that is:
   - Permissive (thr=0.50) inside the FLAIR hyperintensity
   - Graduated (thr=0.40) within 2 voxels outside FLAIR
   - Strict (thr=0.10) in normal-appearing brain
   - Very strict (~0.00) at the top and bottom 20% of brain extent
   - Penalized in the cerebellum/posterior fossa
3. Normalizes CHO/NAA to the mean value in normal-appearing white matter (NAWM mask)
4. Applies the threshold mask and saves the filtered volume

### Additional inputs required

| Path | Description |
|------|-------------|
| `{subject-dir}/mri/{date}_nawm_mask.nii.gz` | Normal-appearing white matter mask |
| `{subject-dir}/ants/segmentations_{date}.nii.gz` | ANTs tissue segmentation |
| NNFit xarray dataset | CHO/NAA ratio map (loaded via MIDAS) |

### CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--subject-dir` | Yes | Path to MIDAS subject folder |
| `--study-date` | Yes | Study date (e.g., `01.31.2018` or `01/31/2018`) |
| `--output-dir` | No | Output directory (default: same as inference output) |
| `--no-qmap` | No | Skip QMAP≥4 filtered output |

## Input Folder Structure

```
subject_folder/
├── subject.xml                          # MIDAS subject metadata
├── mri/
│   ├── {date}_autoseg/
│   │   ├── binary_segmentation_flair_after_resize.nii   # FLAIR seg
│   │   └── preprocessed_flair.nii.gz                     # FLAIR image
│   └── {date}_nawm_mask.nii.gz          # NAWM mask (for create_chonaanorm.py)
├── ants/
│   └── segmentations_{date}.nii.gz      # ANTs segmentation (cerebellum + tissues)
└── nnfit/                               # nnfit spectral data
```

## Docker Compose

Edit `docker-compose.yml` with your paths:

```yaml
services:
  nn-artifact:
    build: .
    volumes:
      - /your/subject/path:/data/subject
    command: ["--subject-dir", "/data/subject", "--study-date", "01.31.2018"]
```

Then run:

```bash
docker compose up
```
