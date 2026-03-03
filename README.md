# NN-Artifact

Hybrid artifact detection for MR spectroscopy data. Combines two expert neural networks (RTNN and Saumya) with spatial voxel-shell gating based on FLAIR hyperintensity segmentation.

## How It Works

1. **RTNN model** (3-channel CNN): processes raw spectra + water reference + fitted spectra → P(artifact)
2. **Saumya model** (1-channel NN): processes raw spectra → P(artifact)
3. **Hybrid voxel-shell gating**: blends both predictions spatially using FLAIR segmentation:
   - Inside FLAIR hyperintensity (dilated 2 voxels): fully trust RTNN (w=1.0)
   - 1 voxel outside: w=0.75
   - 2 voxels outside: w=0.50
   - 3 voxels outside: w=0.25
   - Beyond 3 voxels: fully trust Saumya (w=0.0)
   - Cerebellum outside gate: always use Saumya
   - Optional QMAP==4 override: force artifact probability to 0.0

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

All outputs are NIfTI files in REF (water reference) space:

| File | Description |
|------|-------------|
| `{date}_hybrid_artifact_prob.nii.gz` | Final hybrid P(artifact) — **this is the primary output** |
| `{date}_w_rtnn.nii.gz` | RTNN weight map showing the spatial gating pattern |
| `{date}_rtnn_artifact_prob.nii.gz` | RTNN model P(artifact) alone |
| `{date}_saumya_artifact_prob.nii.gz` | Saumya model P(artifact) alone |

## Input Folder Structure

```
subject_folder/
├── subject.xml                          # MIDAS subject metadata
├── mri/
│   └── {date}_autoseg/
│       ├── binary_segmentation_flair_after_resize.nii   # FLAIR seg
│       └── preprocessed_flair.nii.gz                     # FLAIR image
├── ants/
│   └── segmentations_{date}.nii.gz      # ANTs segmentation (optional, for cerebellum)
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
