# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

NN-Artifact detects artifacts in MR spectroscopy data by combining two pre-trained neural networks with spatial gating. It is designed to run against MIDAS-processed subject folders and outputs NIfTI probability maps.

## Commands

### Docker (primary deployment)
```bash
# Build (CPU)
docker build -t nn-artifact .

# Build (GPU)
docker build -f Dockerfile.gpu -t nn-artifact-gpu .

# Run inference
docker run -v /path/to/subject:/data/subject nn-artifact \
    --subject-dir /data/subject --study-date 01.31.2018
```

### Local development (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt
pip install ./pymidas/
pip install -e .

# Run inference directly
python inference.py --subject-dir /path/to/subject --study-date 01.31.2018

# Run validation (after inference completes)
python tests/test_validate.py --subject-dir /path/to/subject --study-date 01.31.2018
```

## Architecture

### Inference pipeline (`inference.py`)
The main entrypoint orchestrates the full pipeline:
1. Loads MIDAS subject data via `MidasSubject` / `NNFitDataset`
2. Runs the **Normal-Brain CNN** (SavedModel at `models/NNArtifact_tf2/`) on raw spectra only → `P(artifact)`
3. Runs the **Tumor CNN** (Keras `.keras` at `models/RTNNARTIFACT_best.keras`) on 3-channel input (raw + water + fit) → `P(artifact)`, optionally with MC dropout
4. Calls `run_hybrid_gating()` to spatially blend the two predictions
5. Saves 4 NIfTI outputs and updates `subject.xml`

### Two expert models
- **Normal-Brain CNN** (`load_normal_brain_cnn_model`): TF SavedModel, 1-channel input (raw spectra only). Output channel 1 = P(good), so P(artifact) = 1 − P(good).
- **Tumor CNN** (`load_tumor_cnn_model`): Keras model, 3-channel input `[raw_norm, water_norm, fit_norm]`. Single sigmoid output = P(good). Supports MC dropout by calling `model(..., training=True)` K times.

### Hybrid voxel-shell gating (`src/artifactremoval/hybrid_gating.py`)
Builds a spatial weight map `w_tumor` that controls how much to trust the Tumor CNN vs. the Normal-Brain CNN per voxel:
- Registers FLAIR image to REF (water reference) space using rigid Euler3D registration (Mattes MI, multi-resolution)
- Dilates the FLAIR hyperintensity segmentation by `FLAIR_GATE_DILATE_VOX=2` voxels within the brain mask
- Computes distance shells outside the gate: w=1.0 (inside), 0.75, 0.50, 0.25, 0.0 (>3 voxels outside)
- Cerebellum/brainstem voxels outside gate always use Normal-Brain CNN (w=0)
- Optional QMAP==4 override forces P(artifact)=0 within the FLAIR segmentation

Blend formula: `hybrid = w_tumor * P_tumor_cnn + (1 − w_tumor) * P_normal_brain`

### MIDAS data layer (`src/artifactremoval/midas.py`)
Wraps the MIDAS subject XML hierarchy: `MidasSubject → MidasStudy → MidasSeries → MidasProcess → MidasData → MidasFrame`. Spectral data is loaded as complex numpy arrays of shape `(Z, X, Y, 512)`. The `NNFitDataset` class reads nnfit xarray Zarr datasets for baseline and peak fits.

### XML provenance (`src/artifactremoval/update_xml.py`)
After inference, writes an `NNArtifact` input+data+frame node into `subject.xml` under the `SI → Maps` process, recording inference parameters and artifact statistics. Operation is idempotent.

## Key Files

| File | Purpose |
|------|---------|
| `inference.py` | Main CLI entrypoint |
| `src/artifactremoval/hybrid_gating.py` | Spatial blending logic, registration, weight maps |
| `src/artifactremoval/model_inference.py` | Model loading, preprocessing, batch inference |
| `src/artifactremoval/midas.py` | MIDAS XML/data parsing, `NNFitDataset` |
| `src/artifactremoval/update_xml.py` | subject.xml provenance writer |
| `src/artifactremoval/modelarch.py` | `ComplexSpectralModel`/`ComplexSpectralMulti` architecture (training only) |
| `models/RTNNARTIFACT_best.keras` | Tumor CNN weights |
| `models/NNArtifact_tf2/` | Normal-Brain CNN SavedModel |
| `tests/test_validate.py` | Output validation script (requires completed inference run) |
| `pymidas/` | Vendored MIDAS library (installed as a package) |

## Important Conventions

- All spatial arrays use `(Z, Y, X)` ordering (SimpleITK convention via `GetArrayFromImage`).
- Spectral arrays from MIDAS use `(Z, X, Y, S)` ordering where S=512 spectral points.
- Both models output P(good); artifact probability is always `1 − P(good)`.
- Date strings may use dots (`01.31.2018`) or slashes (`01/31/2018`); `inference.py` normalizes both forms.
- The `pymidas` dependency is a vendored local package; it must be installed before `artifactremoval`.
- `src/` is added to `sys.path` in `inference.py` so the package works without installation during development.
