"""
Model inference for Tumor CNN and Normal-Brain CNN artifact classifiers.

Extracted from notebooks/21_inference_MIDAS.ipynb
"""

import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def zscore_per_spectrum(x, eps=1e-6):
    """Z-score normalize each spectrum independently."""
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return (x - mu) / std


def preprocess_spectra(spec, water, fit):
    """
    Preprocess raw spectral data into 3-channel normalized input.

    Parameters
    ----------
    spec : np.ndarray
        Raw SI spectra, shape (Z, X, Y, 512), complex.
    water : np.ndarray
        Water reference spectra (SIREF), shape (Z, X, Y, 512), complex.
    fit : np.ndarray
        Fitted spectra (baseline + peaks), shape (Z, X, Y, 512), real.

    Returns
    -------
    X_input : np.ndarray
        Stacked 3-channel input, shape (N, 512, 3), float32.
        Channel order: [raw, water, fit].
    """
    S = spec.shape[-1]

    raw_flat = np.real(spec.reshape(-1, S))
    fit_flat = np.real(fit.reshape(-1, S))
    water_flat = np.real(water.reshape(-1, S))

    raw_norm = zscore_per_spectrum(raw_flat)
    fit_norm = zscore_per_spectrum(fit_flat)

    wat_log = np.log10(np.abs(water_flat) + 1e-6)
    wmin = wat_log.min(axis=1, keepdims=True)
    wmax = wat_log.max(axis=1, keepdims=True)
    water_norm = (wat_log - wmin) / (wmax - wmin + 1e-6)

    X_input = np.stack([raw_norm, water_norm, fit_norm], axis=-1).astype("float32")
    return X_input


def run_normal_brain_cnn_inference(model, spec, batch_size=4096):
    """
    Run Normal-Brain CNN inference on raw spectra.

    Parameters
    ----------
    model : tf.saved_model
        Loaded Normal-Brain CNN SavedModel.
    spec : np.ndarray
        Raw SI spectra, shape (Z, X, Y, 512), complex.
    batch_size : int
        Inference batch size.

    Returns
    -------
    normal_brain_art : np.ndarray
        P(artifact) per voxel, shape (Z, X, Y), float32.
    """
    Z, X, Y, S = spec.shape
    num_classes = 2
    ypred = np.zeros((Z, X, Y, num_classes), np.float32)

    for z in range(Z):
        xtest = np.real(spec[z]).reshape(X * Y, S).astype(np.float32)

        # Process in batches
        preds_list = []
        for i in range(0, xtest.shape[0], batch_size):
            batch = xtest[i:i + batch_size]
            preds = model.predict(batch)
            probs = tf.nn.softmax(preds).numpy()
            preds_list.append(probs)

        probs_all = np.concatenate(preds_list, axis=0)
        ypred[z] = probs_all.reshape(X, Y, num_classes)

    # Channel 1 = P(good), so P(artifact) = 1 - P(good)
    good_prob = ypred[..., 1].astype(np.float32)
    normal_brain_art = 1.0 - np.clip(good_prob, 0.0, 1.0)
    return normal_brain_art


def _freeze_batchnorm(model):
    """Context manager to temporarily freeze all BatchNorm layers."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        bn_layers = [l for l in model.layers
                     if isinstance(l, tf.keras.layers.BatchNormalization)]
        old = [l.trainable for l in bn_layers]
        try:
            for l in bn_layers:
                l.trainable = False
            yield
        finally:
            for l, t in zip(bn_layers, old):
                l.trainable = t

    return _ctx()


def run_tumor_cnn_inference(model, spec, water, fit, batch_size=4096,
                            mc_passes=0):
    """
    Run Tumor CNN 3-channel inference.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Tumor CNN Keras model.
    spec : np.ndarray
        Raw SI spectra, shape (Z, X, Y, 512), complex.
    water : np.ndarray
        Water reference spectra, shape (Z, X, Y, 512), complex.
    fit : np.ndarray
        Fitted spectra, shape (Z, X, Y, 512), real.
    batch_size : int
        Inference batch size.
    mc_passes : int
        Number of MC dropout forward passes. 0 = single deterministic pass.
        Positive value (e.g., 20) enables MC dropout with that many passes.

    Returns
    -------
    tumor_cnn_art : np.ndarray
        P(artifact) per voxel, shape (Z, X, Y), float32.
    """
    Z, X, Y, S = spec.shape
    X_input = preprocess_spectra(spec, water, fit)

    if mc_passes <= 0:
        # Single deterministic pass
        probs_good = model.predict(X_input, batch_size=batch_size).ravel()
    else:
        # MC dropout: K forward passes with dropout active, BN frozen
        # Process in batches to avoid GPU OOM
        logger.info(f"Running MC dropout with K={mc_passes} passes, batch_size={batch_size}")
        N = X_input.shape[0]
        accum = np.zeros(N, dtype=np.float64)
        with _freeze_batchnorm(model):
            for k in range(mc_passes):
                preds_k = []
                for i in range(0, N, batch_size):
                    batch = tf.convert_to_tensor(X_input[i:i + batch_size], dtype=tf.float32)
                    y = model(batch, training=True)  # dropout active
                    preds_k.append(tf.reshape(y, [-1]).numpy())
                accum += np.concatenate(preds_k, axis=0)
                if (k + 1) % 5 == 0:
                    logger.info(f"  MC pass {k + 1}/{mc_passes}")
        probs_good = (accum / mc_passes).astype(np.float32)

    tumor_cnn_art = 1.0 - np.clip(probs_good, 0.0, 1.0)
    return tumor_cnn_art.reshape(Z, X, Y).astype(np.float32)


def load_normal_brain_cnn_model(model_path):
    """Load Normal-Brain CNN SavedModel."""
    logger.info(f"Loading Normal-Brain CNN model from {model_path}")
    return tf.saved_model.load(str(model_path))


def load_tumor_cnn_model(model_path):
    """Load Tumor CNN Keras model."""
    logger.info(f"Loading Tumor CNN model from {model_path}")
    return tf.keras.models.load_model(str(model_path), compile=False)
