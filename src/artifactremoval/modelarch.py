import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import pickle, numpy as np, pandas as pd
import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def sen_at_spec95(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.where(1 - fpr >= 0.95)[0]          # specificity ≥ 0.95
    return tpr[idx].max() if len(idx) else 0.0

def eval_metrics(y_true, y_prob):
    return {
        "AUC"      : roc_auc_score(y_true, y_prob),
        "F1"       : f1_score(y_true, y_prob > 0.5),
        "Sens@95%" : sen_at_spec95(y_true, y_prob)
    }

def get_view_stack(indices, raw_arr, water_arr, fit1_arr, fit2_arr):
    """Return dict of (len(indices), 512) arrays for each view."""
    return {
        "raw"  : raw_arr[indices],
        "water": water_arr[indices],
        "fit1" : fit1_arr[indices],
        "fit2" : fit2_arr[indices],
    }

def build_tensor(indices, channels, raw, water, fit1, fit2):
    views = get_view_stack(indices, raw, water, fit1, fit2)
    return np.stack([views[c] for c in channels], axis=-1).astype("float32")

class ComplexSpectralModel:
    def __init__(self, tile_indices=None):
        # Define how to split the 512-point spectrum into tiles.
        # Default: six tiles as in your previous model.
            # CNN kernel slicing indices
        if tile_indices is None:
            self.tile_indices = [(0, 127), (128, 159), (160, 191), (192, 255), (256, 287), (288, 512)]
        else:
            self.tile_indices = tile_indices

    def resnet1d_block(self, x, filters, kernel_size, stride=1, downsample=False):
        """
        A basic 1D residual block.
        """
        shortcut = x
        # First conv
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # Second conv
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if downsample:
            shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    def build_resnet1d_tile_model(self, tile_length, tile_number):
        """
        Build a ResNet1D model (adapted from ResNet-18) for one tile.
        The model takes an input of shape (tile_length, 1) and outputs a fixed-length feature vector.
        """
        inputs = tf.keras.Input(shape=(tile_length, 1))
        # Initial convolution and max pooling
        x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        
        # Group 1: Two residual blocks with 64 filters
        x = self.resnet1d_block(x, filters=64, kernel_size=3, stride=1, downsample=False)
        x = self.resnet1d_block(x, filters=64, kernel_size=3, stride=1, downsample=False)
        
        # Group 2: Two residual blocks with 128 filters, first block downsampling
        x = self.resnet1d_block(x, filters=128, kernel_size=3, stride=2, downsample=True)
        x = self.resnet1d_block(x, filters=128, kernel_size=3, stride=1, downsample=False)
        
        # Group 3: Two residual blocks with 256 filters, first block downsampling
        x = self.resnet1d_block(x, filters=256, kernel_size=3, stride=2, downsample=True)
        x = self.resnet1d_block(x, filters=256, kernel_size=3, stride=1, downsample=False)
        
        # Group 4: Two residual blocks with 512 filters, first block downsampling
        x = self.resnet1d_block(x, filters=512, kernel_size=3, stride=2, downsample=True)
        x = self.resnet1d_block(x, filters=512, kernel_size=3, stride=1, downsample=False)
        
        # Global average pooling to create a fixed-length vector
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Flatten()(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"resnet1d_tile_model_{tile_length}_{tile_number}")

    def build_main_model(self, dropout_rate1 = 0.0, dropout_rate2 = 0.0, dense_units=128):
        """
        Build the main model:
          - Splits the 512-point spectrum into tiles.
          - For each tile, applies a ResNet1D-18 model.
          - Concatenates the resulting features.
          - Passes them through two dense layers.
          - Outputs a two-class softmax (good vs. bad spectrum).
        """
        inp = tf.keras.Input(shape=(512,), name="input_spectrum")
        # Expand dims so each spectrum becomes (512, 1)
        x = tf.expand_dims(inp, axis=-1)
        
        tile_outputs = []
        # Process each tile with its own ResNet1D model
        for i, (start, end) in enumerate(self.tile_indices):
            tile = x[:, start:end, :]  # shape: (batch, tile_length, 1)
            tile_length = end - start
            tile_number = i 
            tile_model = self.build_resnet1d_tile_model(tile_length, tile_number)
            tile_feat = tile_model(tile)
            tile_outputs.append(tile_feat)
        
        # Concatenate features from all tiles
        concat = tf.keras.layers.Concatenate(name="concat_tiles")(tile_outputs)
        
        # Two dense layers followed by the final output layer
        x1 = tf.keras.layers.Dense(dense_units, activation="relu", name="dense1")(concat)
        x1 = tf.keras.layers.Dropout(dropout_rate1, name="dropout1")(x1)  # Add this
        x2 = tf.keras.layers.Dense(dense_units, activation="relu", name="dense2")(x1)
        x2 = tf.keras.layers.Dropout(dropout_rate2, name="dropout2")(x2)  # Add this
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x2)    
            
        return tf.keras.Model(inputs=inp, outputs=output, name="ComplexSpectralModel")
    
def zscore_per_spectrum(x, eps=1e-6):
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return (x - mu) / std

def load_most_recent_pickle(output_dir, prefix="spectral_train_"):
    # List all matching files
    pickle_files = sorted(
        output_dir.glob(f"{prefix}*.pkl"),
        key=lambda x: x.stat().st_mtime,  # Sort by modification time
        reverse=True  # Most recent first
    )
    
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found with prefix '{prefix}' in {output_dir}")
    
    most_recent_file = pickle_files[0]
    print(f"Loading most recent file: {most_recent_file.name}")
    
    with open(most_recent_file, "rb") as f:
        data = pickle.load(f)
    
    return data

def build_model(hp):
    # Tunable FC size
    dense_units = hp.Choice("dense_units", [1024, 2048, 4096])

    # Tunable dropout
    dr1 = hp.Float("dropout_rate1", 0.2, 0.3, step=0.025)
    dr2 = hp.Float("dropout_rate2", 0.0, 0.2, step=0.05)

    # Tunable learning‑rate
    lr  = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")

    model = ComplexSpectralModel().build_main_model(
        dropout_rate1=dr1,
        dropout_rate2=dr2,
        dense_units=dense_units  
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

class MyBayesTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs["batch_size"] = hp.Choice("batch_size", [32, 64])
        kwargs.setdefault("epochs", 15)
        kwargs.setdefault("callbacks", [tf.keras.callbacks.EarlyStopping("val_loss", patience=3)])
        return super().run_trial(trial, *args, **kwargs)



class ComplexSpectralMulti:
    def __init__(self, tile_indices=None):
        self.tile_indices = tile_indices or [(0,127),(128,159),(160,191),(192,255),(256,287),(288,512)]

    # ---------- (identical resnet1d_block from your code) ----------
    def resnet1d_block(self, x, filters, kernel, stride=1, downsample=False):
        shortcut = x
        x = tf.keras.layers.Conv1D(filters,kernel,strides=stride,padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv1D(filters,kernel,strides=1,padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if downsample:
            shortcut = tf.keras.layers.Conv1D(filters,1,strides=stride,padding="same")(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut]); x = tf.keras.layers.ReLU()(x)
        return x

    def build_tile(self, tile_len, tile_id, n_ch):
        inp = tf.keras.Input((tile_len, n_ch))
        x   = tf.keras.layers.Conv1D(64,7,2,padding="same")(inp)
        x   = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
        x   = tf.keras.layers.MaxPooling1D(3,2,padding="same")(x)
        # four residual groups (2 blocks each) – same as before
        x = self.resnet1d_block(x,64,3);  x = self.resnet1d_block(x,64,3)
        x = self.resnet1d_block(x,128,3,2,True); x = self.resnet1d_block(x,128,3)
        x = self.resnet1d_block(x,256,3,2,True); x = self.resnet1d_block(x,256,3)
        x = self.resnet1d_block(x,512,3,2,True); x = self.resnet1d_block(x,512,3)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        return tf.keras.Model(inp, x, name=f"tile{tile_id}_len{tile_len}_ch{n_ch}")

    def build(self, n_ch, dense_units=128, dr1=0.0, dr2=0.0):
        inp = tf.keras.Input((512, n_ch))
        tile_feats = []
        for i,(s,e) in enumerate(self.tile_indices):
            tile = inp[:, s:e, :]
            tile_feats.append(self.build_tile(e-s, i, n_ch)(tile))
        x = tf.keras.layers.Concatenate()(tile_feats)
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dr1)(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dr2)(x)
        out= tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inp, out)

def preprocess(data, label_encoder=None):
    """
    Filters out entries without a consensus rating, stacks the four spectral views,
    normalizes them, and returns X‐arrays plus encoded labels.
    
    Args:
      data: list of dicts with keys
            'raw_spectrum','midas_fit','nnfit','water_siref','consensus_rating'
      label_encoder: an existing sklearn LabelEncoder fit on training labels,
                     or None to fit a new one on these data.

    Returns:
      raw_z   (np.ndarray): z-scored raw spectra, shape (N,512)
      water_n (np.ndarray): log10 + min–max normalized water spectra, shape (N,512)
      f1_z    (np.ndarray): z-scored midas_fit spectra, shape (N,512)
      f2_z    (np.ndarray): z-scored nnfit spectra, shape (N,512)
      y       (np.ndarray): float32 labels (0,1,2,…), shape (N,)
      label_encoder: the fitted LabelEncoder (so you can apply it on test)
    """
    # 1) keep only entries with a valid consensus rating
    filtered = [e for e in data if e.get("consensus_rating") is not None]
    
    # 2) extract raw labels
    raw_labels = [e["consensus_rating"] for e in filtered]
    
    # 3) fit or apply label encoder
    if label_encoder is None:
        le = LabelEncoder().fit(raw_labels)
    else:
        le = label_encoder
    y = le.transform(raw_labels).astype("float32")
    
    # 4) stack arrays
    raw_arr   = np.stack([e["raw_spectrum"] for e in filtered])
    fit1_arr  = np.stack([e["midas_fit"]     for e in filtered])
    fit2_arr  = np.stack([e["nnfit"]         for e in filtered])
    water_arr = np.stack([e["water_siref"]   for e in filtered])

    # 5) z-score normalization
    def zscore_per_spectrum(x, eps=1e-6):
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + eps
        return (x - mu) / std
    raw_z = zscore_per_spectrum(raw_arr)
    f1_z  = zscore_per_spectrum(fit1_arr)
    f2_z  = zscore_per_spectrum(fit2_arr)

    # 6) log+min–max normalize water
    eps  = 1e-6
    wlog = np.log10(np.abs(water_arr) + eps)
    wmin = wlog.min(axis=1, keepdims=True)
    wmax = wlog.max(axis=1, keepdims=True) + eps
    water_n = (wlog - wmin) / (wmax - wmin)

    return raw_z, water_n, f1_z, f2_z, y, le


def run_experiment(
    name,
    model_dir,
    channels,
    raw_arr,
    water_arr,
    fit1_arr,
    fit2_arr,
    y,
    k=5,
    seed=42,
    epochs=40,
    batch_size=32,
    tuned_hps=None,          
):
    """
    Runs k-fold CV for the given channel combination, optionally using tuned hyperparameters.
    Saves each fold's best model to `model_dir / f"{name}_fold{fold}.h5"`.
    Returns a DataFrame with one row per fold: [fold, AUC, F1, Sens@95%].
    """

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_arr, y_arr), start=1):
        # 2) Build train & validation tensors
        X_train = build_tensor(train_idx, channels, raw_arr, water_arr, fit1_arr, fit2_arr)
        X_val   = build_tensor(val_idx,   channels, raw_arr, water_arr, fit1_arr, fit2_arr)
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # 3) Instantiate & compile model
        if tuned_hps:
            # use tuned hyperparameters
            model = ComplexSpectralMulti().build(
                n_ch=len(channels),
                dense_units=tuned_hps["dense_units"],
                dr1=tuned_hps["dropout_rate1"],
                dr2=tuned_hps["dropout_rate2"],
            )
            optimizer = tf.keras.optimizers.Adam(tuned_hps["learning_rate"])
            bs = tuned_hps["batch_size"]
        else:
            # default settings
            model = ComplexSpectralMulti().build(n_ch=len(channels), dense_units=128)
            optimizer = tf.keras.optimizers.Adam(1e-4)
            bs = batch_size

        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["AUC"])

        # 4) Train with early stopping
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=bs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )

        # 5) Save the best model for this fold
        model.save(model_dir / f"{name}_fold{fold}", save_format="tf")

        # 6) Evaluate
        y_prob = model.predict(X_val, batch_size=bs, verbose=0).ravel()
        m = eval_metrics(y_val, y_prob)
        m["fold"] = fold
        records.append(m)

        print(
            f"{name} — fold {fold}:  "
            f"AUC={m['AUC']:.3f}, "
            f"F1={m['F1']:.3f}, "
            f"Sens@95%={m['Sens@95%']:.3f}"
        )

    return pd.DataFrame(records)



