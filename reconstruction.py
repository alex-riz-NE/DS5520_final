# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler



# ============================================================
# Autoencoder
# ============================================================

def build_autoencoder(input_dim, latent_dim=16, hidden_dims=(128, 64), dropout=0.0):
    inp = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(hidden_dims[0], activation="relu")(inp)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(hidden_dims[1], activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent")(x)

    x = layers.Dense(hidden_dims[1], activation="relu")(z)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(hidden_dims[0], activation="relu")(x)
    out = layers.Dense(input_dim)(x)

    ae = models.Model(inp, out)
    encoder = models.Model(inp, z)

    ae.compile(optimizer="adam", loss="mse")
    return ae, encoder


def compute_autoencoder_error(autoencoder, X):
    """
    Compute per-sample reconstruction error for an autoencoder.
    """
    X_hat = autoencoder.predict(X, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=1)

def tune_autoencoder(
    X_train,
    X_eval,
    y_eval,
    n_iter=10,
    percentiles=(92.5, 95, 97.5),
    random_state=42
):
    param_grid = {
        "latent_dim": [8, 16, 32],
        "hidden_dims": [(128, 64), (256, 128)],
        "dropout": [0.0, 0.1],
        "epochs": [15, 25],
        "batch_size": [64, 128],
    }

    results = []

    for params in ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state):
        ae, _ = build_autoencoder(
            input_dim=X_train.shape[1],
            latent_dim=params["latent_dim"],
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"]
        )

        ae.fit(
            X_train, X_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0
        )

        errors = compute_autoencoder_error(ae, X_eval)
        ap = average_precision_score(y_eval, errors)

        for p in percentiles:
            threshold = np.percentile(errors, p)
            preds = (errors >= threshold).astype(int)

            results.append({
                **params,
                "percentile": p,
                "threshold": threshold,
                "pr_auc": ap,
                "flag_rate": preds.mean(),
                "precision": y_eval[preds == 1].mean() if preds.sum() else 0.0,
                "recall": preds[y_eval == 1].mean() if y_eval.sum() else 0.0,
                "mean_recon_error": errors.mean()
            })

    return pd.DataFrame(results).sort_values("pr_auc", ascending=False)



## PCA reconstruction error ###

def compute_pca_error(X,n_components,random_state=42,):
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)
    err = np.mean((X - X_hat) ** 2, axis=1)
    return err, pca, Z, X_hat


def tune_pca_recon(X, y_true, components_grid=(10, 25, 50, 100), percentiles=(90, 92.5, 95, 97.5)):
    rows = []
    for k in components_grid:
        err, _ = compute_pca_error(X, n_components=k)
        ap = average_precision_score(y_true, err)
        for p in percentiles:
            thr = np.percentile(err, p)
            pred = (err >= thr).astype(int)
            rows.append({
                "n_components": k,
                "percentile": p,
                "threshold": thr,
                "ap": ap,
                "flag_rate": pred.mean(),
                "precision": (y_true[pred==1].mean() if pred.sum() else 0.0),
                "recall": (pred[y_true==1].mean() if (y_true==1).sum() else 0.0),
            })
    return pd.DataFrame(rows).sort_values(["ap", "precision"], ascending=False)



if __name__ == "__main__":

    print("Running reconstruction.py sanity check...")
    rng = np.random.default_rng(42)
    # Fake structured feature matrix
    X = rng.normal(size=(1000, 50))

    # ------------------
    # Autoencoder
    # ------------------
    autoencoder, encoder = build_autoencoder(
        input_dim=X.shape[1],
        latent_dim=16
    )

    autoencoder.fit(
        X,
        X,
        epochs=5,
        batch_size=64,
        verbose=1
    )

    ae_err, _ = compute_autoencoder_error(autoencoder, X)

    # ------------------
    # PCA
    # ------------------
    pca_err, pca, _, _ = compute_pca_error(
        X,
        n_components=10
    )

    print(f"AE error mean:  {ae_err.mean():.6f}")
    print(f"PCA error mean: {pca_err.mean():.6f}")
    print("reconstruction.py OK ✅")
