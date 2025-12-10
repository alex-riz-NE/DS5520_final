# reconstruction.py

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from data_processing import process

def build_autoencoder(input_dim, latent_dim=64, dropout_rate=0.2):
    """
    Build a simple fully-connected autoencoder for tabular data.

    Returns
    -------
    autoencoder : keras.Model
    encoder     : keras.Model
    """
    input_layer = layers.Input(shape=(input_dim,), name="input")

    # Encoder
    x = layers.Dense(256, activation="relu")(input_layer)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(128, activation="relu")(latent)
    x = layers.Dense(256, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer, name="tabular_autoencoder")
    encoder = models.Model(inputs=input_layer, outputs=latent, name="tabular_encoder")

    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder


def compute_autoencoder_error(autoencoder, X_scaled, batch_size=256):
    """
    Compute autoencoder reconstruction error (MSE per row).
    """
    reconstructed = autoencoder.predict(X_scaled, batch_size=batch_size, verbose=0)
    errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    return errors, reconstructed


def compute_pca_error(X_scaled, n_components=100, random_state=42):
    """
    Fit PCA and compute reconstruction error (MSE per row).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    return errors, pca, X_pca, X_reconstructed


if __name__ == "__main__":

    print("Running reconstruction.py sanity check...")

    import numpy as np

    # Fake data: 1,000 samples, 50 features
    jobs_proc, X= process()

    autoencoder, encoder = build_autoencoder(input_dim=X.shape[1],latent_dim=16)

    autoencoder.fit(
        X, X,
        epochs=5,
        batch_size=64,
        verbose=1
    )

    ae_error, _ = compute_autoencoder_error(autoencoder, X)
    pca_error, pca, _, _ = compute_pca_error(X, n_components=10)

    print(f"AE error mean:  {ae_error.mean():.6f}")
    print(f"PCA error mean: {pca_error.mean():.6f}")
    print("reconstruction.py OK ✅")