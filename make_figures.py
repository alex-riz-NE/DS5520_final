import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

from data_processing import process
from density_anomaly import run_hdbscan_anomaly
from reconstruction import (
    build_autoencoder,
    compute_autoencoder_error,
    compute_pca_error,
)

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)
data = process("fake_job_postings.csv")

X_struct = StandardScaler().fit_transform(data["X_struct"])
X_text = data["X_text"]
y = data["y"]

# PCA baseline
PCA_RECON_DIM = 100

# Structured HDBSCAN
STRUCT_PCA_DIM = 20
STRUCT_MCS = 30
STRUCT_MS = 10

# Text HDBSCAN
TEXT_SVD_DIM = 150
TEXT_MCS = 30
TEXT_MS = 10

# Autoencoder
AE_LATENT_DIM = 16

scores_pca, _, _, _ = compute_pca_error(
    X_struct,
    n_components=PCA_RECON_DIM,
)

X_struct_red = PCA(
    n_components=STRUCT_PCA_DIM,
    random_state=42,
).fit_transform(X_struct)

scores_struct, _ = run_hdbscan_anomaly(
    X_struct_red,
    min_cluster_size=STRUCT_MCS,
    min_samples=STRUCT_MS,
    metric="euclidean",
)

scores_struct = np.nan_to_num(scores_struct, nan=np.nanmin(scores_struct))
X_text_red = TruncatedSVD(
    n_components=TEXT_SVD_DIM,
    random_state=42,
).fit_transform(X_text)

scores_text, _ = run_hdbscan_anomaly(
    X_text_red,
    min_cluster_size=TEXT_MCS,
    min_samples=TEXT_MS,
    metric="euclidean",
)

scores_text = np.nan_to_num(scores_text, nan=np.nanmin(scores_text))




ae, _ = build_autoencoder(
    input_dim=X_struct.shape[1],
    latent_dim=AE_LATENT_DIM,
)

ae.fit(
    X_struct,
    X_struct,
    epochs=8,
    batch_size=256,
    verbose=0,
)

scores_ae = compute_autoencoder_error(ae, X_struct)

def precision_at_k(y, scores, k):
    return y[np.argsort(scores)[-k:]].mean()


def precision_at_k_curve(y, scores, ks):
    order = np.argsort(scores)[::-1]
    return [y[order[:k]].mean() for k in ks]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))



# Precision@500
labels = [
    "PCA Baseline",
    "Structured HDBSCAN",
    "Text HDBSCAN",
    "Autoencoder",
]

precisions = [
    precision_at_k(y, scores_pca, 500),
    precision_at_k(y, scores_struct, 500),
    precision_at_k(y, scores_text, 500),
    precision_at_k(y, scores_ae, 500),
]

axes[0].bar(labels, precisions)
axes[0].set_ylabel("Precision@500")
axes[0].set_title("A. Operational Precision")
axes[0].tick_params(axis="x", rotation=20)
bins = 50

axes[1].hist(
    scores_pca[y == 1],
    bins=bins,
    alpha=0.45,
    label="PCA Reconstruction",
)

axes[1].hist(
    scores_struct[y == 1],
    bins=bins,
    alpha=0.45,
    label="Structured HDBSCAN",
)

axes[1].hist(
    scores_text[y == 1],
    bins=bins,
    alpha=0.45,
    label="Text HDBSCAN",
)

axes[1].hist(
    scores_ae[y == 1],
    bins=bins,
    alpha=0.45,
    label="Autoencoder",
)

axes[1].set_xlabel("Anomaly score")
axes[1].set_ylabel("Count (Fraud only)")
axes[1].set_title("B. Fraud Score Distributions")
axes[1].legend()


# --------------------------------------------------
# Panel C — Precision@K curves
# --------------------------------------------------
ks = np.arange(50, 2000, 50)

axes[2].plot(
    ks,
    precision_at_k_curve(y, scores_pca, ks),
    label="PCA Baseline",
)
axes[2].plot(
    ks,
    precision_at_k_curve(y, scores_struct, ks),
    label="Structured HDBSCAN",
)
axes[2].plot(
    ks,
    precision_at_k_curve(y, scores_text, ks),
    label="Text HDBSCAN",
)
axes[2].plot(
    ks,
    precision_at_k_curve(y, scores_ae, ks),
    label="Autoencoder",
)

axes[2].set_xlabel("K")
axes[2].set_ylabel("Precision@K")
axes[2].set_title("C. Precision@K Curves")
axes[2].legend()

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "combined_results_full.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved → {out_path}")
