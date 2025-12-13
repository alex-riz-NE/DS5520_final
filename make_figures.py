import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------
RESULTS_DIR = "results/fast_tuning"
FIG_DIR = "results/figures"

os.makedirs(FIG_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(RESULTS_DIR, "fast_tuning_results.csv")
DATA_PATH = os.path.join(RESULTS_DIR, "data_with_anomaly_scores.csv")


# --------------------------------------------------
# Load data
# --------------------------------------------------
results = pd.read_csv(RESULTS_PATH)
data = pd.read_csv(DATA_PATH)

y = data["fraudulent"].values  # adjust if label name differs


# ==================================================
# FIGURE 1 — Precision@500 comparison (KEY RESULT)
# ==================================================
print("[1/3] Plotting Precision@500 comparison...")

models = [
    ("Structured HDBSCAN", "structured", "HDBSCAN", None),
    ("PCA (100)", "structured", "PCA_recon", 100),
    ("Autoencoder (32)", "structured", "Autoencoder", 32),
    ("Text HDBSCAN", "text", "HDBSCAN", None),
]

rows = []

for label, modality, model, param in models:
    subset = results[
        (results["modality"] == modality) &
        (results["model"] == model)
    ]

    if model == "PCA_recon":
        subset = subset[subset["n_components"] == param]
    elif model == "Autoencoder":
        subset = subset[subset["latent_dim"] == param]

    rows.append({
        "model": label,
        "precision_at_500": subset["precision_at_500"].iloc[0],
    })

plot_df = pd.DataFrame(rows)

plt.figure(figsize=(7, 4))
plt.bar(plot_df["model"], plot_df["precision_at_500"])
plt.ylabel("Precision@500")
plt.title("Operational Precision Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "precision_at_500_comparison.png"))
plt.close()


# ==================================================
# FIGURE 2 — Anomaly score distributions
# (Structured HDBSCAN, best density model)
# ==================================================
print("[2/3] Plotting anomaly score distributions...")

score_col = "anomaly_struct_hdbscan_mcs30_ms10"

plt.figure(figsize=(7, 4))

plt.hist(
    data.loc[y == 0, score_col],
    bins=50,
    alpha=0.6,
    label="Non-fraud",
)

plt.hist(
    data.loc[y == 1, score_col],
    bins=50,
    alpha=0.6,
    label="Fraud",
)

plt.xlabel("Anomaly score")
plt.ylabel("Count")
plt.title("Anomaly Score Distribution (Structured HDBSCAN)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hdbscan_score_distribution.png"))
plt.close()


# ==================================================
# FIGURE 3 — Precision@K curves
# (Structured HDBSCAN vs PCA reconstruction)
# ==================================================
print("[3/3] Plotting Precision@K curves...")

def precision_at_k_curve(y, scores, ks):
    order = np.argsort(scores)[::-1]
    return [y[order[:k]].mean() for k in ks]


ks = np.arange(50, 2000, 50)

scores_hdb = data["anomaly_struct_hdbscan_mcs30_ms10"].values
scores_pca = data["anomaly_pca_recon_100"].values

p_hdb = precision_at_k_curve(y, scores_hdb, ks)
p_pca = precision_at_k_curve(y, scores_pca, ks)

plt.figure(figsize=(7, 4))
plt.plot(ks, p_hdb, label="Structured HDBSCAN")
plt.plot(ks, p_pca, label="PCA Reconstruction (100)")
plt.xlabel("K")
plt.ylabel("Precision@K")
plt.title("Precision@K Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "precision_at_k_curve.png"))
plt.close()


# --------------------------------------------------
# Done
# --------------------------------------------------
print("\nSaved figures:")
print("  → precision_at_500_comparison.png")
print("  → hdbscan_score_distribution.png")
print("  → precision_at_k_curve.png")
