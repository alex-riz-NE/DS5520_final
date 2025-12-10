
import os
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from data_processing import build_feature_matrix, build_feature_matrix
from reconstruction import (
    build_autoencoder,
    compute_autoencoder_error,
    compute_pca_error,
)
from clustering import run_hdbscan


# ============================================================
# Utility functions
# ============================================================

def normalize_score(scores):
    """
    Min-max normalize a 1D score array to [0, 1].
    Handles edge case where all scores are identical.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores

    min_val = np.nanmin(scores)
    max_val = np.nanmax(scores)
    denom = max_val - min_val

    if denom <= 1e-12:
        return np.zeros_like(scores)

    return (scores - min_val) / denom


def engineer_suspicious_features(df):
    """
    Feature engineering inspired by your unsupervised fraud-detection design.
    Works off a 'text' column if present; otherwise builds one from job fields.

    Columns created:
      - exclamation_count
      - all_caps_ratio
      - url_count
      - email_personal
      - text_length
      - word_count
      - salary_missing (if 'salary' present)
      - salary_unrealistic (if 'salary' present)
      - state_frequency (if 'state' present)
    """
    df = df.copy()

    # Build a 'text' field if missing, using common job text columns
    if "text" not in df.columns:
        text_cols = [c for c in [
            "title", "company_profile", "description", "requirements", "benefits"
        ] if c in df.columns]
        if text_cols:
            df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
        else:
            df["text"] = ""

    feats = pd.DataFrame(index=df.index)
    text_series = df["text"].fillna("")

    # Text-based features
    feats["exclamation_count"] = text_series.str.count("!").astype(float)

    feats["all_caps_ratio"] = text_series.apply(
        lambda x: (sum(1 for c in x if c.isupper()) / (len(x) + 1))
    ).astype(float)

    feats["url_count"] = text_series.str.count(r"http|www").astype(float)

    feats["email_personal"] = text_series.str.contains(
        r"@gmail|@yahoo|@hotmail", case=False, regex=True
    ).astype(int)

    # Length-based features
    feats["text_length"] = text_series.str.len().astype(float)
    feats["word_count"] = text_series.str.split().str.len().fillna(0).astype(float)

    # Salary anomalies (optional if salary column exists)
    if "salary" in df.columns:
        s = df["salary"]
        feats["salary_missing"] = s.isna().astype(int)
        feats["salary_unrealistic"] = (
            (s > 500000) | (s < 15000)
        ).fillna(False).astype(int)

    # State frequency (rare states potentially suspicious)
    if "state" in df.columns:
        counts = df["state"].value_counts()
        feats["state_frequency"] = df["state"].map(counts).fillna(0).astype(float)

    return feats


def build_full_feature_matrix(jobs_raw, verbose=True):
    """
    1. Run your existing preprocessing (build_feature_matrix + build_feature_matrix).
    2. Engineer suspicious pattern features.
    3. Concatenate them and scale to [0,1].

    Returns
    -------
    jobs_proc        : preprocessed DataFrame
    suspicious_feats : DataFrame of engineered suspicious features
    X_scaled         : np.ndarray features for AE/PCA/HDBSCAN
    """
    if verbose:
        print("[1/6] Preparing base features...")

    jobs_proc = build_feature_matrix(jobs_raw, verbose=verbose)

    # Your existing structured/text feature matrix
    X_base = build_feature_matrix(jobs_proc)
    if hasattr(X_base, "toarray"):
        X_base = X_base.toarray()

    if verbose:
        print(f"      Base feature matrix shape: {X_base.shape}")

    if verbose:
        print("[2/6] Engineering suspicious pattern features...")
    suspicious_feats = engineer_suspicious_features(jobs_proc)
    if verbose:
        print(f"      Suspicious feature matrix shape: {suspicious_feats.shape}")

    # Concatenate
    X_full = np.hstack([X_base, suspicious_feats.values])
    if verbose:
        print(f"      Combined feature matrix shape: {X_full.shape}")

    # Scale to [0,1] for AE/PCA
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_full)

    return jobs_proc, suspicious_feats, X_scaled


# ============================================================
# Main unsupervised anomaly pipeline
# ============================================================

def run_unsupervised_pipeline(
    csv_path,
    latent_dim=64,
    pca_components=100,
    min_cluster_size=20,
    min_samples=5,
    ensemble_percentile=95,
    top_k_anomalies=20,
    ae_epochs=50,
    ae_batch_size=32,
    ae_validation_split=0.2
):
    print("=== FULL UNSUPERVISED PIPELINE START ===")

    # --------------------------------------------------------
    # Load data & build features
    # --------------------------------------------------------
    print("[0/6] Loading data...")
    jobs_raw = pd.read_csv(csv_path)
    print(f"      Loaded {len(jobs_raw)} job postings.")

    jobs_proc, suspicious_feats, X_scaled = build_full_feature_matrix(jobs_raw, verbose=True)
    n_samples, n_features = X_scaled.shape

    # --------------------------------------------------------
    # Train autoencoder on ALL data (unsupervised)
    # --------------------------------------------------------
    print("[3/6] Training tabular autoencoder (unsupervised)...")
    autoencoder, encoder = build_autoencoder(
        input_dim=n_features,
        latent_dim=latent_dim,
        dropout_rate=0.2
    )

    history = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=ae_epochs,
        batch_size=ae_batch_size,
        validation_split=ae_validation_split,
        verbose=1
    )

    ae_error, _ = compute_autoencoder_error(autoencoder, X_scaled)
    latent = encoder.predict(X_scaled, batch_size=ae_batch_size, verbose=0)
    print("      Autoencoder training complete.")
    print(f"      Latent space shape: {latent.shape}")

    # --------------------------------------------------------
    # PCA reconstruction error
    # --------------------------------------------------------
    print("[4/6] Computing PCA reconstruction errors...")
    pca_error, pca_model, X_pca, X_pca_recon = compute_pca_error(
        X_scaled,
        n_components=pca_components
    )
    print("      PCA step complete.")

    # --------------------------------------------------------
    # HDBSCAN clustering in latent space
    # --------------------------------------------------------
    print("[5/6] Running HDBSCAN in latent space...")
    cluster_res = run_hdbscan(
        latent,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = cluster_res["labels"]
    outlier_scores = cluster_res["outlier_scores"]
    distances = cluster_res["distances"]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"      HDBSCAN discovered {n_clusters} clusters (label -1 = noise).")

    # Silhouette score on non-outliers (if possible)
    non_outliers = labels != -1
    if non_outliers.sum() > 1 and len(np.unique(labels[non_outliers])) > 1:
        sil = silhouette_score(latent[non_outliers], labels[non_outliers])
        print(f"      Silhouette Score (non-outliers): {sil:.3f}")
    else:
        print("      Silhouette Score not defined (too few non-outlier clusters).")

    # --------------------------------------------------------
    # Ensemble anomaly score
    # --------------------------------------------------------
    print("[6/6] Building ensemble anomaly score...")

    anomaly_matrix = np.column_stack([
        normalize_score(ae_error),                # High = anomalous
        normalize_score(pca_error),               # High = anomalous
        (labels == -1).astype(float),             # 1 if HDBSCAN noise
        normalize_score(outlier_scores),          # High = anomalous
        normalize_score(distances),               # High = far from cluster centers
    ])

    feature_names = [
        "AE Error",
        "PCA Error",
        "HDBSCAN Outlier (label=-1)",
        "Outlier Score",
        "Distance to Cluster",
    ]

    # You can tune these; starting from your example
    weights = np.array([0.35, 0.20, 0.20, 0.15, 0.10])
    final_anomaly_score = anomaly_matrix @ weights

    threshold = np.percentile(final_anomaly_score, ensemble_percentile)
    is_anomaly = final_anomaly_score > threshold

    print(f"      Using percentile threshold = {ensemble_percentile}th")
    print(f"      Number of anomalies flagged: {is_anomaly.sum()} / {n_samples}")

    # --------------------------------------------------------
    # Plots: AE error, PCA error, combined score
    # --------------------------------------------------------
    os.makedirs("fig", exist_ok=True)

    plt.figure(figsize=(12, 4))

    # Histogram 1: AE Error
    plt.subplot(1, 3, 1)
    plt.hist(ae_error, bins=50, alpha=0.7)
    plt.axvline(np.percentile(ae_error, ensemble_percentile),
                linestyle="--")
    plt.xlabel("Autoencoder Reconstruction Error")
    plt.ylabel("Count")
    plt.title("AE Error Distribution")

    # Histogram 2: PCA Error
    plt.subplot(1, 3, 2)
    plt.hist(pca_error, bins=50, alpha=0.7)
    plt.axvline(np.percentile(pca_error, ensemble_percentile),
                linestyle="--")
    plt.xlabel("PCA Reconstruction Error")
    plt.ylabel("Count")
    plt.title("PCA Error Distribution")

    # Histogram 3: Final Anomaly Score
    plt.subplot(1, 3, 3)
    plt.hist(final_anomaly_score, bins=50, alpha=0.7)
    plt.axvline(threshold, linestyle="--")
    plt.xlabel("Final Anomaly Score")
    plt.ylabel("Count")
    plt.title("Combined Anomaly Score")

    plt.tight_layout()
    fig_path = os.path.join("fig", "anomaly_distributions.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"      Saved anomaly distribution figure to: {fig_path}")

    # --------------------------------------------------------
    # Cluster statistics
    # --------------------------------------------------------
    print("\n=== Cluster Statistics ===")
    for c in sorted(np.unique(labels)):
        mask = labels == c
        count = mask.sum()
        avg_score = final_anomaly_score[mask].mean() if count > 0 else np.nan
        if c == -1:
            print(f"  Outliers (label -1): {count} postings | avg anomaly = {avg_score:.3f}")
        else:
            print(f"  Cluster {c:2d}: {count} postings | avg anomaly = {avg_score:.3f}")

    # --------------------------------------------------------
    # Top anomalies inspection (unsupervised validation)
    # --------------------------------------------------------
    print(f"\n=== Top {top_k_anomalies} Most Anomalous Postings ===")
    top_idx = np.argsort(final_anomaly_score)[-top_k_anomalies:]
    top_idx = top_idx[::-1]  # highest first

    # Build a reasonable text preview column
    if "text" in jobs_proc.columns:
        text_source = jobs_proc["text"].fillna("")
    else:
        text_cols = [c for c in [
            "title", "company_profile", "description", "requirements", "benefits"
        ] if c in jobs_proc.columns]
        if text_cols:
            text_source = jobs_proc[text_cols].fillna("").agg(" ".join, axis=1)
        else:
            text_source = pd.Series([""] * len(jobs_proc), index=jobs_proc.index)

    for rank, idx in enumerate(top_idx, start=1):
        print(f"\n[Rank {rank}] Posting index {idx}")
        print(f"  Final Anomaly Score: {final_anomaly_score[idx]:.4f}")
        print(f"  AE Error:            {ae_error[idx]:.4f}")
        print(f"  PCA Error:           {pca_error[idx]:.4f}")
        print(f"  HDBSCAN Label:       {labels[idx]}")
        print(f"  Outlier Score:       {outlier_scores[idx]:.4f}")
        print(f"  Distance to Cluster: {distances[idx]:.4f}")
        preview = text_source.iloc[idx][:200].replace("\n", " ")
        print(f"  Text preview:        {preview}...")

    # Feature correlations with final anomaly score
    print("\n=== Correlation of Signals with Final Anomaly Score ===")
    for i in range(anomaly_matrix.shape[1]):
        corr = np.corrcoef(anomaly_matrix[:, i], final_anomaly_score)[0, 1]
        print(f"  {feature_names[i]:30s}: {corr:.3f}")

    # --------------------------------------------------------
    # Save results with anomaly columns
    # --------------------------------------------------------
    results = jobs_proc.copy()
    results["ae_recon_error"] = ae_error
    results["pca_recon_error"] = pca_error
    results["hdbscan_label"] = labels
    results["hdbscan_outlier_score"] = outlier_scores
    results["latent_distance_to_cluster"] = distances
    results["final_anomaly_score"] = final_anomaly_score
    results[f"is_anomaly_top_{ensemble_percentile}pct"] = is_anomaly.astype(int)

    out_csv = "jobs_with_anomaly_scores.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved full results with anomaly scores to: {out_csv}")
    print("=== FULL UNSUPERVISED PIPELINE END ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unsupervised anomaly detection pipeline for job postings."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="fake_job_postings.csv",
        help="Path to CSV of job postings."
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Latent dimension of the autoencoder."
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=100,
        help="Number of PCA components."
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=20,
        help="HDBSCAN min_cluster_size."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples."
    )
    parser.add_argument(
        "--ensemble_percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for anomaly labeling."
    )

    args = parser.parse_args()

    run_unsupervised_pipeline(
        csv_path=args.input,
        latent_dim=args.latent_dim,
        pca_components=args.pca_components,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        ensemble_percentile=args.ensemble_percentile,
    )


if __name__ == "__main__":
    run_unsupervised_pipeline(
        csv_path="fake_job_postings.csv",
        latent_dim=64,
        pca_components=100,
        min_cluster_size=20,
        min_samples=5,
        ensemble_percentile=95,
        top_k_anomalies=20,
        ae_epochs=50,
        ae_batch_size=32
    )