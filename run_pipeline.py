import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from data_processing import build_feature_matrix, split_train_test
from reconstruction import (
    build_autoencoder,
    compute_autoencoder_error,
    compute_pca_error,
)
from clustering import run_hdbscan

FINAL_CONFIG = {
    "tfidf_components": 150,
    "min_cluster_size": 20,
    "min_samples": 5,
    "ensemble_percentile": 95.0
}


def normalize_score(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def save_latent_with_metadata(Z, df, out_path):
    latent_df = pd.DataFrame(
        Z,
        columns=[f"z_{i}" for i in range(Z.shape[1])]
    )

    out = pd.concat(
        [
            df[["fraudulent",
                "final_anomaly_score",
                "hdbscan_label"]],
            latent_df
        ],
        axis=1
    )

    out.to_csv(out_path, index=False)

def build_tfidf_features(df, n_components=150, max_features=5000):
    cols = [
        c for c in (
            "company_profile",
            "description",
            "requirements",
        ) if c in df.columns
    ]
    if not cols:
        return None
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1)

    X = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
    ).fit_transform(text)

    k = min(n_components, X.shape[1] - 1)
    X_red = TruncatedSVD(k, random_state=42).fit_transform(X)

    print(f"TF-IDF → SVD: {X_red.shape}")
    return X_red


def run_unsupervised_pipeline(
    csv_path,
    *,
    tfidf_components=125,
    latent_dim=64,
    pca_components=20,
    min_cluster_size=20,
    min_samples=5,
    ensemble_percentile=95,
    ae_epochs=50,
    ae_batch_size=32,
    ae_validation_split=0.2,
    save_results=True,
    out_dir="results",
):

    print("\n=== UNSUPERVISED PIPELINE ===")

    jobs_raw = pd.read_csv(csv_path)
    print(f"[data] rows={len(jobs_raw)}")

    jobs_proc = build_feature_matrix(jobs_raw)

    # ----- build full feature matrix -----
    X_struct = jobs_proc.select_dtypes(include=[np.number]).values
    y = jobs_proc["fraudulent"].values

    X_text = build_tfidf_features(
        jobs_raw,
        n_components=tfidf_components,
    )
    X = np.hstack([X_struct, X_text]) if X_text is not None else X_struct
    X = MinMaxScaler().fit_transform(X)

    # ----- novelty-detection split -----
    train_mask = (y == 0)
    X_train = X[train_mask]
    X_test  = X
    y_test  = y

    print(f"[features] X={X.shape}, X_train={X_train.shape}")

    # ----- autoencoder (train on normal only) -----
    autoencoder, encoder = build_autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        dropout_rate=0.2,
    )
    autoencoder.fit(
        X_train,
        X_train,
        epochs=ae_epochs,
        batch_size=ae_batch_size,
        validation_split=ae_validation_split,
        verbose=1,
    )

    # ----- AE reconstruction error (test) -----
    ae_err, _ = compute_autoencoder_error(autoencoder, X_test)

    # latent spaces
    latent_train = encoder.predict(X_train, batch_size=ae_batch_size, verbose=0)
    latent_test  = encoder.predict(X_test,  batch_size=ae_batch_size, verbose=0)

    # ----- PCA reconstruction error (test) -----
    pca_err, _, _, _ = compute_pca_error(
        X_test,
        n_components=min(pca_components, X.shape[1]),
    )

    # ----- HDBSCAN (fit on normal latent only) -----
    res = run_hdbscan(
        latent_train,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    # HDBSCAN outputs (defined on train space)
    train_labels = res["labels"]
    train_out_scores = res["outlier_scores"]
    train_dists = res["distances"]

    n_clusters = len(set(train_labels)) - (train_labels == -1).any()
    print(f"[cluster] k={n_clusters}")

    # ----- map HDBSCAN effects to test set -----
    # Novelty detection: train defines structure, test is scored
    hdb_noise_flag = np.zeros(len(X_test))
    hdb_noise_flag[train_mask] = (train_labels == -1).astype(float)

    hdb_out_scores = np.zeros(len(X_test))
    hdb_out_scores[train_mask] = normalize_score(train_out_scores)

    hdb_dists = np.zeros(len(X_test))
    hdb_dists[train_mask] = normalize_score(train_dists)

    # ----- ensemble anomaly score -----
    A = np.column_stack([
        normalize_score(ae_err),
        normalize_score(pca_err),
        hdb_noise_flag,
        hdb_out_scores,
        hdb_dists,
    ])

    score = A @ np.array([0.35, 0.20, 0.20, 0.15, 0.10])
    thresh = np.percentile(score, ensemble_percentile)
    is_anomaly = score > thresh

    print(f"[anomaly] flagged={is_anomaly.sum()} ({ensemble_percentile}pctl)")

    # ----- outputs -----
    out = jobs_proc.copy()
    out["ae_recon_error"] = ae_err
    out["pca_recon_error"] = pca_err
    out["hdbscan_label"] = -1
    out.loc[train_mask, "hdbscan_label"] = train_labels

    out["hdbscan_outlier_score"] = hdb_out_scores
    out["latent_distance"] = hdb_dists
    out["final_anomaly_score"] = score
    out[f"is_anomaly_top_{ensemble_percentile}pct"] = is_anomaly.astype(int)

    if save_results:
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/jobs_with_anomaly_scores.csv"
        out.to_csv(out_path, index=False)
        print(f"[done] {out_path}")

    save_latent_with_metadata(
        latent_test,
        out,
        "results/latent_vectors_with_metadata.csv"
    )

    print(
        "[eval] fraud capture rate:",
        (is_anomaly & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)
    )

    return {
        "tfidf_dim": None,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "ensemble_percentile": ensemble_percentile,
        "n_clusters": n_clusters,
        "n_anomalies": int(is_anomaly.sum()),
        "anomaly_rate": is_anomaly.mean(),
    }


def quick_tune(csv_path):
    results = []

    tfidf_dims = [75, 150, 300]
    hdbscan_params = [(15, 5), (20, 5), (30, 10)]
    percentiles = [92.5, 95, 97.5]

    for tfidf_dim in tfidf_dims:
        for min_cs, min_s in hdbscan_params:
            for pct in percentiles:

                print(
                    f"\n--- tfidf={tfidf_dim} | "
                    f"hdbscan=({min_cs},{min_s}) | "
                    f"pct={pct} ---"
                )

                summary = run_unsupervised_pipeline(
                    csv_path=csv_path,
                    tfidf_components=tfidf_dim,
                    min_cluster_size=min_cs,
                    min_samples=min_s,
                    ensemble_percentile=pct,
                    save_results=False
                )

                summary["tfidf_dim"] = tfidf_dim
                results.append(summary)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["run", "tune"],
        default="run",
        help="Run full pipeline or tune hyperparameters"
    )

    args = parser.parse_args()
    MODE = args.mode

    OUT_DIR = "results/final_v1"
    os.makedirs(OUT_DIR, exist_ok=True)

    INPUT_CSV = "fake_job_postings.csv"

    if MODE == "run":
        summary = run_unsupervised_pipeline(
            csv_path=INPUT_CSV,
            **FINAL_CONFIG,
            save_results=True,
            out_dir=OUT_DIR,
        )

    elif MODE == "tune":
        df = quick_tune(INPUT_CSV)
        df.to_csv("results/tuning_summary.csv", index=False)
        print("\nSaved tuning results → results/tuning_summary.csv")


if __name__ == "__main__":
    main()
