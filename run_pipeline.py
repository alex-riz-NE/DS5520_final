import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from data_processing import build_feature_matrix
from reconstruction import (
    build_autoencoder,
    compute_autoencoder_error,
    compute_pca_error,
)
from clustering import run_hdbscan

FINAL_CONFIG = {
    "tfidf_components": 125,
    "latent_dim": 64,
    "pca_components": 20,
    "min_cluster_size": 30,
    "min_samples": 10,
    "ensemble_percentile": 95,
    "ae_epochs": 50,
    "ae_batch_size": 32,
    "ae_validation_split": 0.2,
}


def normalize_score(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


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
    X_struct = jobs_proc.select_dtypes(include=[np.number]).values

    X_text = build_tfidf_features(
    jobs_raw,
    n_components=tfidf_components,
    )
    X = np.hstack([X_struct, X_text]) if X_text is not None else X_struct
    X = MinMaxScaler().fit_transform(X)

    print(f"[features] {X.shape}")

    autoencoder, encoder = build_autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        dropout_rate=0.2,
    )
    autoencoder.fit(
        X,
        X,
        epochs=ae_epochs,
        batch_size=ae_batch_size,
        validation_split=ae_validation_split,
        verbose=1,
    )

    ae_err, _ = compute_autoencoder_error(autoencoder, X)
    latent = encoder.predict(X, batch_size=ae_batch_size, verbose=0)

    pca_err, _, _, _ = compute_pca_error(
        X,
        n_components=min(pca_components, X.shape[1]),
    )

    res = run_hdbscan(
        latent,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    labels = res["labels"]
    out_scores = res["outlier_scores"]
    dists = res["distances"]

    n_clusters = len(set(labels)) - (labels == -1).any()
    print(f"[cluster] k={n_clusters}")

    mask = labels != -1
    if mask.sum() > 1 and np.unique(labels[mask]).size > 1:
        sil = silhouette_score(latent[mask], labels[mask])
        print(f"[cluster] silhouette={sil:.3f}")

    A = np.column_stack([
    normalize_score(ae_err),
    normalize_score(pca_err),
    (labels == -1).astype(float),
    normalize_score(out_scores),
    normalize_score(dists),])

    score = A @ np.array([0.35, 0.20, 0.20, 0.15, 0.10])
    thresh = np.percentile(score, ensemble_percentile)
    is_anomaly = score > thresh

    print(f"[anomaly] flagged={is_anomaly.sum()} ({ensemble_percentile}pctl)")

    out = jobs_proc.copy()
    out["ae_recon_error"] = ae_err
    out["pca_recon_error"] = pca_err
    out["hdbscan_label"] = labels
    out["hdbscan_outlier_score"] = out_scores
    out["latent_distance"] = dists
    out["final_anomaly_score"] = score
    out[f"is_anomaly_top_{ensemble_percentile}pct"] = is_anomaly.astype(int)
    if save_results:
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/jobs_with_anomaly_scores.csv"
        out.to_csv(out_path, index=False)
        print(f"[done] {out_path}")



    print(f"[done] {out_path}")
    summary = {
    "tfidf_dim": None,  # filled by caller
    "min_cluster_size": min_cluster_size,
    "min_samples": min_samples,
    "ensemble_percentile": ensemble_percentile,
    "n_clusters": n_clusters,
    "n_anomalies": int(is_anomaly.sum()),
    "anomaly_rate": is_anomaly.mean(),
    }

    return summary

def quick_tune(csv_path):
    results = []

    tfidf_dims = [75, 125, 200]
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
                )

                summary["tfidf_dim"] = tfidf_dim
                results.append(summary)

    return pd.DataFrame(results)


def main():
    MODE = "run"   # "run" or "tune"

    OUT_DIR = "results/final_v1"
    os.makedirs(OUT_DIR, exist_ok=True)

    INPUT_CSV = "fake_job_postings.csv"

    if MODE == "run":
        run_unsupervised_pipeline(
            csv_path=INPUT_CSV,
            **FINAL_CONFIG,
            save_results=True,
            out_dir=OUT_DIR,   # if you added this parameter
        )

    elif MODE == "tune":
        df = quick_tune(INPUT_CSV)
        df.to_csv("results/tuning_summary.csv", index=False)
        print("\nSaved tuning results → results/tuning_summary.csv")


if __name__ == "__main__":
    main()
