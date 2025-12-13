import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import roc_auc_score, average_precision_score

from data_processing import process
from reconstruction import (
    build_autoencoder,
    compute_autoencoder_error,
    compute_pca_error,
)
from density_anomaly import run_hdbscan_anomaly


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def precision_at_k(y, scores, k):
    idx = np.argsort(scores)[-k:]
    return y[idx].mean()


def recall_at_k(y_true, scores, k):
    idx = np.argsort(scores)[-k:]
    denom = y_true.sum()
    return 0.0 if denom == 0 else y_true[idx].sum() / denom



# --------------------------------------------------
# Tuning grid
# --------------------------------------------------
FAST_TUNING = {
    "hdbscan": {
        "min_cluster_size": [10, 20, 30],
        "min_samples": [5, 10],
        "metric": ["euclidean"],
        "svd_components": [75, 150],
    },
    "pca": {
        "n_components": [20, 50, 100],
    },
    "autoencoder": {
        "latent_dim": [16, 32]
    },
}


# --------------------------------------------------
# Fast tuning
# --------------------------------------------------
def run_fast_tuning(data, out_dir):
    rows = []
    df_data = data["jobs_proc"].copy()

    X_text = data["X_text"]
    X_struct = StandardScaler().fit_transform(data["X_struct"])
    y = data["y"]

    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Cache SVDs (text)
    # --------------------------------------------------
    print("\n[1/4] Precomputing SVD projections (text)...")
    svd_cache = {}
    for svd_dim in FAST_TUNING["hdbscan"]["svd_components"]:
        svd_cache[svd_dim] = TruncatedSVD(
            n_components=min(svd_dim, X_text.shape[1] - 1),
            random_state=42,
        ).fit_transform(X_text)

    # --------------------------------------------------
    # TEXT: HDBSCAN
    # --------------------------------------------------
    print("[2/4] Running HDBSCAN on text embeddings...")
    for mcs, ms, metric, svd_dim in product(
        FAST_TUNING["hdbscan"]["min_cluster_size"],
        FAST_TUNING["hdbscan"]["min_samples"],
        FAST_TUNING["hdbscan"]["metric"],
        FAST_TUNING["hdbscan"]["svd_components"],
    ):
        print(f"  → text HDBSCAN | mcs={mcs} ms={ms} svd={svd_dim}")

        X_red = svd_cache[svd_dim]
        scores, labels = run_hdbscan_anomaly(X_red, mcs, ms, metric)

        if np.isnan(scores).any():
            scores = np.nan_to_num(scores, nan=np.nanmin(scores))

        col_name = f"anomaly_text_hdbscan_mcs{mcs}_ms{ms}_svd{svd_dim}"
        df_data[col_name] = scores

        rows.append({
            "modality": "text",
            "model": "HDBSCAN",
            "min_cluster_size": mcs,
            "min_samples": ms,
            "svd_components": svd_dim,
            "roc_auc": roc_auc_score(y, scores),
            "ap": average_precision_score(y, scores),
            "precision_at_500": precision_at_k(y, scores, 500),
            "recall_at_500": recall_at_k(y, scores, 500),
        })

    # --------------------------------------------------
    # STRUCTURED: HDBSCAN (PCA-reduced)
    # --------------------------------------------------
    print("[3/4] Running HDBSCAN on structured data...")
    X_struct_red = PCA(n_components=20, random_state=42).fit_transform(X_struct)

    for mcs, ms in product(
        FAST_TUNING["hdbscan"]["min_cluster_size"],
        FAST_TUNING["hdbscan"]["min_samples"],
    ):
        print(f"  → structured HDBSCAN | mcs={mcs} ms={ms}")

        scores, labels = run_hdbscan_anomaly(
            X_struct_red,
            mcs,
            ms,
            metric="euclidean",
        )

        if np.isnan(scores).any():
            scores = np.nan_to_num(scores, nan=np.nanmin(scores))

        col_name = f"anomaly_struct_hdbscan_mcs{mcs}_ms{ms}"
        df_data[col_name] = scores

        rows.append({
            "modality": "structured",
            "model": "HDBSCAN",
            "pca_components": 20,
            "min_cluster_size": mcs,
            "min_samples": ms,
            "roc_auc": roc_auc_score(y, scores),
            "ap": average_precision_score(y, scores),
            "precision_at_500": precision_at_k(y, scores, 500),
            "recall_at_500": recall_at_k(y, scores, 500),
        })

    # --------------------------------------------------
    # STRUCTURED: PCA reconstruction
    # --------------------------------------------------
    print("[4/4] Running reconstruction models...")
    for n_comp in FAST_TUNING["pca"]["n_components"]:
        print(f"  → PCA reconstruction | n_components={n_comp}")

        err, _, _, _ = compute_pca_error(
            X_struct,
            n_components=min(n_comp, X_struct.shape[1]),
        )

        col_name = f"anomaly_pca_recon_{n_comp}"
        df_data[col_name] = err

        rows.append({
            "modality": "structured",
            "model": "PCA_recon",
            "n_components": n_comp,
            "roc_auc": roc_auc_score(y, err),
            "ap": average_precision_score(y, err),
            "precision_at_500": precision_at_k(y, err, 500),
            "recall_at_500": recall_at_k(y, err, 500),
        })

    # --------------------------------------------------
    # STRUCTURED: Autoencoder
    # --------------------------------------------------
    for ld in FAST_TUNING["autoencoder"]["latent_dim"]:
        print(f"  → Autoencoder | latent={ld}")

        ae, _ = build_autoencoder(
        input_dim=X_struct.shape[1],
        latent_dim=ld,
    )
        ae.fit(
            X_struct,
            X_struct,
            epochs=8,
            batch_size=256,
            verbose=0,
        )

        err = compute_autoencoder_error(ae, X_struct)

        col_name = f"anomaly_autoencoder_ld{ld}"
        df_data[col_name] = err

        rows.append({
            "modality": "structured",
            "model": "Autoencoder",
            "latent_dim": ld,
            "roc_auc": roc_auc_score(y, err),
            "ap": average_precision_score(y, err),
            "precision_at_500": precision_at_k(y, err, 500),
            "recall_at_500": recall_at_k(y, err, 500),
        })

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    results_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results_df.to_csv(os.path.join(out_dir, "fast_tuning_results.csv"), index=False)

    df_data.to_csv(os.path.join(out_dir, "data_with_anomaly_scores.csv"), index=False)

    print("\nTop models:")
    print(results_df.head(10))
    print("\nSaved:")
    print("  → fast_tuning_results.csv")
    print("  → data_with_anomaly_scores.csv")


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fake_job_postings.csv")
    parser.add_argument("--out", default="results/fast_tuning")
    args = parser.parse_args()

    data = process(args.input)
    run_fast_tuning(data, args.out)


if __name__ == "__main__":
    main()
