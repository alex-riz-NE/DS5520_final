# Suppress specific warnings from HDBSCAN
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message=".*force_all_finite.*")
warnings.filterwarnings("ignore",category=RuntimeWarning,message="invalid value encountered in scalar divide")


import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from tensorflow.keras import layers, models, callbacks
import hdbscan
from data_processing import build_feature_matrix, build_preprocessor
from sklearn.model_selection import ParameterSampler
from text_reconstruction import build_tfidf, reduce_text, run_text_autoencoder  
from reconstruction import pca_reconstruction_error



# ============================================================
# Utilities
# ============================================================

def precision_at_k(y, scores, k):
    idx = np.argsort(scores)[-k:]
    return y[idx].mean()


def build_autoencoder(input_dim, latent_dim):
    x = layers.Input(shape=(input_dim,))
    z = layers.Dense(latent_dim, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(z)
    model = models.Model(x, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def compute_local_anomaly(df, score_col, cluster_col, min_cluster_size=10):
    """
    Local (within-cluster) anomaly score using z-score.
    Noise clusters (-1) are ignored.
    """
    local = pd.Series(0.0, index=df.index)

    for cid, group in df.groupby(cluster_col):
        if cid == -1 or len(group) < min_cluster_size:
            continue

        mu = group[score_col].mean()
        sd = group[score_col].std()

        if sd > 0:
            local.loc[group.index] = (group[score_col] - mu) / sd

    return local


# ============================================================
# One run of the pipeline for a given parameter set
# ============================================================

def run_local_pipeline(
    X,
    jobs_proc,
    min_cluster_size,
    min_samples,
    latent_dim,
    k_eval=10
):
    df = jobs_proc.copy()

    # ---- HDBSCAN for context ----
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    ).fit(X)

    df["cluster"] = clusterer.labels_

    # ---- Autoencoder reconstruction ----
    ae = build_autoencoder(X.shape[1], latent_dim)
    ae.fit(
        X, X,
        epochs=40,
        batch_size=64,
        validation_split=0.1,
        callbacks=[callbacks.EarlyStopping(patience=5)],
        verbose=0
    )

    X_rec = ae.predict(X, verbose=0)
    df["recon_error"] = np.mean((X - X_rec) ** 2, axis=1)

    # ---- Local anomaly ----
    df["local_anomaly"] = compute_local_anomaly(
        df,
        score_col="recon_error",
        cluster_col="cluster"
    )

    df["local_anomaly_norm"] = MinMaxScaler().fit_transform(
        df[["local_anomaly"]]
    )

    # ---- Evaluate ----
    y = df["fraudulent"].values
    scores = df["local_anomaly_norm"].values

    return {
        "precision@10": precision_at_k(y, scores, 10),
        "precision@25": precision_at_k(y, scores, 25),
    }


# ============================================================
# Hyperparameter tuning
# ============================================================

def tune_parameters(X, jobs_proc):
    results = []

    for min_cluster_size in [20, 30, 50]:
        for min_samples in [5, 10, 20]:
            for latent_dim in [4, 8, 16]:

                metrics = run_local_pipeline(
                    X,
                    jobs_proc,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    latent_dim=latent_dim
                )

                row = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "latent_dim": latent_dim,
                    **metrics
                }

                results.append(row)
                print(row)

    results_df = pd.DataFrame(results).sort_values(
        "precision@10", ascending=False
    )

    return results_df

def run_final_pipeline(X, jobs_proc, best_params):
    df = jobs_proc.copy()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"]
    ).fit(X)

    df["hdbscan_cluster"] = clusterer.labels_

    ae = build_autoencoder(X.shape[1], best_params["latent_dim"])
    ae.fit(X, X, epochs=40, batch_size=64, verbose=0)

    X_rec = ae.predict(X, verbose=0)
    df["recon_error"] = np.mean((X - X_rec) ** 2, axis=1)

    df["local_anomaly"] = compute_local_anomaly(
        df, "recon_error", "hdbscan_cluster"
    )

    df["local_anomaly_norm"] = MinMaxScaler().fit_transform(
        df[["local_anomaly"]]
    )

    return df



# ============================================================
# Main
# ============================================================

def main():
    print("=== LOCAL ANOMALY TUNING START ===")

    # ---- Load data ----
    jobs = pd.read_csv("fake_job_postings.csv")
    jobs_proc = build_feature_matrix(jobs)
    X = build_preprocessor().fit_transform(jobs_proc)

    # ---- Tune ----
    tuning_df = tune_parameters(X, jobs_proc)
    tuning_df.to_csv("results/local_tuning_results.csv", index=False)

    print("\n=== TOP CONFIGURATIONS ===")
    print(tuning_df.head(10))

    print("\n✅ Tuning complete. Results saved to results/local_tuning_results.csv")

    best = tuning_df.iloc[0].to_dict()

    final_df = run_final_pipeline(X, jobs_proc, best)
    final_df.to_csv(
        "results/jobs_with_clusters.csv",
        index=False
    )


if __name__ == "__main__":
    main()