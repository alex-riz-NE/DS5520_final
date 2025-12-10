# Suppress specific warnings from HDBSCAN
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message=".*force_all_finite.*")
warnings.filterwarnings("ignore",category=RuntimeWarning,message="invalid value encountered in scalar divide")

import pandas as pd
import numpy as np
import hdbscan
from data_processing import build_feature_matrix,build_preprocessor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterSampler
from tensorflow.keras import layers,models
from data_processing import process
from sklearn.metrics import pairwise_distances

import sys


def run_hdbscan(latent,
                min_cluster_size=20,
                min_samples=5,
                metric="euclidean"):
    """
    Run HDBSCAN on latent representations and produce:
      - cluster labels
      - outlier scores
      - distance to nearest cluster center
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True
    )

    labels = clusterer.fit_predict(latent)
    outlier_scores = clusterer.outlier_scores_

    # Compute cluster centers in latent space
    unique_labels = np.unique(labels)
    centers = []
    for c in unique_labels:
        if c == -1:
            continue  # skip noise
        mask = labels == c
        centers.append(latent[mask].mean(axis=0))
    centers = np.array(centers)

    if centers.size > 0:
        distances = pairwise_distances(latent, centers).min(axis=1)
    else:
        # Edge case: no clusters found
        distances = np.zeros(latent.shape[0])

    return {
        "model": clusterer,
        "labels": labels,
        "outlier_scores": outlier_scores,
        "cluster_centers": centers,
        "distances": distances,
    }

if __name__ == "__main__":
    print("Running clustering.py sanity check...")

    import numpy as np

    # Fake latent space
    latent = np.random.normal(size=(500, 16))

    res = run_hdbscan(
        latent,
        min_cluster_size=15,
        min_samples=5
    )

    labels = res["labels"]
    outlier_scores = res["outlier_scores"]
    distances = res["distances"]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"Clusters found: {n_clusters}")
    print(f"Outliers found: {(labels == -1).sum()}")
    print(f"Mean outlier score: {outlier_scores.mean():.4f}")
    print(f"Mean distance: {distances.mean():.4f}")
    print("clustering.py OK ✅")