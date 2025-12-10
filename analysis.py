import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================================================
# Paths / constants
# =========================================================

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


# =========================================================
# Tables
# =========================================================

def precision_at_k(y_true, scores, k):
    idx = np.argsort(scores)[-k:]
    return y_true[idx].mean()


def fraud_enrichment_table(df):
    tbl = (
        df.groupby("is_anomaly_top_95pct")["fraudulent"]
          .agg(["count", "mean"])
          .rename(columns={"count": "n_jobs", "mean": "fraud_rate"})
    )
    tbl["fraud_rate"] *= 100
    return tbl


def top_anomalies_table(df, k=10):
    cols = [
        "job_id",
        "title",
        "fraudulent",
        "final_anomaly_score",
        "ae_recon_error",
        "pca_recon_error",
        "hdbscan_label",
    ]
    cols = [c for c in cols if c in df.columns]

    return (
        df.sort_values("final_anomaly_score", ascending=False)
          .loc[:, cols]
          .head(k)
    )


def cluster_summary_table(df):
    return (
        df.assign(is_noise=df["hdbscan_label"] == -1)
          .groupby("is_noise")
          .agg(
              n_jobs=("job_id", "count"),
              avg_anomaly_score=("final_anomaly_score", "mean"),
              fraud_rate=("fraudulent", "mean"),
          )
          .assign(fraud_rate=lambda x: x["fraud_rate"] * 100)
    )


# =========================================================
# Plot helpers
# =========================================================

def plot_anomaly_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(df["final_anomaly_score"], bins=50)
    ax.axvline(
        df.loc[df["is_anomaly_top_95pct"] == 1, "final_anomaly_score"].min(),
        linestyle="--",
        label="95th percentile threshold"
    )

    ax.set_xlabel("Final anomaly score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of anomaly scores")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "anomaly_score_distribution.png"))
    plt.close(fig)



def plot_anomaly_by_fraud(df):
    plt.figure(figsize=(6, 4))

    df[df["fraudulent"] == 0]["final_anomaly_score"].hist(
        bins=50, alpha=0.6, label="Legitimate"
    )
    df[df["fraudulent"] == 1]["final_anomaly_score"].hist(
        bins=50, alpha=0.6, label="Fraudulent"
    )

    plt.axvline(
        df.loc[df["is_anomaly_top_95pct"] == 1, "final_anomaly_score"].min(),
        linestyle="--",
        label="95th percentile threshold",
    )

    plt.xlabel("Final anomaly score")
    plt.ylabel("Count")
    plt.title("Anomaly score by job label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "anomaly_score_by_label.png"))
    plt.close()


def plot_recon_vs_anomaly(df):
    plt.figure(figsize=(6, 4))

    plt.scatter(
        df["ae_recon_error"],
        df["final_anomaly_score"],
        s=10,
        alpha=0.4
    )

    plt.xlabel("Autoencoder reconstruction error")
    plt.ylabel("Final anomaly score")
    plt.title("Reconstruction error vs anomaly score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reconstruction_vs_anomaly.png"))
    plt.close()


def plot_noise_effect(df):
    plt.figure(figsize=(6, 4))

    df = df.copy()
    df["is_hdbscan_noise"] = df["hdbscan_label"] == -1

    df.boxplot(column="final_anomaly_score", by="is_hdbscan_noise")

    plt.xlabel("HDBSCAN noise (True = outlier)")
    plt.ylabel("Final anomaly score")
    plt.title("Noise points vs clustered points")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "noise_vs_clustered.png"))
    plt.close()


# =========================================================
# Fraud ranking helpers
# =========================================================

def fraud_ranking_curve(df, score_col, label_col="fraudulent", max_points=None):
    df_sorted = (
        df[[score_col, label_col]]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )

    cum_fraud = df_sorted[label_col].cumsum()
    n = len(df_sorted)
    x = np.arange(1, n + 1)

    if max_points is not None:
        x = x[:max_points]
        cum_fraud = cum_fraud[:max_points]

    total_fraud = df_sorted[label_col].sum()
    random_baseline = x * (total_fraud / n)

    return x, cum_fraud, random_baseline


def plot_ranking_grid(df, max_points=500):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    models = [
        ("ae_recon_error", "Autoencoder"),
        ("pca_recon_error", "PCA"),
        ("hdbscan_outlier_score", "HDBSCAN"),
    ]

    for ax, (col, label) in zip(axes, models):
        x, y, baseline = fraud_ranking_curve(df, col, max_points=max_points)
        ax.plot(x, y, linewidth=2, label=label)
        ax.plot(x, baseline, linestyle="--", label="Random")
        ax.set_title(label)
        ax.set_xlabel("Postings reviewed")

    axes[0].set_ylabel("Cumulative frauds detected")
    axes[0].legend()

    plt.suptitle("Fraud Ranking by Individual Anomaly Signals")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ranking_grid.png"))
    plt.close()


def plot_ranking_overlay(df, max_points=500):
    plt.figure(figsize=(7, 5))

    curves = [
        ("final_anomaly_score", "Ensemble", 2.5),
        ("ae_recon_error", "Autoencoder", 2),
        ("pca_recon_error", "PCA", 2),
        ("hdbscan_outlier_score", "HDBSCAN", 2),
    ]

    for col, label, lw in curves:
        x, y, baseline = fraud_ranking_curve(df, col, max_points=max_points)
        plt.plot(x, y, linewidth=lw, label=label)

    plt.plot(x, baseline, linestyle="--", color="black", label="Random")

    plt.xlabel("Number of postings reviewed")
    plt.ylabel("Cumulative frauds detected")
    plt.title("Fraud Capture by Anomaly Ranking (All Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ranking_overlay.png"))
    plt.close()

def plot_latent_space(
    Z,
    df,
    color_col,
    title,
    fname,
    cmap="viridis",
    out_dir="fig"
):
    os.makedirs(out_dir, exist_ok=True)

    Z_2d = PCA(n_components=2).fit_transform(Z)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        Z_2d[:, 0],
        Z_2d[:, 1],
        c=df[color_col],
        s=6,
        cmap=cmap,
        alpha=0.3
    )

    plt.xlabel("Latent PC 1")
    plt.ylabel("Latent PC 2")
    plt.title(title)
    plt.colorbar(sc, label=color_col)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fraud_capture_curve(
    df,
    score_col="final_anomaly_score",
    label_col="fraudulent",
    out_dir="fig",
    fname="fraud_capture_curve.png"
):
    os.makedirs(out_dir, exist_ok=True)

    df_sorted = df.sort_values(score_col, ascending=False)

    y = df_sorted[label_col].values
    cum_fraud = np.cumsum(y)
    total_fraud = y.sum()

    pct_inspected = np.arange(1, len(y)+1) / len(y)
    pct_fraud_captured = cum_fraud / total_fraud

    plt.figure(figsize=(6,5))
    plt.plot(pct_inspected, pct_fraud_captured, label="Model", linewidth=2)
    plt.plot([0,1], [0,1], linestyle="--", label="Random")

    plt.xlabel("Fraction of jobs inspected")
    plt.ylabel("Fraction of fraud captured")
    plt.title("Fraud Capture Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def plot_fraud_rate_by_bucket(
    df,
    score_col="final_anomaly_score",
    label_col="fraudulent",
    n_bins=10,
    out_dir="fig",
    fname="fraud_rate_by_bucket.png"
):
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    df["bucket"] = pd.qcut(df[score_col], q=n_bins, duplicates="drop")

    bucket_rates = (
        df.groupby("bucket", observed=False)[label_col]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(7,5))
    sns.barplot(
        data=bucket_rates,
        x="bucket",
        y=label_col
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Fraud rate")
    plt.xlabel("Anomaly score bucket")
    plt.title("Fraud Rate by Anomaly Score Bucket")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()



# =========================================================
# Main analysis
# =========================================================

def analyze_results(results_csv, ensemble_percentile=95, top_k_anomalies=10):
    df = pd.read_csv(results_csv)

    threshold = np.percentile(df["final_anomaly_score"], ensemble_percentile)
    df["is_anomaly_top_95pct"] = (df["final_anomaly_score"] >= threshold).astype(int)

    # ---- Plots ----
    plot_anomaly_distribution(df)
    plot_anomaly_by_fraud(df)
    plot_recon_vs_anomaly(df)
    plot_noise_effect(df)
    plot_ranking_grid(df)
    plot_ranking_overlay(df)

    # ---- Tables ----
    print("\n=== Fraud enrichment ===")
    print(fraud_enrichment_table(df))

    print("\n=== Top anomalies ===")
    print(top_anomalies_table(df, top_k_anomalies))

    print("\n=== Cluster summary ===")
    print(cluster_summary_table(df))
    jobs_proc=pd.read_csv("results/latent_vectors_with_metadata.csv")
    plot_latent_space(
        Z=jobs_proc.filter(like="z_").values,
        df=jobs_proc,
        color_col="final_anomaly_score",
        title="Latent Space Colored by Anomaly Score",
        fname="latent_space_anomaly_score.png",
    )
    plot_latent_space(
    Z=jobs_proc.filter(like="z_").values,
    df=jobs_proc,
    color_col="fraudulent",
    title="Latent Space Colored by Fraud Label",
    cmap="coolwarm",
    fname="latent_fraud.png")
    plot_fraud_capture_curve(jobs_proc)
    plot_fraud_rate_by_bucket(jobs_proc)





# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    analyze_results("results/final_v1/jobs_with_anomaly_scores.csv")
