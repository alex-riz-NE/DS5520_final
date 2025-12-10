import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fraud_enrichment_table(df):
    tbl = (
        df.groupby("is_anomaly_top_95pct")["fraudulent"]
          .agg(["count", "mean"])
          .rename(columns={
              "count": "n_jobs",
              "mean": "fraud_rate"
          })
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
    return (
        df.sort_values("final_anomaly_score", ascending=False)
          .loc[:, cols]
          .head(k)
    )

def plot_anomaly_distribution(df):
    plt.figure()
    df["final_anomaly_score"].hist(bins=50)
    plt.axvline(
        df.loc[df["is_anomaly_top_95pct"] == 1, "final_anomaly_score"].min(),
        linestyle="--",
        label="95th percentile threshold"
    )
    plt.xlabel("Final anomaly score")
    plt.ylabel("Count")
    plt.title("Distribution of anomaly scores")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_recon_vs_anomaly(df):
    plt.figure()
    plt.scatter(
        df["ae_recon_error"],
        df["final_anomaly_score"],
        s=15,
        alpha=0.4
    )
    plt.xlabel("Autoencoder reconstruction error")
    plt.ylabel("Final anomaly score")
    plt.title("Reconstruction error vs anomaly score")
    plt.tight_layout()
    plt.show()

def plot_noise_effect(df):
    plt.figure()

    df = df.copy()
    df["is_hdbscan_noise"] = df["hdbscan_label"] == -1

    df.boxplot(
        column="final_anomaly_score",
        by="is_hdbscan_noise"
    )

    plt.xlabel("HDBSCAN noise (True = outlier)")
    plt.ylabel("Final anomaly score")
    plt.title("Noise points vs clustered points")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()


def plot_anomaly_by_fraud(df):
    plt.figure()

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
    plt.show()


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


import numpy as np
import matplotlib.pyplot as plt

def plot_fraud_ranking(
    df,
    score_col,
    label_col="fraudulent",
    title=None,
    max_points=None
):
    # Sort by anomaly score (descending)
    df_sorted = (
        df[[score_col, label_col]]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )

    # Cumulative frauds found
    cumulative_fraud = df_sorted[label_col].cumsum()

    # Normalize x-axis (fraction reviewed)
    n = len(df_sorted)
    x = np.arange(1, n + 1)

    if max_points:
        x = x[:max_points]
        cumulative_fraud = cumulative_fraud[:max_points]

    # Random baseline
    total_fraud = df_sorted[label_col].sum()
    random_baseline = x * (total_fraud / n)

    plt.figure(figsize=(8, 6))
    plt.plot(x, cumulative_fraud, label="Model ranking", linewidth=2)
    plt.plot(x, random_baseline, linestyle="--", label="Random baseline")

    plt.xlabel("Number of postings reviewed")
    plt.ylabel("Cumulative frauds detected")

    plt.title(title or f"Fraud Capture by Anomaly Ranking ({score_col})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_results(
    results_csv,
    ensemble_percentile=95,
    top_k_anomalies=10
):
    df = pd.read_csv(results_csv)

    ae_error = df["ae_recon_error"].values
    pca_error = df["pca_recon_error"].values
    final_anomaly_score = df["final_anomaly_score"].values
    labels = df["hdbscan_label"].values
    outlier_scores = df["hdbscan_outlier_score"].values
    distances = distances = df["latent_distance"].values

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    os.makedirs("fig", exist_ok=True)

    threshold = np.percentile(final_anomaly_score, ensemble_percentile)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(ae_error, bins=50, alpha=0.7)
    plt.axvline(np.percentile(ae_error, ensemble_percentile), linestyle="--")
    plt.title("AE Error Distribution")

    plt.subplot(1, 3, 2)
    plt.hist(pca_error, bins=50, alpha=0.7)
    plt.axvline(np.percentile(pca_error, ensemble_percentile), linestyle="--")
    plt.title("PCA Error Distribution")

    plt.subplot(1, 3, 3)
    plt.hist(final_anomaly_score, bins=50, alpha=0.7)
    plt.axvline(threshold, linestyle="--")
    plt.title("Final Anomaly Score")

    plt.tight_layout()
    fig_path = os.path.join("fig", "anomaly_distributions.png")
    plt.savefig(fig_path)
    plt.close()

    plot_fraud_ranking(
        df,
        score_col="ensemble_anomaly_score",
        title="Fraud Ranking — Ensemble Anomaly Score",
        max_points=500
    )

    plot_fraud_ranking(
        df,
        score_col="ae_recon_error",
        title="Fraud Ranking — Autoencoder Reconstruction Error",
        max_points=500
    )
    plot_fraud_ranking(
        df,
        score_col="outlier_score",
        title="Fraud Ranking — HDBSCAN Outlier Score",
        max_points=500
    )
        

    print(f"Saved anomaly distribution figure to: {fig_path}")

    # --------------------------------------------------------
    # Cluster statistics
    # --------------------------------------------------------
    print("\n=== Cluster Statistics ===")
    for c in sorted(np.unique(labels)):
        mask = labels == c
        count = mask.sum()
        avg_score = final_anomaly_score[mask].mean()
        label = "Outliers" if c == -1 else f"Cluster {c}"
        print(f"  {label:12s}: {count:5d} | avg anomaly = {avg_score:.3f}")

    # --------------------------------------------------------
    # Top anomalies inspection
    # --------------------------------------------------------
    print(f"\n=== Top {top_k_anomalies} Most Anomalous Postings ===")
    top_idx = np.argsort(final_anomaly_score)[-top_k_anomalies:][::-1]

    text_cols = [c for c in [
        "title", "company_profile", "description", "requirements", "benefits"
    ] if c in df.columns]

    text_source = (
        df[text_cols].fillna("").agg(" ".join, axis=1)
        if text_cols else pd.Series([""] * len(df))
    )

    for rank, idx in enumerate(top_idx, start=1):
        print(f"\n[Rank {rank}] Index {idx}")
        print(f"  Final Score:   {final_anomaly_score[idx]:.4f}")
        print(f"  AE Error:      {ae_error[idx]:.4f}")
        print(f"  PCA Error:     {pca_error[idx]:.4f}")
        print(f"  HDBSCAN Label: {labels[idx]}")
        print(f"  Outlier Score: {outlier_scores[idx]:.4f}")
        print(f"  Distance:     {distances[idx]:.4f}")
        preview = text_source.iloc[idx][:200].replace("\n", " ")
        print(f"  Text: {preview}...")

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------
    print("\n=== Correlation with Final Anomaly Score ===")
    signals = ["ae_recon_error", "pca_recon_error", "hdbscan_outlier_score"]
    for col in signals:
        corr = np.corrcoef(df[col], final_anomaly_score)[0, 1]
        print(f"  {col:25s}: {corr:.3f}")


    print("\n=== Fraud enrichment ===")
    print(fraud_enrichment_table(df))

    print("\n=== Top anomalies ===")
    print(top_anomalies_table(df))

    print("\n=== Cluster summary ===")
    print(cluster_summary_table(df))

    plot_anomaly_distribution(df)
    plot_anomaly_by_fraud(df)
    plot_recon_vs_anomaly(df)
    plot_noise_effect(df)


if __name__ == "__main__":
    analyze_results("results/final_v1/jobs_with_anomaly_scores.csv")
