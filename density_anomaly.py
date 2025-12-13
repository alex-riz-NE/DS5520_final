import numpy as np, pandas as pd, hdbscan
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterSampler

def fit_hdbscan_anomaly(X, min_cluster_size, min_samples):
    m = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True).fit(X)
    scores = m.outlier_scores_
    if np.isnan(scores).any():
        min_score = np.nanmin(scores)
        scores = scores.copy()
        scores[np.isnan(scores)] = min_score

    return scores, m.labels_

def evaluate_scores(y_true, scores):
    return {
        "auc": roc_auc_score(y_true, scores),
        "ap": average_precision_score(y_true, scores),
        "score_mean": scores.mean(),
        "score_std": scores.std()
    }

def run_hdbscan_anomaly(X, min_cluster_size, min_samples, metric):
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True,
    ).fit(X)

    scores = model.outlier_scores_
    return scores, model.labels_

def tune_hdbscan_anomaly(X, y_true, n_iter=20):
    dist = {"min_cluster_size": range(10,150), "min_samples": range(2,20)}
    search = ParameterSampler(dist, n_iter=n_iter, random_state=42)
    best = None
    rows = []
    for p in search:
        s, lbl = fit_hdbscan_anomaly(X, p["min_cluster_size"], p["min_samples"])
        auc = roc_auc_score(y_true, s)
        ap = average_precision_score(y_true, s)
        rows.append({"min_cluster_size": p["min_cluster_size"], "min_samples": p["min_samples"], "auc": auc, "ap": ap, "mean": s.mean(), "std": s.std(), "n_clusters": len(set(lbl)) - (1 if -1 in lbl else 0), "n_noise": sum(lbl == -1)})
        if best is None or auc > best["auc"]:
            best = {"auc": auc, "ap": ap, "params": p}
    return pd.DataFrame(rows), best
