import numpy as np, pandas as pd, hdbscan
from sklearn.metrics import roc_auc_score, average_precision_score

def precision_at_k(y, scores, k):
    idx = np.argsort(scores)[-k:]
    return y[idx].mean()

def anomaly_metrics(y, scores):
    scores = np.nan_to_num(scores)
    return {
        "roc_auc": roc_auc_score(y, scores),
        "ap": average_precision_score(y, scores),
        "p_at_500": precision_at_k(y, scores, 500),
    }
