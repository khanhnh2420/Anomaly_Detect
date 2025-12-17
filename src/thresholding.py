import numpy as np
from sklearn.metrics import precision_recall_curve

def best_f1_threshold(y, scores):
    # Nếu multiclass, chuyển sang binary: anomaly=1, normal=0
    if len(np.unique(y)) > 2:
        y_bin = (y != 0).astype(int)
    else:
        y_bin = y
    p, r, t = precision_recall_curve(y_bin, scores)
    f1 = 2 * p * r / (p + r + 1e-6)
    return t[f1.argmax()]

def adaptive_threshold(scores, factor=1.5):
    mu, sigma = scores.mean(), scores.std()
    return mu + factor * sigma
