import numpy as np

def fuse_scores(scores, weights=None):
    keys = list(scores.keys())
    if weights is None:
        weights = {k:1.0 for k in keys}
    total_weight = sum(weights.values())
    fused = np.zeros_like(next(iter(scores.values())))
    for k in keys:
        s = scores[k]
        # Normalize score to [0,1]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        fused += weights[k] * s
    fused /= total_weight
    return fused
