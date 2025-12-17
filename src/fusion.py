import numpy as np

def fuse_scores(scores, weights=None, method='rank'):
    keys = list(scores.keys())
    if not keys:
        # Return an empty array or raise an error if no scores are provided
        # For now, return an empty array if the first score is empty
        return np.array([])
    if weights is None:
        weights = {k:1.0 for k in keys}
    
    if method == 'rank':
        # Rank-based fusion: Sum of ranks
        fused = np.zeros_like(next(iter(scores.values())), dtype=float)
        for k in keys:
            s = scores[k]
            # Rank the scores (higher score = higher rank)
            # Use 'min' method to assign the smallest rank to ties
            ranks = np.argsort(np.argsort(s)) + 1
            fused += weights[k] * ranks
        
        # Normalize the final rank score to [0,1]
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        return fused
        
    elif method == 'weighted_sum':
        # Weighted sum fusion: Min-Max normalization followed by weighted sum
        total_weight = sum(weights.values())
        fused = np.zeros_like(next(iter(scores.values())))
        for k in keys:
            s = scores[k]
            # Normalize score to [0,1]
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            fused += weights[k] * s
        fused /= total_weight
        return fused
    
    else:
        raise ValueError(f"Unknown fusion method: {method}")
