import numpy as np

def global_detector(rec_err, latent, contamination=0.01):
    # Sử dụng reconstruction error + latent norm
    latent_norm = np.linalg.norm(latent, axis=1)
    score = rec_err + latent_norm
    # The score is the anomaly score, higher is more anomalous
    return score
