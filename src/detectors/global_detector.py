import numpy as np

def global_detector(rec_err, latent, contamination=0.01):
    # Sử dụng reconstruction error + latent norm
    latent_norm = np.linalg.norm(latent, axis=1)
    score = rec_err + latent_norm
    # top contamination
    th = np.percentile(score, 100*(1-contamination))
    return score
