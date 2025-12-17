import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def local_detector(latent, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.01, novelty=True)
    lof.fit(latent)
    score = -lof.decision_function(latent)  # higher -> anomaly
    return score
