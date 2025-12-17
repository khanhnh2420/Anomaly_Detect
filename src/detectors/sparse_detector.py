import numpy as np
from sklearn.ensemble import IsolationForest

def sparse_detector(Xp, contamination=0.01):
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(Xp)
    score = -clf.score_samples(Xp)  # higher -> anomaly
    return score
