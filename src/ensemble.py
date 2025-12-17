import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ensemble_scores(rec_err, if_score, lof_score=None):
    scaler = MinMaxScaler()

    rec = scaler.fit_transform(rec_err.reshape(-1, 1)).ravel()
    iff = scaler.fit_transform(if_score.reshape(-1, 1)).ravel()

    if lof_score is not None:
        lof = scaler.fit_transform(lof_score.reshape(-1, 1)).ravel()
        return 0.5 * rec + 0.3 * iff + 0.2 * lof

    return 0.6 * rec + 0.4 * iff
