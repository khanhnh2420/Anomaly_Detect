import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

from .dataset_profiler import profile_dataset
from .auto_config import auto_config
from .preprocessing import build_preprocessor
from .ae_model import AutoEncoder
from .detectors.global_detector import global_detector
from .detectors.local_detector import local_detector
from .detectors.sparse_detector import sparse_detector
from .fusion import fuse_scores
from .thresholding import adaptive_threshold, best_f1_threshold
from .evaluation import evaluate

def run_pipeline(csv_path, label_col=None):
    df = pd.read_csv(csv_path)

    # Encode labels
    if label_col:
        y_raw = df[label_col].values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        X = df.drop(columns=[label_col])
    else:
        X, y = df, None

    profile, num_cols, cat_cols = profile_dataset(X)
    cfg = auto_config(profile)
    print("AUTO CONFIG:", cfg)

    pre = build_preprocessor(num_cols, cat_cols)
    Xp = pre.fit_transform(X)

    # Sparse -> SVD
    if sp.issparse(Xp):
        n_comp = min(128, Xp.shape[1])
        Xp = TruncatedSVD(n_comp).fit_transform(Xp)

    X_tensor = torch.tensor(Xp, dtype=torch.float32)

    ae = AutoEncoder(X_tensor.shape[1], cfg["latent_dim"])
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=True)
    best_loss = float('inf')
    patience_counter = 0

    for ep in range(cfg["epochs"]):
        epoch_loss = 0
        for (x,) in loader:
            x_hat, _ = ae(x)
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(X_tensor)

        if cfg.get("early_stopping", False):
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"Early stopping at epoch {ep}")
                    break

    # Inference
    with torch.no_grad():
        x_hat, latent = ae(X_tensor)
        rec_err = ((x_hat - X_tensor) ** 2).mean(1).numpy()
        latent = latent.numpy()

    scores = {}
    if cfg["use_global"]:
        scores["global"] = global_detector(rec_err, latent, cfg["contamination"])
    if cfg["use_local"]:
        scores["local"] = local_detector(latent)
    if cfg["use_sparse"]:
        scores["sparse"] = sparse_detector(Xp)

    final_score = fuse_scores(scores, method=cfg["fusion_method"])

    if y is not None:
        th = best_f1_threshold(y, final_score)
    else:
        th = adaptive_threshold(final_score)

    y_pred = (final_score > th).astype(int)

    if y is not None:
        print("METRICS:", evaluate(y, y_pred, final_score))

    return final_score, y_pred
