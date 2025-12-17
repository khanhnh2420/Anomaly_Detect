from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate(y_true, y_pred, scores=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores) if scores is not None else None
    return {"Precision": precision, "Recall": recall, "F1": f1, "ROC_AUC": roc_auc}
