from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate(y_true, y_pred, scores=None):
    # Dùng 'macro' hoặc 'weighted' cho multiclass
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ROC AUC cho multiclass
    roc_auc = None
    if scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, scores, multi_class='ovr')
        except ValueError:
            # Nếu scores không phù hợp multiclass
            roc_auc = None

    return {"Precision": precision, "Recall": recall, "F1": f1, "ROC_AUC": roc_auc}
