from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(y_true, y_hat, y_score):
    scores = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_hat),
        'precision': precision_score(y_true, y_hat),
        'recall': recall_score(y_true, y_hat),
        'f1': f1_score(y_true, y_hat),
        'roc_auc': roc_auc_score(y_true, y_score[:, 1])
    }
    
    return scores


