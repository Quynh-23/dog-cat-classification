from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def compute_metrics(y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_prob)

    return acc, auc