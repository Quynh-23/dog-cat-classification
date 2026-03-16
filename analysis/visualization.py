import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(labels, preds, fold):

    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Fold {fold}")
    plt.legend()

    plt.savefig(f"results/roc_fold{fold}.png")
    plt.close()