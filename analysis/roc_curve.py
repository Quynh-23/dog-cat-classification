import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc(y_true, y_prob):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.plot(fpr, tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.title("ROC Curve")

    plt.show()