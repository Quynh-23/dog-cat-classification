import matplotlib.pyplot as plt

def plot_histogram(probs):

    plt.hist(probs, bins=20)

    plt.title("Prediction Probability Histogram")

    plt.show()