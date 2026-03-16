import pandas as pd
import os

def save_results(model_name, acc, auc):

    file = "experiments/results.csv"

    data = {

        "model": model_name,
        "accuracy": round(acc,3),
        "auc": round(auc,3)

    }

    df = pd.DataFrame([data])

    if os.path.exists(file):

        df.to_csv(file, mode='a', header=False, index=False)

    else:

        df.to_csv(file, index=False)