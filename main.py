import nni
import torch
import pandas as pd
import os

from sklearn.model_selection import KFold

from config import config
from data_utils import prepare_dataset
from dataset import get_dataset, get_loader
from models.custom_cnn import CustomCNN
from train import train_epoch
from evaluate import evaluate
from analysis.visualization import plot_roc, plot_histogram
from analysis.gradcam_utils import generate_gradcam

import torch.nn as nn


def main():

    os.makedirs("results", exist_ok=True)

    params = nni.get_next_parameter()

    lr = params.get("lr", config["lr"])
    batch_size = params.get("batch_size", config["batch_size"])

    print("NNI params:", params)

    dataset_path = prepare_dataset()

    dataset = get_dataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Classes:", dataset.classes)

    kf = KFold(n_splits=5, shuffle=True)

    device = config["device"]

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        print("\n==========================")
        print(f"FOLD {fold}")
        print("==========================")

        train_loader = get_loader(dataset, train_idx, batch_size)
        val_loader = get_loader(dataset, val_idx, batch_size)

        model = CustomCNN().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(config["epochs"]):

            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            print(
                f"Epoch {epoch+1}/{config['epochs']} "
                f"Loss: {loss:.4f}"
            )

        acc, auc, preds, labels = evaluate(
            model,
            val_loader,
            device
        )

        print(f"Validation ACC: {acc:.4f}")
        print(f"Validation AUC: {auc:.4f}")

        plot_roc(labels, preds, fold)
        plot_histogram(preds, fold)

        generate_gradcam(
            model,
            val_loader,
            device,
            target_layer=model.conv2,
            fold=fold,
            num_images=5
        )

        torch.save(
            model.state_dict(),
            f"results/model_fold{fold}.pth"
        )

        results.append({
            "fold": fold,
            "accuracy": acc,
            "auc": auc
        })

    df = pd.DataFrame(results)

    df.to_excel("results/results.xlsx", index=False)

    mean_acc = df["accuracy"].mean()

    print("\n==========================")
    print("FINAL RESULTS")
    print("==========================")

    print(df)

    print("\nMean Accuracy:", mean_acc)
    print("STD Accuracy:", df["accuracy"].std())

    print("\nMean AUC:", df["auc"].mean())
    print("STD AUC:", df["auc"].std())

    # =========================
    # Report result to NNI
    # =========================
    nni.report_final_result(mean_acc)


if __name__ == "__main__":
    main()