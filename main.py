import torch
import pandas as pd
import os

from sklearn.model_selection import KFold

from config import config
from data_utils import prepare_dataset
from dataset import get_dataset, get_loader
from models import CustomCNN
from train import train_epoch
from evaluate import evaluate
from analysis.visualization import plot_roc, plot_histogram
from analysis.gradcam_utils import generate_gradcam

import torch.nn as nn


def main():

    os.makedirs("results", exist_ok=True)

    # =========================
    # Dataset
    # =========================
    dataset_path = prepare_dataset()

    dataset = get_dataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Classes:", dataset.classes)


    # =========================
    # K-Fold Setup
    # =========================
    kf = KFold(n_splits=5, shuffle=True)

    device = config["device"]

    results = []


    # =========================
    # Fold Loop
    # =========================
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        print("\n==========================")
        print(f"FOLD {fold}")
        print("==========================")

        train_loader = get_loader(dataset, train_idx, config["batch_size"])
        val_loader = get_loader(dataset, val_idx, config["batch_size"])

        model_path = f"results/model_fold{fold}.pth"

        model = CustomCNN().to(device)

        # =========================
        # Skip training if model exists
        # =========================
        if os.path.exists(model_path):

            print("Model already exists. Loading checkpoint...")

            model.load_state_dict(torch.load(model_path))

        else:

            print("Training model...")

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["lr"]
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

            torch.save(model.state_dict(), model_path)


        # =========================
        # Evaluation
        # =========================
        acc, auc, preds, labels = evaluate(
            model,
            val_loader,
            device
        )

        print(f"Validation ACC: {acc:.4f}")
        print(f"Validation AUC: {auc:.4f}")


        # =========================
        # Visualization
        # =========================
        plot_roc(labels, preds, fold)

        plot_histogram(preds, fold)


        # =========================
        # Grad-CAM
        # =========================
        generate_gradcam(
            model,
            val_loader,
            device,
            target_layer=model.conv2,
            fold=fold,
            num_images=5
        )


        # =========================
        # Save fold result
        # =========================
        results.append({
            "fold": fold,
            "accuracy": acc,
            "auc": auc
        })


    # =========================
    # Final statistics
    # =========================
    df = pd.DataFrame(results)

    df.to_excel("results/results.xlsx", index=False)

    print("\n==========================")
    print("FINAL RESULTS")
    print("==========================")

    print(df)

    print("\nMean Accuracy:", df["accuracy"].mean())
    print("STD Accuracy:", df["accuracy"].std())

    print("\nMean AUC:", df["auc"].mean())
    print("STD AUC:", df["auc"].std())


if __name__ == "__main__":
    main()