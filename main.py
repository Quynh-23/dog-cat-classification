import os
import copy
import random
from models.MultiBranchResNet import MultiBranchResNet
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nni

from sklearn.model_selection import KFold

from config import config
from data_utils import prepare_dataset
from dataset import get_dataset, get_loader
from models.CustomResNetSE import CustomResNetSE
from train import train_epoch
from evaluate import evaluate
from analysis.visualization import plot_roc, plot_histogram
from analysis.gradcam_utils import generate_gradcam


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    set_seed(42)

    os.makedirs("results", exist_ok=True)

    # params = nni.get_next_parameter()
    params = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "dropout": 0.3
    }

    lr = params.get("learning_rate", config["lr"])
    batch_size = params.get("batch_size", config["batch_size"])

    print("NNI params:", params)

    dataset_path = prepare_dataset()
    dataset = get_dataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Classes:", dataset.classes)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    device = config["device"]

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        print(f"FOLD {fold}")

        train_loader = get_loader(dataset, train_idx, config["batch_size"])
        val_loader = get_loader(dataset, val_idx, config["batch_size"])

        model = MultiBranchResNet(num_classes=2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_auc = -1
        best_model = copy.deepcopy(model.state_dict())

        for epoch in range(config["epochs"]):

            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            acc, auc, _, _ = evaluate(
                model,
                val_loader,
                device
            )

            print(
                f"Epoch {epoch+1}/{config['epochs']} "
                f"Loss: {loss:.4f} "
                f"Val ACC: {acc:.4f} "
                f"Val AUC: {auc:.4f}"
            )

            if auc > best_auc:
                best_auc = auc
                best_model = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model)

        acc, auc, preds, labels = evaluate(
        model,
        val_loader,
        device
    )

        print(f"\nBest Validation ACC: {acc:.4f}")
        print(f"Best Validation AUC: {auc:.4f}")

        torch.save(
            model.state_dict(),
            f"results/model_fold_MultiBranch{fold}.pth"
        )

        plot_roc(labels, preds, fold)
        plot_histogram(preds, fold)

        generate_gradcam(
            model,
            val_loader,
            device,
            target_layer=model.layer4.fuse_conv,
            fold=fold,
            num_images=5
        )

        torch.save(
            model.state_dict(),
            f"results/model_fold_MultiBranch{fold}.pth"
        )

        results.append({
            "fold": fold,
            "accuracy": acc,
            "auc": auc
        })

    df = pd.DataFrame(results)
    df.to_excel("results/results_MultiBranch.xlsx", index=False)

    mean_acc = df["accuracy"].mean()
    std_acc = df["accuracy"].std()

    mean_auc = df["auc"].mean()
    std_auc = df["auc"].std()

    ci_auc = 1.96 * std_auc / (len(df) ** 0.5)

    print("FINAL RESULTS")

    print(df)

    print("\nMean Accuracy:", mean_acc)
    print("STD Accuracy:", std_acc)

    print("\nMean AUC:", mean_auc)
    print("STD AUC:", std_auc)
    print("95% CI AUC:", ci_auc)

    nni.report_final_result(mean_auc)


if __name__ == "__main__":
    main()