import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import nni

from sklearn.model_selection import KFold

from config import config
from data_utils import prepare_dataset
from dataset import get_dataset, get_loader

from evaluate import evaluate
from analysis.visualization import plot_roc, plot_histogram
from models.MultiBranchResNet import MultiBranchResNetImproved
from train_epoch_multiscale import train_epoch_multiscale


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    # =========================
    # 🔥 NNI PARAMS
    # =========================
    try:
        params = nni.get_next_parameter()
    except:
        params = {}

    print("NNI params:", params)

    lr = params.get("learning_rate", 0.0003)
    batch_size = params.get("batch_size", 32)
    lambda_cl = params.get("lambda_cl", 0.01)
    step_size = params.get("step_size", 10)
    gamma = params.get("gamma", 0.5)

    # =========================
    # DATA
    # =========================
    dataset_path = prepare_dataset()
    dataset = get_dataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Classes:", dataset.classes)

    # 🔥 FIX: KFold giảm xuống 5
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=15, shuffle=True, random_state=42)

    device = config["device"]

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== FOLD {fold} =====")

        train_loader = get_loader(dataset, train_idx, batch_size)
        val_loader = get_loader(dataset, val_idx, batch_size)

        model = MultiBranchResNetImproved(num_classes=2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

        best_auc = -1
        best_model = copy.deepcopy(model.state_dict())

        for epoch in range(config["epochs"]):

            loss = train_epoch_multiscale(
                model,
                train_loader,
                optimizer,
                device,
                lambda_cl=lambda_cl
            )

            acc, auc, _, _ = evaluate(model, val_loader, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{config['epochs']} "
                f"Loss: {loss:.4f} "
                f"Val ACC: {acc:.4f} "
                f"Val AUC: {auc:.4f}"
            )

            nni.report_intermediate_result(auc)

            if auc > best_auc:
                best_auc = auc
                best_model = copy.deepcopy(model.state_dict())

        # ===== Load best =====
        model.load_state_dict(best_model)

        acc, auc, preds, labels = evaluate(model, val_loader, device)

        print(f"\nBest ACC: {acc:.4f}")
        print(f"Best AUC: {auc:.4f}")

        torch.save(model.state_dict(), f"results/model_fold_{fold}.pth")

        plot_roc(labels, preds, fold)
        plot_histogram(preds, fold)

        results.append({
            "fold": fold,
            "accuracy": acc,
            "auc": auc
        })

    # =========================
    # FINAL RESULT
    # =========================
    df = pd.DataFrame(results)
    df.to_excel("results/results.xlsx", index=False)

    mean_acc = df["accuracy"].mean()
    std_acc = df["accuracy"].std()

    mean_auc = df["auc"].mean()
    std_auc = df["auc"].std()

    ci_auc = 1.96 * std_auc / (len(df) ** 0.5)

    print("\n===== FINAL RESULTS =====")
    print(df)

    print(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"95% CI AUC: {ci_auc:.4f}")

    nni.report_final_result(mean_auc)


if __name__ == "__main__":
    main()