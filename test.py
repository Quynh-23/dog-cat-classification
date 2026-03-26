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
from models.MultiBranchResNet import MultiBranchResNetImproved
from evaluate import evaluate
from analysis.visualization import plot_roc, plot_histogram

# from train_multiscale import train_epoch_multiscale

def train_epoch_multiscale(model, loader, optimizer, device, lambda_cl=0.1):
    model.train()

    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 🔥 2 views for contrastive learning
        images_aug = images + 0.01 * torch.randn_like(images)

        # forward 1
        out1, proj1, feats1 = model(images, return_features=True)

        # forward 2
        out2, proj2, feats2 = model(images_aug, return_features=True)

        # ======================
        # 🔥 Multistage Loss
        # ======================
        ce_loss = 0
        for f in feats1:
            ce_loss += criterion(out1, labels)

        ce_loss = ce_loss / len(feats1)

        # ======================
        # 🔥 Contrastive Loss
        # ======================
        cl_loss = contrastive_loss(proj1, proj2)

        loss = ce_loss + lambda_cl * cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.cat([
        torch.diag(torch.matmul(z1, z2.T)),
        torch.diag(torch.matmul(z2, z1.T))
    ], dim=0)

    positives = positives.unsqueeze(1)

    logits = torch.cat([positives, similarity_matrix], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    logits = logits / temperature

    return F.cross_entropy(logits, labels)

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

    lr = params.get("learning_rate", config["lr"])
    batch_size = params.get("batch_size", config["batch_size"])
    lambda_cl = params.get("lambda_cl", 0.1)
    step_size = params.get("step_size", 10)
    gamma = params.get("gamma", 0.5)

    # =========================
    # DATA
    # =========================
    dataset_path = prepare_dataset()
    dataset = get_dataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Classes:", dataset.classes)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = config["device"]

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== FOLD {fold} =====")

        train_loader = get_loader(dataset, train_idx, batch_size)
        val_loader = get_loader(dataset, val_idx, batch_size)

        model = MultiBranchResNetImproved(num_classes=2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 🔥 Scheduler (tunable luôn)
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

            # 🔥 báo NNI mỗi epoch
            nni.report_intermediate_result(auc)

            if auc > best_auc:
                best_auc = auc
                best_model = copy.deepcopy(model.state_dict())

        # =========================
        # LOAD BEST MODEL
        # =========================
        model.load_state_dict(best_model)

        acc, auc, preds, labels = evaluate(model, val_loader, device)

        print(f"\nBest ACC: {acc:.4f}")
        print(f"Best AUC: {auc:.4f}")

        torch.save(
            model.state_dict(),
            f"results/model_fold_{fold}.pth"
        )

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

    # 🔥 báo kết quả cuối cho NNI
    nni.report_final_result(mean_auc)


if __name__ == "__main__":
    main()