import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return acc, auc


def train_model(model, train_loader, val_loader, device, epochs=10):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(epochs):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

        acc, auc = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "acc": acc,
            "auc": auc
        })

        print(f"[{model.mode}] Epoch {epoch}: Acc={acc:.4f}, AUC={auc:.4f}")

    return history