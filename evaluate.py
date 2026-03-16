import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate(model, loader, device):

    model.eval()

    preds = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:,1]

            preds.extend(probs.cpu().numpy())
            labels_list.extend(labels.numpy())

    acc = accuracy_score(labels_list, [p>0.5 for p in preds])
    auc = roc_auc_score(labels_list, preds)

    return acc, auc, preds, labels_list