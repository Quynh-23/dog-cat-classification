import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve

from config import config
from data_utils import prepare_dataset
from dataset import get_dataset, get_loader
from models.custom_cnn import CustomCNN
from train import train_epoch
from evaluate import evaluate

import torch.nn as nn


def main():

    dataset_path = prepare_dataset()

    dataset = get_dataset(dataset_path)

    kf = KFold(n_splits=5, shuffle=True)

    results = []

    device = config["device"]

    for fold,(train_idx,val_idx) in enumerate(kf.split(dataset)):

        print(f"Fold {fold+1}")

        train_loader = get_loader(dataset,train_idx,config["batch_size"])
        val_loader = get_loader(dataset,val_idx,config["batch_size"])

        model = CustomCNN().to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])

        criterion = nn.CrossEntropyLoss()

        for epoch in range(config["epochs"]):

            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            print(f"Epoch {epoch} Loss {loss}")

        acc,auc,preds,labels = evaluate(model,val_loader,device)

        print("ACC:",acc,"AUC:",auc)

        results.append({"fold":fold,"acc":acc,"auc":auc})

        fpr,tpr,_ = roc_curve(labels,preds)

        plt.plot(fpr,tpr)

        torch.save(model.state_dict(),f"results/model_fold{fold}.pth")

    df = pd.DataFrame(results)

    df.to_excel("results/results.xlsx")

    print(df)

    print("Mean ACC:",df["acc"].mean())
    print("STD ACC:",df["acc"].std())


if __name__ == "__main__":
    main()