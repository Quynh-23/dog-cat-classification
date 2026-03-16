import torch
import pandas as pd

from config import config
from dataset import get_dataloader
from data.data_utils import prepare_dataset
from models.custom_cnn import CustomCNN
from train import train

import torch.nn as nn

def main():

    dataset_path = prepare_dataset()

    train_loader = get_dataloader(
        dataset_path,
        config["batch_size"]
    )

    device = config["device"]

    model = CustomCNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    criterion = nn.CrossEntropyLoss()

    results = []

    for epoch in range(config["epochs"]):

        loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        print(f"Epoch {epoch} Loss {loss}")

    torch.save(model.state_dict(), "results/model.pth")

    results.append({"loss": loss})

    df = pd.DataFrame(results)

    df.to_excel("results/results.xlsx")


if __name__ == "__main__":
    main()