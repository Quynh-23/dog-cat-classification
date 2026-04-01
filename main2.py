import torch
import random
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.multiscale_net import MultiScaleNet
from utils.train_eval import train_model


# ===== SEED =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DATA =====
data_dir = "./data/PetImages"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ===== TRAIN ALL MODELS =====
results = []

for mode in ['net0', 'net1', 'net2', 'net3']:
    print(f"\n=== Training {mode} ===")

    model = MultiScaleNet(mode=mode)

    history = train_model(model, train_loader, val_loader, device, epochs=10)

    final = history[-1]

    results.append({
        "model": mode,
        "accuracy": final["acc"],
        "auc": final["auc"]
    })


# ===== SAVE CSV =====
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\nFinal Results:")
print(df)