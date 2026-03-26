import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from torchcam.methods import GradCAM
import os

from models.CustomResNetSE import CustomResNetSE
from dataset import get_dataset, get_loader
from data_utils import prepare_dataset
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

model_path = "results/model_fold0.pth"

model = CustomResNetSE()  # create architecture
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("✅ Model loaded successfully!")

dataset_path = prepare_dataset()
dataset = get_dataset(dataset_path)

loader = get_loader(dataset, list(range(len(dataset))), batch_size=32)

all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

pred_labels = (np.array(all_probs) > 0.5).astype(int)
acc = accuracy_score(all_labels, pred_labels)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ AUC: {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve")
plt.legend()
plt.savefig("results/roc.png", dpi=300)
plt.close()

plt.figure()
plt.hist(all_probs, bins=50)
plt.title("Prediction Confidence")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.savefig("results/histogram.png", dpi=300)
plt.close()

target_layer = model.layer3[-1].conv2

cam_extractor = GradCAM(model, target_layer=target_layer)

print("🔥 Generating Grad-CAM...")

with torch.set_grad_enabled(True):
    for i, (images, labels) in enumerate(loader):
        if i >= 1:
            break

        images = images.to(device)

        for j in range(min(5, images.size(0))):
            img = images[j].unsqueeze(0)

            output = model(img)
            class_idx = output.argmax(dim=1).item()

            activation_map = cam_extractor(class_idx, output)[0].cpu()

            img_np = img[0].cpu().permute(1, 2, 0).numpy()

            img_np = np.clip(
                img_np * [0.229, 0.224, 0.225] +
                [0.485, 0.456, 0.406],
                0, 1
            )

            plt.figure(figsize=(6, 6))
            plt.imshow(img_np)
            plt.imshow(activation_map.squeeze(), cmap='jet', alpha=0.6)
            plt.axis('off')

            plt.savefig(f"results/gradcam_{j}.png", dpi=300, bbox_inches='tight')
            plt.close()

print("✅ Grad-CAM saved!")

df = pd.DataFrame({
    "Accuracy": [acc],
    "AUC": [roc_auc]
})

df.to_excel("results/final_results.xlsx", index=False)

print("✅ Results saved to Excel!")