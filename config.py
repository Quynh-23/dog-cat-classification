import torch

config = {
    "image_size": 224,
    "batch_size": 64,
    "epochs": 5,
    "lr": 0.001,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}