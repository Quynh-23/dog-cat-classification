import torch

from contrastive_loss import contrastive_loss


def train_epoch_multiscale(model, loader, optimizer, device, lambda_cl=0.01):
    model.train()

    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 🔥 Augmentation FIX (real difference)
        images_aug = images.clone()
        if torch.rand(1).item() > 0.5:
            images_aug = torch.flip(images_aug, dims=[3])

        # ===== Forward 1 =====
        # out1, proj1, _ = model(images, return_features=True)
        out1, proj1, _, _ = model(images, return_features=True)

        # ===== Forward 2 =====
        out2, proj2, _, _ = model(images_aug, return_features=True)

        # ======================
        # 🔥 CE LOSS (FIXED)
        # ======================
        ce_loss = criterion(out1, labels)

        # ======================
        # 🔥 CL LOSS
        # ======================
        cl_loss = contrastive_loss(proj1, proj2)

        loss = ce_loss + lambda_cl * cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)