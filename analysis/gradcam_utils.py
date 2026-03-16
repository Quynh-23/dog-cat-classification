import torch
import matplotlib.pyplot as plt
import os

from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image


def generate_gradcam(model, loader, device, target_layer, fold, num_images=5):

    cam_extractor = GradCAM(model, target_layer)

    model.eval()

    os.makedirs("results/gradcam", exist_ok=True)

    count = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):

                if count >= num_images:
                    return

                img = images[i]

                output = outputs[i].unsqueeze(0)

                class_idx = preds[i].item()

                activation_map = cam_extractor(class_idx, output)

                heatmap = activation_map[0].cpu()

                image = img.cpu()

                save_gradcam_image(image, heatmap, fold, count)

                count += 1


def save_gradcam_image(image, heatmap, fold, idx):

    plt.figure()

    img = image.permute(1,2,0)

    plt.imshow(img)

    plt.imshow(heatmap.squeeze(), cmap="jet", alpha=0.5)

    plt.axis("off")

    plt.title("Grad-CAM")

    plt.savefig(f"results/gradcam/fold{fold}_img{idx}.png")

    plt.close()