import torch
import matplotlib.pyplot as plt
import os

from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image


def generate_gradcam(model, loader, device, target_layer, fold, num_images=5):
    cam_extractor = GradCAM(model, target_layer=target_layer)
    
    model.eval()
    os.makedirs("results/gradcam", exist_ok=True)
    
    count = 0

    for images, labels in loader:
        if count >= num_images:
            break

        images = images.to(device)
        images.requires_grad_(True)           

        with torch.set_grad_enabled(True):
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if count >= num_images:
                    return

                class_idx = preds[i].item()

                activation_map = cam_extractor(class_idx, outputs[i].unsqueeze(0))

                heatmap = activation_map[0].cpu().detach()   
                img = images[i].cpu().detach()

                save_gradcam_image(img, heatmap, fold, count)
                count += 1


def save_gradcam_image(image, heatmap, fold, idx):
    plt.figure(figsize=(6, 6))
    
    img = image.permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  
    img = img.clip(0, 1)

    plt.imshow(img)
    plt.imshow(heatmap.squeeze(), cmap='jet', alpha=0.6)  
    plt.axis("off")
    plt.title(f"Grad-CAM - Fold {fold} - Image {idx}")
    
    plt.savefig(f"results/gradcam/fold{fold}MultiBranch_img{idx}.png", dpi=300, bbox_inches='tight')
    plt.close()