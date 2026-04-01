import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


def save_heatmap(model, input_tensor, original_img, filename):
    target_layer = model.up_b3.conv[0]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)

    cv2.imwrite(filename, visualization)


def plot_histogram(features):
    arr = features.detach().cpu().numpy().flatten()

    plt.hist(arr, bins=50)
    plt.title("Feature Activation Histogram")
    plt.savefig("histogram.png")
    plt.close()