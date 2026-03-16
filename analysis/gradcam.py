import torch
import cv2
import numpy as np

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

    def generate(self, input_image):

        # forward
        output = self.model(input_image)

        # backward
        output[:, output.argmax()].backward()

        gradients = self.target_layer.grad

        activations = self.target_layer.output

        weights = gradients.mean(dim=(2,3))

        cam = (weights[:, :, None, None] * activations).sum(1)

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        return cam