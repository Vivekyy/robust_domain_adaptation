import numpy as np
from PIL import Image
import torch

def setDevice():
    if torch.cuda.is_available():
        deviceName = "cuda"
        if __name__ == "__main__":
            print("Using GPU")
    else:
        deviceName = "cpu"
        if __name__ == "__main__":
            print("Using CPU")
    device = torch.device(deviceName)
    return device

#From https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
#Duplicates grayscale pixels across 3 channels
class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

#Allows you to manually run models with/without gradient
def setRequiresGrad(model, boolean):
    for parameter in model.parameters():
        parameter.requires_grad = boolean