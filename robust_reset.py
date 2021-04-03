import torch
import os

from net import Net
from utils import setDevice

device = setDevice()

def reset(modelPath = None):
    if modelPath is None:
        modelPath = input("Please enter the path for the model you would like to reset: ")

    #Take out _robust
    resetTarget = modelPath.split('_')[0] + "_" + modelPath.split('_')[1] + "_" + modelPath.split('_')[2] + ".pt"

    if os.path.exists(resetTarget):
        ask = input("Would you like to reset to %s? " % resetTarget)
        if ask == "y" or ask == "Y":
            model = Net().to(device)
            model.load_state_dict(torch.load(resetTarget))
            torch.save(model.state_dict(), modelPath)
    else:
        print("No available reset model found")

if __name__ == "__main__":
    reset()