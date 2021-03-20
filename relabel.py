import os
import pandas as pd
from skimage import io, transform

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

from train_nr import setDataset
from utils import setDevice, GrayscaleToRgb
from net import Net

# Ignore warnings

# import warnings
# warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class RelabeledDataset(Dataset):

    def __init__(self, datasetPath, modelPath, transform=None):
        
        self.device = setDevice()

        self.model = Net().to(self.device)

        try:
            self.model.load_state_dict(torch.load(modelPath))
            self.successfulLoad = True
        except:
            self.successfulLoad = False
            print("Bad Model Path [Relabel]")

        self.dataset = setDataset(datasetPath)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry_x, entry_y = self.dataset[idx]
        entry_x = entry_x.to(self.device)
        entry_y = entry_y.to(self.device)

        if self.transform:
            entry_x = self.transform(entry_x)

        if self.successfulLoad:
            pred_y = self.model(entry_x.view(-1, 3, 28, 28))
            entry_y = pred_y.max(1)[1]
        
        sample = {'image': entry_x, 'label': entry_y}

        return sample
