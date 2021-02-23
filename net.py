import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_identifier = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 5),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=.4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Dropout(p=.4),
            nn.Linear(500, 10)
        )

    def forward(self,x):
        features = self.feature_identifier(x)
        features = features.view(x.shape[0], -1)
        probs = self.classifier(features)
        return probs
