import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
from tqdm import tqdm

from utils import setDevice
from net import Net

device = setDevice()

#Using norm = infinity
#alpha = step size
def attack_pgd(model, X, y, epsilon=.3, alpha=.01, steps=40):
    
    #delta = change to X
    best_loss = torch.zeros(y.shape[0]).to(device)
    best_delta = torch.zeros_like(X).to(device)

    restarts = 1
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)

        #Start delta in a random place within the epsilon ball
        delta.uniform_(-epsilon, epsilon)
        delta.requires_grad == True

        for _ in range(steps):
            y_pred = model(X+delta)

            loss = F.cross_entropy(y_pred, y)
            loss.backward()

            grad = delta.grad.detach()
            delta = torch.clamp(delta + alpha*torch.sign(grad), min=-epsilon, max=epsilon)

            #Avoid messing with gradient
            delta.grad.zero_()

            new_loss = F.cross_entropy(model(X+delta), y, reduction="none")
        
            if(new_loss >= best_loss):
                best_delta = delta
                best_loss = loss
        
    return best_delta

def robustify(trainData, modelPath="none"):
    if modelPath == "none":
        modelPath = input("Please input the model path")

    model = Net().to(device)
    try:
        model.load_state_dict(torch.load(modelPath))
    except:
        print("Bad model path [robustify]")
        return

    trainLoader = DataLoader(trainData, batch_size=128, num_workers=1, pin_memory=True)

    model.train()
    trainOptim = optim.SGD(model.parameters(), lr=.01)
    loss_type = nn.CrossEntropyLoss()

    epochs = 50
    for _ in tqdm(epochs):
        for (X,y) in enumerate(trainLoader):
            X, y = X.to(device), y.to(device)
            x_pgd = attack_pgd(model, X, y)

            trainOptim.zero_grad()
            y_pred = model(x_pgd)
            loss = loss_type(y_pred, y)

            loss.backward()
            trainOptim.step()
