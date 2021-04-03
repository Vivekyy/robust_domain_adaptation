import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
from tqdm import tqdm

from utils import setDevice
from net import Net

device = setDevice()

def makeDataLoaders(targetDataset, customDataset=True):
        randIndeces = np.random.permutation(len(targetDataset))
        trainIndeces = randIndeces[:int(.8*len(targetDataset))]
        valIndeces = randIndeces[int(.8*len(targetDataset)):]

        if customDataset:
            targetDL = DataLoader(targetDataset, batch_size=50, sampler=SubsetRandomSampler(trainIndeces), num_workers=0, pin_memory=True)
            evalDL = DataLoader(targetDataset, batch_size=50, sampler=SubsetRandomSampler(valIndeces), num_workers=0, pin_memory=True)
        else:
            targetDL = DataLoader(targetDataset, batch_size=50, sampler=SubsetRandomSampler(trainIndeces), num_workers=1, pin_memory=True)
            evalDL = DataLoader(targetDataset, batch_size=50, sampler=SubsetRandomSampler(valIndeces), num_workers=1, pin_memory=True)

        return targetDL, evalDL

#Using norm = infinity
#alpha = step size
def attack_pgd(model, X, y, epsilon=.3, alpha=.01, steps=20):
    
    #delta = change to X
    best_loss = torch.zeros(y.shape[0]).to(device)
    best_delta = torch.zeros_like(X).to(device)

    restarts = 1
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)

        #Start delta in a random place within the epsilon ball
        delta.uniform_(-epsilon, epsilon)
        delta.requires_grad = True

        for _ in range(steps):
            y_pred = model(X+delta)

            loss = F.cross_entropy(y_pred, y)
            loss.backward()

            grad = delta.grad.detach()

            #Use d to keep delta as a leaf tensor
            d = delta.clone()
            d = torch.clamp(d + alpha*torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d

            #Avoid messing with gradient
            delta.grad.zero_()

            new_loss = F.cross_entropy(model(X+delta), y, reduction="none")
        
            best_delta[new_loss >= best_loss] = delta.detach()[new_loss >= best_loss]
            best_loss = torch.max(best_loss, new_loss)
        
    return best_delta

from test import test

def robustify(trainData, modelPath=None, model=None, customDataset=True):
    if modelPath is None and model is None:
        modelPath = input("Please input the model path")
    
    if model is None:
        model = Net().to(device)
        try:
          model.load_state_dict(torch.load(modelPath))
        except:
           print("Bad model path [Robustify]")
           return

    trainLoader, evalLoader = makeDataLoaders(trainData, customDataset)

    trainOptim = optim.SGD(model.parameters(), lr=.01)
    loss_type = nn.CrossEntropyLoss()

    epochs = 15
    for epoch in range(1, epochs+1):
        loss_counter = 0

        
        model.train()
        for (X,y) in tqdm(trainLoader, leave=False):
            X, y = X.to(device), y.to(device)
            x_pgd = attack_pgd(model, X, y)

            trainOptim.zero_grad()
            y_pred = model(x_pgd)
            loss = loss_type(y_pred, y)

            loss.backward()
            trainOptim.step()

            loss_counter += loss.item()

        meanLoss = loss_counter/len(trainLoader)
        print("---Epoch %s Complete [PGD]---" % epoch)
        print("PGD Loss: ", meanLoss)

        model.eval()
        with torch.no_grad():
            _, valAcc = test(model, evalLoader)
            print("Validation Accuracy: ", valAcc)
            print()
    
    return model
