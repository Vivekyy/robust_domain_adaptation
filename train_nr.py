import numpy as np
import torch
import torchvision

from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from net import Net

from utils import GrayscaleToRgb, setDevice
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm

device = setDevice()

def setDataset(ask):
    if ask == "mnist":
        dataset = datasets.MNIST("MNIST", train=True, download=True, transform = transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]))
    elif ask == "svhn":
        dataset = datasets.SVHN("SVHN", split="train", download=True, transform=transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()]))
    elif ask == "usps":
        dataset = datasets.USPS("USPS", train=True, download=True, transform=transforms.Compose([GrayscaleToRgb(), transforms.Resize((28,28)), transforms.ToTensor()]))
    else:
        print("Invalid dataset")
        setDataset(ask)
    return dataset

def formatData(dataset, batchSize):
    randIndeces = np.random.permutation(len(dataset))
    trainIndeces = randIndeces[:int(.8*len(dataset))]
    valIndeces = randIndeces[int(.8*len(dataset)):]

    trainLoader = DataLoader(dataset, batch_size=batchSize, drop_last=True, 
                            sampler=SubsetRandomSampler(trainIndeces),
                            num_workers = 1, pin_memory=True)
    valLoader = DataLoader(dataset, batch_size=batchSize, drop_last=False, 
                            sampler=SubsetRandomSampler(valIndeces),
                            num_workers = 1, pin_memory=True)

    return trainLoader, valLoader

import torch.optim as optim

def runEpoch(model, dataLoader, loss_type, optimizer=None):
    loss_counter = 0
    accuracy_counter = 0

    for x, y_real in tqdm(dataLoader, leave=False):
        x, y_real = x.to(device), y_real.to(device)
        y_pred = model(x)
        loss = loss_type(y_pred, y_real)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_counter += loss.item()
        accuracy_counter += (y_pred.max(1)[1] == y_real).float().mean().item()
    
    mean_loss = loss_counter/len(dataLoader)
    mean_accuracy = accuracy_counter/len(dataLoader)

    return mean_loss, mean_accuracy

def main(dataset, path):
    trainLoader, valLoader = formatData(dataset, 50)
    
    model = Net().to(device)
    trainOptim = optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(trainOptim, patience=1, verbose=True)
    loss_type = nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, 31):
        print("Starting Epoch", epoch)
        model.train()
        trainLoss, trainAcc = runEpoch(model, trainLoader, loss_type, optimizer=trainOptim)

        model.eval()
        with torch.no_grad():
            valLoss, valAcc = runEpoch(model, valLoader, loss_type)

        if valAcc > best_accuracy:
            print("New Best Accuracy: Saving Epoch", epoch)
            print("Validation Accuracy: ", valAcc)
            best_accuracy = valAcc
            torch.save(model.state_dict(), "models/" + path)
        print()

        lr_schedule.step(valLoss)

import os.path as path

if __name__ == "__main__":
    ask = input("Which dataset would you like to use?").lower()
    dataset = setDataset(ask)
    path = "nr_" + ask + ".pt"
    if path.exists(path):
        answer = input("You already have a %s model, would you like to overwrite? (Y/N)" % ask)
        if answer == 'y' or 'Y':
            main(dataset, path)
    else:
        main(dataset, path)