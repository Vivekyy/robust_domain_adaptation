import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm
import numpy as np
from itertools import cycle
import os

from net import Net
from test import test
from train import setDataset
from utils import setDevice, GrayscaleToRgb, setRequiresGrad

class ADDA():
    def __init__(self, sourceNetPath, targetAsk = None, robustOut=False, robustIn=False):

        self.device = setDevice()

        self.sourceNetPath = sourceNetPath

        sourceNet = Net().to(self.device)
        targetNet = Net().to(self.device)

        try:
            sourceNet.load_state_dict(torch.load(sourceNetPath))
            targetNet.load_state_dict(torch.load(sourceNetPath))
            self.successfulLoad = True
        except:
            self.successfulLoad = False
            print("Bad model path [ADDA]")
        
        self.finalNet = sourceNet
        self.sourceNet = sourceNet.feature_identifier
        self.targetNet = targetNet.feature_identifier

        self.discriminator = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        ).to(self.device)

        self.discrimOptim = optim.Adam(self.discriminator.parameters())
        self.targetOptim = optim.Adam(self.targetNet.parameters())

        self.LossType = nn.BCEWithLogitsLoss()

        self.currentAccuracy = 0

        self.robust = robustOut
        self.robustIn = robustIn
        self.targetAsk = targetAsk

        #7 because of models/
        if sourceNetPath[7]=="r":
            self.robustIn = True
        
    def getDatasets(self):

        #Get name of dataset from path
        sourceAsk = self.sourceNetPath
        sourceAsk = sourceAsk.split('_')[1]
        sourceAsk = sourceAsk.split('.')[0] 

        sourceDataset = setDataset(sourceAsk)

        if self.targetAsk is None:
            self.targetAsk = input("Please input the name of the dataset you would like to target: ")
        targetDataset = setDataset(self.targetAsk)

        if self.robust == False:
            self.path = sourceAsk + "_to_" + self.targetAsk + ".pt"
        else:
            self.path = sourceAsk + "_to_" + self.targetAsk + "_robust.pt"

        if self.robustIn:
            self.path = "r" + self.path

        self.sourceDataset = sourceDataset
        self.targetDataset = targetDataset

        return sourceDataset, targetDataset

    def makeDataLoaders(self):
        if self.successfulLoad == False:
            return

        sourceDataset, targetDataset = self.getDatasets()

        randIndeces = np.random.permutation(len(targetDataset))
        trainIndeces = randIndeces[:int(.8*len(targetDataset))]
        valIndeces = randIndeces[int(.8*len(targetDataset)):]

        sourceDL = DataLoader(sourceDataset, batch_size=50, shuffle=True, num_workers=1, pin_memory=True)

        targetDL = DataLoader(targetDataset, batch_size=50, sampler=SubsetRandomSampler(trainIndeces), num_workers=1, pin_memory=True)
        evalDL = DataLoader(targetDataset, batch_size=50, sampler=SubsetRandomSampler(valIndeces), num_workers=1, pin_memory=True)

        return sourceDL, targetDL, evalDL

    def doEpoch(self, epoch, dataloaders):
        if self.successfulLoad == False:
            return

        (sourceDL, targetDL, evalDL) = dataloaders
        source_iter = cycle(sourceDL)
        target_iter = cycle(targetDL)

        loss_counter = 0
        accuracy_counter = 0
        for _ in tqdm(range(500), leave=False):
            setRequiresGrad(self.targetNet, False)
            setRequiresGrad(self.discriminator, True)

            #Train discriminator
            for _ in range(1):
                (source_x, _) = next(source_iter)
                (target_x, _) = next(target_iter)
                source_x = source_x.to(self.device)
                target_x = target_x.to(self.device)

                sourceFeatures = self.sourceNet(source_x).view(source_x.shape[0], -1)
                targetFeatures = self.targetNet(target_x).view(target_x.shape[0], -1)

                discrim_x = torch.cat((sourceFeatures, targetFeatures), dim=0)
                discrim_real = torch.cat((torch.ones(source_x.shape[0], device=self.device), 
                                        torch.zeros(target_x.shape[0], device=self.device)))
                discrim_predict = self.discriminator(discrim_x).squeeze()

                self.discrimOptim.zero_grad()
                discrim_loss = self.LossType(discrim_predict, discrim_real)

                discrim_loss.backward()
                self.discrimOptim.step()

                loss_counter += discrim_loss.item()
                accuracy_counter += (discrim_predict.long() == discrim_real.long()).float().mean().item()

            #Train target CNN
            setRequiresGrad(self.targetNet, True)
            setRequiresGrad(self.discriminator, False)
            for _ in range(1):
                (target_x, _) = next(target_iter)
                target_x = target_x.to(self.device)
                
                targetFeatures = self.targetNet(target_x).view(target_x.shape[0], -1)

                #Trying to fool discriminator
                tgt_goal = torch.ones(target_x.shape[0], device=self.device)
                tgt_predict = self.discriminator(targetFeatures).squeeze()

                self.targetOptim.zero_grad()
                self.discrimOptim.zero_grad()

                tgt_loss = self.LossType(tgt_predict, tgt_goal)

                tgt_loss.backward()
                self.targetOptim.step()

        mean_loss = loss_counter/500
        mean_accuracy = accuracy_counter/500

        self.finalNet.feature_identifier = self.targetNet

        print("---Epoch %s Complete [ADDA]---" % epoch)
        print("Discriminator Loss:", mean_loss)
        print("Discriminator Accuracy:", mean_accuracy)
        print()

        with torch.no_grad():
            _, valAcc = test(self.finalNet, evalDL)
    
        if valAcc > self.currentAccuracy:
            #print("New Best Accuracy: Saving Model")
            self.currentAccuracy = valAcc
        torch.save(self.finalNet.state_dict(), "models/" + self.path)
        print("Validation Accuracy: ", valAcc)
        print()

def getPath(sourceNetPath, targetAsk, robust = True, robustIn = False):

        #Get name of dataset from path
        sourceAsk = sourceNetPath
        sourceAsk = sourceAsk.split('_')[1]
        sourceAsk = sourceAsk.split('.')[0] 

        if robust == False:
            path = sourceAsk + "_to_" + targetAsk + ".pt"
        else:
            path = sourceAsk + "_to_" + targetAsk + "_robust.pt"
        
        if robustIn:
            path = "r" + path

        return path

if __name__ == "__main__":
    modelPath = input("Please input the path of your source model: ")
    adda = ADDA(modelPath)
    dataloaders = adda.makeDataLoaders()

    if os.path.exists("models/" + adda.path):
        answer = input("You already have a %s model, would you like to overwrite? (Y/N) " % adda.path)
        if answer == 'y' or answer == 'Y':
            for epoch in range(1, 11):
                adda.doEpoch(epoch, dataloaders)
    else:
        for epoch in range(1, 11):
            adda.doEpoch(epoch, dataloaders)