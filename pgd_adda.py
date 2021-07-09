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
from robustify import attack_pgd

class ADDA():
    def __init__(self, sourceNetPath, targetAsk = None, robustIn=False):

        self.device = setDevice()

        self.sourceNetPath = sourceNetPath

        sourceNet = Net().to(self.device)
        targetNet = Net().to(self.device)
        testingNet = Net().to(self.device)

        try:
            sourceNet.load_state_dict(torch.load(sourceNetPath))
            targetNet.load_state_dict(torch.load(sourceNetPath))
            testingNet.load_state_dict(torch.load(sourceNetPath))
            self.successfulLoad = True
        except:
            self.successfulLoad = False
            print("Bad model path [ADDA]")
        
        self.finalNet = sourceNet
        self.testingNet = testingNet
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
        
        self.path = sourceAsk + "_to_" + self.targetAsk + "_pgd.pt"

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

            for _ in range(1):
                (source_x, source_y) = next(source_iter)
                (target_x, target_y) = next(target_iter)
                source_x, source_y = source_x.to(self.device), source_y.to(self.device)
                target_x, target_y = target_x.to(self.device), target_y.to(self.device)

                """
                #For using a pgd attack on both datasets during discriminator training
                self.testingNet.feature_identifier = self.sourceNet
                delta_s = attack_pgd(self.testingNet, source_x, source_y)

                self.testingNet.feature_identifier = self.targetNet
                delta_t = attack_pgd(self.testingNet, target_x, target_y)

                source_x += delta_s
                target_x += delta_t
                """

                sourceFeatures = self.sourceNet(source_x).view(source_x.shape[0], -1)
                targetFeatures = self.targetNet(target_x).view(target_x.shape[0], -1)

                discrim_x = torch.cat((sourceFeatures, targetFeatures), dim=0)
                discrim_real = torch.cat((torch.ones(source_x.shape[0], device=self.device), 
                                        torch.zeros(target_x.shape[0], device=self.device)))
                discrim_predict = self.discriminator(discrim_x).squeeze()

                discrim_loss = self.LossType(discrim_predict, discrim_real)

                self.discrimOptim.zero_grad()
                discrim_loss.backward()
                self.discrimOptim.step()

                loss_counter += discrim_loss.item()
                accuracy_counter += (discrim_predict.long() == discrim_real.long()).float().mean().item()

            #Train target CNN 10 times for every time you train the discriminator
            setRequiresGrad(self.targetNet, True)
            setRequiresGrad(self.discriminator, False)
            for rep in range(10):
                (target_x, target_y) = next(target_iter)
                target_x, target_y = target_x.to(self.device), target_y.to(self.device)

                #PGD attack on targetNet every other iteration
                if rep % 2 == 0:
                    self.testingNet.feature_identifier = self.targetNet
                    delta = attack_pgd(self.testingNet, target_x, target_y)
                    target_x += delta
                
                targetFeatures = self.targetNet(target_x).view(target_x.shape[0], -1)

                #Trying to fool discriminator
                discriminator_goal = torch.ones(target_x.shape[0], device=self.device)
                discrim_predict = self.discriminator(targetFeatures).squeeze()

                discrim_loss = self.LossType(discrim_predict, discriminator_goal)

                self.targetOptim.zero_grad()
                discrim_loss.backward()
                self.targetOptim.step()

        mean_loss = loss_counter/500
        mean_accuracy = accuracy_counter/500

        self.finalNet.feature_identifier = self.targetNet

        print("---Epoch %s Complete [ADDA]---" % epoch)
        print("Discriminator Loss:", mean_loss)
        print("Discriminator Accuracy:", mean_accuracy)

        with torch.no_grad():
            _, valAcc = test(self.finalNet, evalDL)
            print("Validation Accuracy: ", valAcc)
            print()
            self.currentAccuracy = valAcc
        
        torch.save(self.finalNet.state_dict(), "models/" + self.path)

def getPath(sourceNetPath, targetAsk, robustIn = False):

        #Get name of dataset from path
        sourceAsk = sourceNetPath
        sourceAsk = sourceAsk.split('_')[1]
        sourceAsk = sourceAsk.split('.')[0] 

        path = sourceAsk + "_to_" + targetAsk + "_pgd.pt"
        
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