import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from net import Net
from utils import GrayscaleToRgb, setDevice

from train_nr import runEpoch

class Tester():
    def __init__(self):
        device = setDevice()

        self.model = Net().to(device)
        modelName = input("Please enter the model path: ")
        self.sucessfullyLoaded = True
        try:
            self.model.load_state_dict(torch.load(modelName))
        except:
            self.sucessfullyLoaded = False
            print("Invalid path")
        self.model.eval()
    
    def MNIST(self):
        if(self.sucessfullyLoaded == True):
            testset = datasets.MNIST("MNIST", train=False, download=True, transform = transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]))
            testLoader = DataLoader(testset, batch_size=50, drop_last=False, num_workers = 1, pin_memory=True)

            loss_type = nn.CrossEntropyLoss()
            with torch.no_grad():
                testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type)
            
            print("Accuracy on MNIST is: ", testAccuracy)
    
    def SVHN(self):
        if(self.sucessfullyLoaded == True):
            testset = datasets.SVHN("SVHN", split = 'test', download=True, transform=transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()]))
            testLoader = DataLoader(testset, batch_size=50, drop_last=False, num_workers = 1, pin_memory=True)

            loss_type = nn.CrossEntropyLoss()
            with torch.no_grad():
                testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type)
            
            print("Accuracy on SVHN is: ", testAccuracy)
    
    def USPS(self):
        if(self.sucessfullyLoaded == True):
            testset = datasets.USPS("USPS", train=False, download=True, transform=transforms.Compose([GrayscaleToRgb(), transforms.Resize((28,28)), transforms.ToTensor()]))
            testLoader = DataLoader(testset, batch_size=50, drop_last=False, num_workers = 1, pin_memory=True)

            loss_type = nn.CrossEntropyLoss()
            with torch.no_grad():
                testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type)
            
            print("Accuracy on USPS is: ", testAccuracy)

def test(model, dataLoader):
    
    loss_type = nn.CrossEntropyLoss()
    with torch.no_grad():
        testLoss, testAccuracy = runEpoch(model, dataLoader, loss_type)

    return testLoss, testAccuracy

if __name__ == "__main__":
    tester = Tester()
    tester.MNIST()
    tester.SVHN()
    tester.USPS()