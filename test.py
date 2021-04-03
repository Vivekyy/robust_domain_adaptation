import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from net import Net
from utils import GrayscaleToRgb, setDevice

from tqdm import tqdm

device = setDevice()

class Tester():
    def __init__(self):
        self.model = Net().to(device)
        modelName = input("Please enter the model path: ")
        self.sucessfullyLoaded = True
        try:
            self.model.load_state_dict(torch.load(modelName))
        except:
            self.sucessfullyLoaded = False
            print("Invalid path")
        self.model.eval()
    
    def MNIST(self, robust=False):
        if(self.sucessfullyLoaded == True):
            testset = datasets.MNIST("MNIST", train=False, download=True, transform = transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]))
            testLoader = DataLoader(testset, batch_size=50, drop_last=False, num_workers = 1, pin_memory=True)

            loss_type = nn.CrossEntropyLoss()
            if robust is False:
                with torch.no_grad():
                    testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type, PGD=robust)
            else:
                testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type, PGD=robust)
            
            print("Accuracy on MNIST is: ", testAccuracy)
    
    def SVHN(self, robust=False):
        if(self.sucessfullyLoaded == True):
            testset = datasets.SVHN("SVHN", split = 'test', download=True, transform=transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()]))
            testLoader = DataLoader(testset, batch_size=50, drop_last=False, num_workers = 1, pin_memory=True)

            loss_type = nn.CrossEntropyLoss()
            if robust is False:
                with torch.no_grad():
                    testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type, PGD=robust)
            else:
                testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type, PGD=robust)
            
            print("Accuracy on SVHN is: ", testAccuracy)
    
    def USPS(self, robust=False):
        if(self.sucessfullyLoaded == True):
            testset = datasets.USPS("USPS", train=False, download=True, transform=transforms.Compose([GrayscaleToRgb(), transforms.Resize((28,28)), transforms.ToTensor()]))
            testLoader = DataLoader(testset, batch_size=50, drop_last=False, num_workers = 1, pin_memory=True)

            loss_type = nn.CrossEntropyLoss()
            if robust is False:
                with torch.no_grad():
                    testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type, PGD=robust)
            else:
                testLoss, testAccuracy = runEpoch(self.model, testLoader, loss_type, PGD=robust)
            
            print("Accuracy on USPS is: ", testAccuracy)

def test(model, dataLoader, loss_type=nn.CrossEntropyLoss(), robust=False):

    if robust is False:
        with torch.no_grad():
             testLoss, testAccuracy = runEpoch(model, dataLoader, loss_type, PGD=robust)
    else:
        testLoss, testAccuracy = runEpoch(model, dataLoader, loss_type, PGD=robust)

    return testLoss, testAccuracy

from robustify import attack_pgd

def runEpoch(model, dataLoader, loss_type, optimizer=None, PGD=False):
    loss_counter = 0
    accuracy_counter = 0

    for x, y_real in tqdm(dataLoader, leave=False):
        x, y_real = x.to(device), y_real.to(device)

        if PGD:
            delta = attack_pgd(model, x, y_real)
            x += delta
        
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

if __name__ == "__main__":
    tester = Tester()
    tester.MNIST()
    tester.SVHN()
    tester.USPS()

    ask = input("Would you like to PGD attack the datasets? (Y/N) ").lower()

    if ask == "y":
        tester.MNIST(True)
        tester.SVHN(True)
        tester.USPS(True)
