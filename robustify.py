import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os

from utils import setDevice

device = setDevice()

#Using norm = infinity
#alpha = step size
def attack_pgd(model, X, y, epsilon, alpha, steps):
    
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
        
    return max_delta
    
def runEpoch():
    #Do stuff

def main():
    #Do stuff

if __name__ == "__main__":
    main()
