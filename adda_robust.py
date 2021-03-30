import torch
import os

from adda import ADDA
from relabel import RelabeledDataset
from robustify import robustify

if __name__ == "__main__":
    
    modelPath = input("Please input the path of your source model: ")
    adda = ADDA(modelPath, robust=True)
    dataloaders = adda.makeDataLoaders()

    if os.path.exists("models/" + adda.path):
        answer = input("You already have a %s model, would you like to redo ADDA? (Y/N) " % adda.path)
        if answer == 'y' or answer == 'Y':
            for epoch in range(1, 11):
                adda.doEpoch(epoch, dataloaders)
    else:
        for epoch in range(1, 11):
            adda.doEpoch(epoch, dataloaders)
    
    dataPath = modelPath
    dataPath = dataPath.split('_')[1]
    dataPath = dataPath.split('.')[0] 

    trainData = RelabeledDataset(dataPath, "models/" + adda.path)
    finalModel = robustify(trainData, modelPath=modelPath)

    torch.save(finalModel.state_dict(), "models/" + adda.path)