import torch
import os

from adda import ADDA, getPath
from relabel import RelabeledDataset
from robustify import robustify
from robust_reset import reset

if __name__ == "__main__":
    
    modelPath = input("Please input the path of your source model: ")
    targetAsk = input("Please input the name of the dataset you would like to target: ")
    ADDApath = getPath(modelPath, targetAsk)

    if os.path.exists("models/" + ADDApath):
        answer = input("You already have a %s model, would you like to redo ADDA? (Y/N) " % ADDApath)
        if answer == 'y' or answer == 'Y':
            for epoch in range(1, 11):
                adda = ADDA(modelPath, robust=True)
                dataloaders = adda.makeDataLoaders()
                adda.doEpoch(epoch, dataloaders)
        else:
            reset(ADDApath)
    else:
        for epoch in range(1, 11):
            adda = ADDA(modelPath, robust=True)
            dataloaders = adda.makeDataLoaders()
            adda.doEpoch(epoch, dataloaders)
    
    dataPath = modelPath
    dataPath = dataPath.split('_')[1]
    dataPath = dataPath.split('.')[0]

    trainData = RelabeledDataset(dataPath, "models/" + ADDApath)
    model_5, model_10, finalModel = robustify(trainData, modelPath=modelPath)

    torch.save(model_5.state_dict(), "models/5_" + ADDApath)
    torch.save(model_10.state_dict(), "models/10_" + ADDApath)
    torch.save(finalModel.state_dict(), "models/" + ADDApath)