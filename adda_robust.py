import torch
import os

from adda import ADDA, getPath
from relabel import RelabeledDataset
from robustify import robustify
from robust_reset import reset

def adda_robust(modelPath, targetAsk, robust=False):
    
    ADDApath = getPath(modelPath, targetAsk, robustIn=robust)

    if os.path.exists("models/" + ADDApath):
        answer = input("You already have a %s model, would you like to redo ADDA? (Y/N) " % ADDApath)
        if answer == 'y' or answer == 'Y':
            adda = ADDA(modelPath, robustOut=True, robustIn=robust)
            for epoch in range(1, 11):
                dataloaders = adda.makeDataLoaders()
                adda.doEpoch(epoch, dataloaders)
        else:
            reset(ADDApath)
    else:
        adda = ADDA(modelPath, robustOut=True, robustIn=robust)
        for epoch in range(1, 11):
            dataloaders = adda.makeDataLoaders()
            adda.doEpoch(epoch, dataloaders)
    
    dataPath = modelPath
    dataPath = dataPath.split('_')[1]
    dataPath = dataPath.split('.')[0]

    trainData = RelabeledDataset(dataPath, "models/" + ADDApath)
    finalModel = robustify(trainData, modelPath=modelPath)

    torch.save(finalModel.state_dict(), "models/" + ADDApath)

if __name__ == "__main__":
    modelPath = input("Please input the path of your source model: ")
    if modelPath[7] == "r":
        rBool=True
    else:
        rBool=False
    targetAsk = input("Please input the name of the dataset you would like to target: ")
    adda_robust(modelPath, targetAsk, robust=rBool)