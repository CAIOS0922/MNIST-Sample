import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

# %%
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# %%
rootDir = "./data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.Lambda(lambda x: x.view(-1)),
])

trainSet = datasets.MNIST(
    root=rootDir,
    train=True,
    download=True,
    transform=transform,
)

testSet = datasets.MNIST(
    root=rootDir,
    train=False,
    download=True,
    transform=transform,
)

# %%
firstTrainImage, firstTrainLabel = trainSet[0]
print("data size: ", len(trainSet))
print("image type: ", type(firstTrainImage))
print("image shape: ", firstTrainImage.shape)

# %%
print("min: ", firstTrainImage.data.min())
print("max: ", firstTrainImage.data.max())

# %%
firstTestImage, firstTestLabel = testSet[0]
print("data size: ", len(testSet))
print("image type: ", type(firstTestImage))
print("image shape: ", firstTestImage.shape)

# %%
print("min: ", firstTestImage.data.min())
print("max: ", firstTestImage.data.max())

# %%
batchSize = 500

trainLoader = DataLoader(
    trainSet,
    batch_size=batchSize,
    shuffle=True,
)

testLoader = DataLoader(
    testSet,
    batch_size=batchSize,
    shuffle=True,
)

# %%
print("train batch size: ", len(trainLoader))
print("test batch size: ", len(testLoader))