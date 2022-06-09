import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
print("train data size: ", len(trainSet))
print("train image type: ", type(firstTrainImage))
print("train image shape: ", firstTrainImage.shape)

# %%
print("train min: ", firstTrainImage.data.min())
print("train max: ", firstTrainImage.data.max())

# %%
firstTestImage, firstTestLabel = testSet[0]
print("test data size: ", len(testSet))
print("test image type: ", type(firstTestImage))
print("test image shape: ", firstTestImage.shape)

# %%
print("test min: ", firstTestImage.data.min())
print("test max: ", firstTestImage.data.max())

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

# %%
nInput = 784
nOutput = 10
nHidden = 128


# %%
class Net(nn.Module):
    def __init__(self, nInput, nOutput, nHidden):
        super().__init__()
        self.l1 = nn.Linear(nInput, nHidden)
        self.l2 = nn.Linear(nHidden, nOutput)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3


# %%
net = Net(nInput, nOutput, nHidden).to(device)
criterion = nn.CrossEntropyLoss()

# %%
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)

# %%
history = np.zeros((0, 5))

# %%
nEpoch = 100

for epoch in range(nEpoch):
    trainAcc, trainLoss = 0, 0
    valAcc, valLoss = 0, 0
    nTrain, nTest = 0, 0

    for inputs, labels in tqdm(trainLoader):
        nTrain += len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predicted = torch.max(outputs, 1)[1]

        trainLoss += loss.item()
        trainAcc += (predicted == labels).sum().item()

    for inputsTest, labelsTest in testLoader:
        nTest += len(labelsTest)

        inputsTest = inputsTest.to(device)
        labelsTest = labelsTest.to(device)

        outputsTest = net(inputsTest)
        lossTest = criterion(outputsTest, labelsTest)

        predictedTest = torch.max(outputsTest, 1)[1]

        valLoss += lossTest.item()
        valAcc += (predictedTest == labelsTest).sum().item()

    trainAcc /= nTrain
    valAcc /= nTest

    trainLoss *= (batchSize / nTrain)
    valLoss *= (batchSize / nTest)

    print(f"Epoch [{epoch + 1}/{nEpoch}], loss: {trainLoss:.5f} acc: {trainAcc:.5f} valLoss: {valLoss:.5f} valAcc: {valAcc:.5f}")

    items = np.array([epoch+1, trainLoss, trainAcc, valLoss, valAcc])
    history = np.vstack((history, items))


# %%
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(history[:, 0], history[:, 1], "b", label="train")
plt.plot(history[:, 0], history[:, 3], "k", label="test")
plt.xlabel("iter")
plt.ylabel("loss")
plt.title("loss carve")
plt.legend()
plt.show()


# %%
for testImages, testLabels in testLoader:
    break

testInputs = testImages.to(device)
testLabels = testLabels.to(device)
testOutputs = net(testInputs)
testPredicted = torch.max(testOutputs, 1)[1]

plt.figure(figsize=(10, 8))

for index in range(50):
    ax = plt.subplot(5, 10, index + 1)

    testImage = testImages[index]
    testLabel = testLabels[index]
    testPred = testPredicted[index]

    color = "k" if testPred == testLabel else "b"

    image = (testImage + 1) / 2

    plt.imshow(image.reshape(28, 28), cmap="gray_r")
    ax.set_title(f"{testLabel}:{testPred}", c=color)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()