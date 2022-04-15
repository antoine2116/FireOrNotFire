import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import matplotlib.pyplot as pp

from model import VGG16

lossArray = []
accArray = []
lossValArray = []
accValArray = []

batch_size = 4
epochs = 2
PATH = "fire_or_not_fire.pth"

def main():

    classes = ["Fire", "NoFire"]

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    ])

    traindata = ImageFolder('Dataset', transform=transform)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    dataset_size = len(traindata)

    testdata = ImageFolder('Test_Dataset', transform=transform)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    testset_size = len(testdata)
    criterion = nn.CrossEntropyLoss()

    model = VGG16()
    model.load_state_dict(torch.load(PATH))
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    device = torch.device("cuda")
    print("The model will be running on", device, "device\n")
    model.to(device)

    best_accuracy = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        epoch_loss = 0
        epoch_acc = 0

        # Iterate over data.
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = (running_loss / dataset_size) * 100
            epoch_acc = (running_corrects.double() / dataset_size) * 100
            print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        print('Epoch stats : Loss = {}, Accuracy = {}'.format(epoch_loss, epoch_acc))
        lossArray.append(epoch_loss)
        accArray.append(epoch_acc)

        if epoch_acc > best_accuracy :
            best_accuracy = epoch_acc
            saveModel(model)

    # with torch.no_grad():
    #     n_correct = 0
    #     n_samples = 0
    #     n_class_correct = [0 for i in range(10)]
    #     n_class_samples = [0 for i in range(10)]
    #     for images, labels in testloader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         # max returns (value ,index)
    #         _, predicted = torch.max(outputs, 1)
    #         n_samples += labels.size(0)
    #         n_correct += (predicted == labels).sum().item()
    #
    #         for i in range(2):
    #             label = labels[i]
    #             pred = predicted[i]
    #             if label == pred:
    #                 n_class_correct[label] += 1
    #             n_class_samples[label] += 1
    #
    #     acc = 100.0 * n_correct / n_samples
    #     print(f'Accuracy of the network: {acc} %')
    #
    #     for i in range(10):
    #         acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #         print(f'Accuracy of {classes[i]}: {acc} %')

        setGraph()

def saveModel(model):
    torch.save(model.state_dict(), PATH)

def setGraph():
    #Loss
    loss = pp
    acc = pp
    loss.plot(epochs, lossArray)
    loss.ylabel('Loss %')
    loss.xlabel('Epoch')
    loss.show()

    acc.plot(epochs, accArray)
    acc.ylabel('Accuraccy %')
    acc.xlabel('Epoch')
    acc.show()



if __name__ == '__main__':
    main()
