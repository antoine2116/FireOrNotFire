import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam

from VGG16 import VGG16

lossArray = []
accArray = []

batch_size = 10
epochs = 100
PATH = "fire_or_not_fire.pth"


def main():
    classes = ["Fire", "NoFire"]

    transform = transforms.Compose([
        transforms.Resize([224, 244]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    ])

    train_data = ImageFolder('Dataset', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    dataset_size = len(train_data)

    criterion = nn.CrossEntropyLoss()

    model = VGG16()
    model.load_state_dict(torch.load(PATH))

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    device = torch.device("cuda")
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0
        epoch_loss = 0
        epoch_acc = 0

        # Iterate over data.
        for inputs, labels in train_loader:
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
            epoch_acc = (running_corrects / dataset_size) * 100

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        print('Epoch stats : Loss = {}, Accuracy = {}'.format(epoch_loss, epoch_acc))

        lossArray.append(epoch_loss)
        accArray.append(epoch_acc.cpu())

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save(model.state_dict(), PATH)

        # Plot
        epochs_array = [i for i in range(0, epoch + 1)]
        plt.plot(epochs_array, lossArray, label="Loss")
        plt.plot(epochs_array, accArray, label="Accuracy")
        plt.xlabel("Epochs #")
        plt.ylabel("%")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
