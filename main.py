import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import VGG16


def main():
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    ])

    traindata = ImageFolder('Dataset', transform=transform)
    trainloader = DataLoader(traindata, batch_size=4, shuffle=True, pin_memory=True, drop_last=False)
    dataset_size = len(traindata)

    criterion = nn.CrossEntropyLoss()

    model = VGG16()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cpu")

    for epoch in range(2):
        print('Epoch {}/{}'.format(epoch, 2 - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

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

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        print()

    torch.save(model.state_dict(), 'fire_or_not_fire.pth')


if __name__ == '__main__':
    main()
