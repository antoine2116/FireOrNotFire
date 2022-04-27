import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam

from VGG16 import VGG16

class_names = ['Fire', 'NoFire']

batch_size = 10
EPOCHS = 100
PATH = "fire_or_not_fire.pth"


def main():
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder('Dataset', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

    valid_data = ImageFolder('Test_Dataset', transform=transform)
    valid_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    dataset_size = len(train_data)

    model = VGG16()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_accuracy_list = []
    train_loss_list = []
    valid_accuracy_list = []
    valid_loss_list = []

    valid_loss_min = np.Inf

    for epoch in range(1, (EPOCHS + 1)):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_acc = train_acc + torch.sum(preds == target.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):

            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)

            valid_acc = valid_acc + torch.sum(preds == target.data)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        valid_acc = valid_acc / len(valid_loader.dataset)

        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_accuracy_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

        print(
            'Epoch: {} \tTraining Acc: {:6f} \tTraining Loss: {:6f} \tValidation Acc: {:6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_acc,
                train_loss,
                valid_acc,
                valid_loss
            ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), PATH)
            valid_loss_min = valid_loss

        plt.figure()

        plt.plot(train_accuracy_list.cpu(), label="train_acc")
        plt.plot(valid_accuracy_list.cpu(), label="valid_acc")

        plt.title("Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left")

        plt.show()


if __name__ == '__main__':
    main()
