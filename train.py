import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_prof import FireNet

BATCH_SIZE = 100
EPOCHS = 100
LEARNING_RATE = 0.0005

PATH = "fire_or_not_fire_b100_lr0005.pth"


def main():
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder('Dataset', transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    valid_data = ImageFolder('Test_Dataset', transform=transform)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    model = FireNet()
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_accuracies = []
    valid_accuracies = []
    valid_losses = []

    min_valid_loss = 1

    for epoch in range(1, (EPOCHS + 1)):
        train_loss = 0.0
        valid_loss = 0.0
        train_accuracy = 0.0
        valid_accuracy = 0.0

        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_accuracy = train_accuracy + torch.sum(preds == target.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        # Validation
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)

            valid_accuracy = valid_accuracy + torch.sum(preds == target.data)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        train_accuracy = train_accuracy / len(train_loader.dataset)
        valid_accuracy = valid_accuracy / len(valid_loader.dataset)

        train_accuracies.append(train_accuracy.cpu())
        valid_accuracies.append(valid_accuracy.cpu())
        valid_losses.append(valid_loss.cpu())

        print(
            'Epoch: {} \tTraining Accuracy: {:6f} \tTraining Loss: {:6f} \tValidation Accuracy: {:6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_accuracy,
                train_loss,
                valid_accuracy,
                valid_loss
            ))

        if valid_loss <= min_valid_loss:
            print('Saving model')
            torch.save(model.state_dict(), PATH)
            min_valid_loss = valid_loss

        # Accuracy plot
        plt.figure()
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(valid_accuracies, label="Validation accuracy")
        plt.xlabel("Epoch #")
        plt.legend()
        plt.show()

        # Loss plot
        plt.figure()
        plt.plot(valid_losses, label="Valid loss")
        plt.xlabel("Epoch #")
        plt.legend()

        plt.show()


if __name__ == '__main__':
    main()
