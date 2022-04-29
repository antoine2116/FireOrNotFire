import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from firenet import FireNet

# Global variables
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005

PATH = "trained_model.pth"


def main():
    # Transforms datasets
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training data
    train_data = ImageFolder('Dataset', transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    # Load validation data
    valid_data = ImageFolder('Test_Dataset', transform=transform)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    # Declare model
    model = FireNet()

    # Use CUDA device
    model = model.cuda()

    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Arrays for plots
    train_accuracies = []
    train_losses = []
    valid_accuracies = []
    valid_losses = []

    # Min valid loss for saving
    min_valid_loss = 1

    for epoch in range(1, (EPOCHS + 1)):
        train_loss = 0.0
        valid_loss = 0.0
        train_accuracy = 0.0
        valid_accuracy = 0.0

        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Load images to device
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            # Pass images through the CNN
            output = model(data)

            # Get predictions
            _, preds = torch.max(output, 1)

            # Loss calculation
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            # Batch statistics
            train_accuracy = train_accuracy + torch.sum(preds == target.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        # Validation
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # Load images to device
            data, target = data.cuda(), target.cuda()

            # Pass images through the CNN
            output = model(data)

            # Get predictions
            _, preds = torch.max(output, 1)

            # Loss calculation
            loss = criterion(output, target)

            # Batch statistics
            valid_accuracy = valid_accuracy + torch.sum(preds == target.data)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # Epochs statistics

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        train_accuracy = train_accuracy / len(train_loader.dataset)
        valid_accuracy = valid_accuracy / len(valid_loader.dataset)

        # Append statistics for the plots
        train_accuracies.append(train_accuracy.cpu())
        train_losses.append(train_loss.cpu())
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

        # Save model when loss is lower than the previous min loss
        if valid_loss <= min_valid_loss:
            print('Saving model')
            torch.save(model.state_dict(), PATH)
            min_valid_loss = valid_loss

        xmaxval = int(epoch - 1) if epoch != 1 else 1

        # Accuracy plot
        plt.figure()
        plt.plot(train_accuracies, label="Train accuracy", marker ="o")
        plt.plot(valid_accuracies, label="Validation accuracy", marker ="o")
        plt.xlabel("Epoch #")
        plt.xlim([0, xmaxval])
        plt.legend()
        plt.show()

        # Loss plot
        plt.figure()
        plt.plot(train_losses, label="Train loss", marker ="o")
        plt.plot(valid_losses, label="Valid loss", marker ="o")
        plt.xlabel("Epoch #")
        plt.xlim([0, xmaxval])
        plt.legend()

        plt.show()


if __name__ == '__main__':
    main()
