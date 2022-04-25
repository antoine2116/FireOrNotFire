import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from VGG16 import VGG16

BATCH_SIZE = 10
CLASSES = ["Fire", "NoFire"]


def main():
    # Outil de transorformation de dataset
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    ])

    # Chargement du dataset de test
    test_data = ImageFolder('Test_Dataset', transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    # Chargement de l'architecture du model
    model = VGG16()

    # Chargement du model entrainé
    model.load_state_dict(torch.load('fire_or_not_fire.pth'))
    model.eval()

    # Choix du processeur
    device = torch.device("cpu")

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for y in range(10)]

        # Itère sur les données de test
        for images, labels in test_loader:
            # Chargement des données sur le device
            images = images.to(device)
            labels = labels.to(device)

            # Test le model
            outputs = model(images)

            # Statistiques
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            current_acc = 100.0 * n_correct / n_samples
            print(f'Current accuracy: {current_acc} %')

            try:
                for i in range(BATCH_SIZE):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
            except IndexError:
                print('Index Error')

        acc = 100.0 * n_correct / n_samples

        # Afiche les résultats du test
        print("-" * 20)
        print(f'Accuracy of the network: {acc} %')

        for i in range(len(CLASSES)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {CLASSES[i]}: {acc} %')


if __name__ == '__main__':
    main()
