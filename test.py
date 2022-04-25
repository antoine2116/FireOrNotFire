import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import VGG16

batch_size = 4
epochs = 10

classes = ["Fire", "NoFire"]

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

testdata = ImageFolder('Test_Dataset', transform=transform)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

criterion = nn.CrossEntropyLoss()

model = VGG16()
model.load_state_dict(torch.load('fire_or_not_fire.pth'))
model.eval()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cpu")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for y in range(10)]

    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        print(f'Corrects : {n_correct} / {n_samples} samples')

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(len(classes)):
        if n_class_samples[i] == 0:
            acc = 0
        else:
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]

        print(f'Accuracy of {classes[i]}: {acc} %')
