import torch.nn as nn
import torch.nn.functional as F


class FireNet(nn.Module):
    def __init__(self):
        super(FireNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.AvgPool2d(2, stride=None)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(2304, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop1(x)
        x = x.view(-1, 6 * 6 * 64)
        x = F.relu(self.dense1(x))
        x = self.drop2(x)
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x



