import torch.nn as nn
import torch.nn.functional as f


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(12 * 12 * 64, 12 * 12)
        self.fc2 = nn.Linear(12 * 12, 12)
        self.fc3 = nn.Linear(12, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = f.relu(self.fc1(out))
        out = f.relu(self.fc2(out))
        out = f.tanh(self.fc3(out))

        return out
