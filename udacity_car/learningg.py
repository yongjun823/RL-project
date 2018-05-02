import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from udacity_car.data_loader import DriveDataLoader
import torch.functional as f

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

drive_dataset = DriveDataLoader()
train_loader = torch.utils.data.DataLoader(dataset=drive_dataset,
                                           batch_size=200,
                                           shuffle=True)


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

        out = nn.ReLU(self.fc1(out))
        out = nn.ReLU(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out


cnn_net = CnnNet()

if 'model.pkl' in os.listdir('./'):
    print('model load!!')
    cnn_net = torch.load('model.pkl')

if torch.cuda.is_available():
    print('cuda start')
    cnn_net.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn_net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn_net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(drive_dataset) // batch_size, loss.data[0]))

    torch.save(cnn_net, 'model.pkl')
