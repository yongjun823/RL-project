import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from udacity_car.data_loader import DriveDataLoader
from udacity_car.model import CnnNet

# Hyper Parameters
num_epochs = 20
batch_size = 200
learning_rate = 0.0001

drive_dataset = DriveDataLoader()
train_loader = torch.utils.data.DataLoader(dataset=drive_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

cnn_net = CnnNet()

if 'mode1l.pkl' in os.listdir('./'):
    print('model load!!')
    cnn_net = torch.load('mode1l.pkl')

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

    torch.save(cnn_net, 'mode1l.pkl')
