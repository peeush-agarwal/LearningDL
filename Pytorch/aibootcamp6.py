import torch
import torchvision
import torchvision.transforms as transforms

# CIFAR-10 => 10 classes of 60000(50K = Train, 10K = Test) images. https://www.cs.toronto.edu/~kriz/cifar.html
# CIFAR => Canadian Institure For Advanced Research
# Convert to tensor and then Normalize

# Load dataset from internet to Data folder
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root = './Data', train=True, transform=transforms, download=False) # download = False, because we don't want to download again
testset = torchvision.datasets.CIFAR10(root = './Data', train=False, transform=transforms, download=False)

# Image format => PIL 
# NN => composition of functions (Function creates mappings)
# DL helps me to extract features out of an image
# Perceptron => Input Layer (all neurons are fully connected) => Hidden Layer => Output Layer
# Initial weights are assigned using => Randomly, Xavier initialization*, Zero(Not used)

# Load dataset from Data folder into memory
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid((images)))
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, img):
        x = F.relu(self.pool(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('training started\n')

epochs = 2
for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        output = net(inputs)

        loss = loss_func(output, labels)

        loss.backward() # updates weights
        optimizer.step() # Reduces loss function. Helps in Backward propogation

        running_loss += loss.item()
        if i%2000 == 1999:
            print('iter:{0}, loss:{1}'.format(i, loss.item()))
            print('epoch:{0}, loss:{1}'.format(epoch+1, running_loss/2000))
            running_loss = 0.0
    print('epoch:{0}, loss:{1}'.format(epoch+1, running_loss))

print('\n training finished')