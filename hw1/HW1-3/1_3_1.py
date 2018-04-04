import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class Data(Dataset):
    def __init__(self, train):
        tmp = datasets.MNIST('./MNIST_data', train=train, download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        self._x = []
        self._y = []
        #for i in range(len(tmp)):
        for i in range(1000): # 60000 training data is hard to train, use 1000 instead.
            self._x.append(tmp[i][0])
            
            """ change training data's label to random """
            if (train):
                self._y.append(np.random.randint(10))
            else:
                self._y.append(tmp[i][1])
            
    def __getitem__(self, index):
        return self._x[index], self._y[index]

    def __len__(self):
        return len(self._x)

dset_train = Data(True)
train_loader = DataLoader(dset_train, batch_size=64, num_workers=1)
dset_test = Data(False)
test_loader = DataLoader(dset_test, batch_size=64, num_workers=1)

hidden = 256
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1*28*28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 1*28*28)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

""" initialize """
model = Net()
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


train_loss_plot = []
test_loss_plot = []
epoch_plot = []


def train_loss():
    model.eval()
    train_loss = 0
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        train_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss

    train_loss /= len(train_loader.dataset)
    print('\ntrain set: Average loss: {:.4f}\n'.format(train_loss))
    train_loss_plot.append(train_loss)
    model.train()

def test_loss():
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    test_loss_plot.append(test_loss)
    model.train()

""" train """
for epoch in range(1, 101):
    train(epoch)
    train_loss()
    test_loss()
    epoch_plot.append(epoch)

""" plot """
plt.plot(epoch_plot, train_loss_plot, label='train')
plt.plot(epoch_plot, test_loss_plot, label='test')
plt.legend(loc='upper left')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("1_3_1.png")
plt.show()