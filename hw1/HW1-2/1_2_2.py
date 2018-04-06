import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Data(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath)
        self._x = data[0][:].astype(np.double)
        self._y = data[1][:].astype(np.double)

    def __getitem__(self, index):
        x = torch.FloatTensor([self._x[index]])
        y = torch.FloatTensor([self._y[index]])
        return x, y

    def __len__(self):
        return len(self._x)

dset_train = Data("1_2_2_train.npy")
train_loader = DataLoader(dset_train, batch_size=10, num_workers=1)
dset_test = Data("1_2_2_test.npy")
test_loader = DataLoader(dset_test, batch_size=10, num_workers=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""specify loss function and optimizer"""
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

iterations_plot = []
grad_plot = []
loss_plot = []

for epoch in range(500):
    iterations_plot.append(epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        x, y = data
        x, y = Variable(x), Variable(y)

        optimizer.zero_grad()

        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
            
    print('[%d] loss: %.6f' %
        (epoch + 1, running_loss/10))
    
    loss_plot.append(running_loss / 10)

    """ compute gradient norm """
    grad_all = 0.0
    for p in net.parameters():
        gr = 0.0
        if p.grad is not None:
            gr = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += gr
    grad_plot.append(grad_all**0.5)
    
print('Finished Training')

""" plot """
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(iterations_plot, loss_plot)
axarr[0].set_ylabel('loss')
axarr[0].set_title('sinc function')
axarr[1].plot(iterations_plot, grad_plot)
axarr[1].set_ylabel('grad')
axarr[1].set_xlabel('iteration')
plt.savefig("1_2_2.png")
plt.show()