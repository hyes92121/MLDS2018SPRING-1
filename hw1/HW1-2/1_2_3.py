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
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""specify loss function and optimizer"""
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adadelta(net.parameters())


loss_plot = []
minimalRatio_plot = []  

for train_time in range(100):
    for epoch in range(30):
        for i, data in enumerate(train_loader, 0):  

            x, y = data
            x, y = Variable(x), Variable(y) 

            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward(retain_graph=True)

            """ compute gradient norm """
            grad_all = 0.0
            for p in net.parameters():
                gr = 0.0
                if p.grad is not None:
                    gr = (p.grad.cpu().data.numpy() ** 2).sum()
                grad_all += gr

            """ replace loss by gradient norm in order to minimize gradient norm """
            loss.data = torch.FloatTensor([grad_all**0.5])
            loss.backward()
            optimizer.step()

    """ store parameters for sampling """
    tmpPara = []
    for p in net.parameters():
        tmpPara.append(p.data)
    outputs = net(x)
    loss_truth = criterion(outputs, y)

    """ random sample instead of computing hessien matrix """
    minimal_sample = 0
    sampleNbr = 1000
    for sample in range(sampleNbr):
        net1 = Net()
        _ = 0
        for p in net1.parameters():
            p.data = tmpPara[_] + torch.randn(tmpPara[_].size()) / 1000.0
            _ += 1
        outputs = net1(x)
        loss = criterion(outputs, y)

        if loss.data.numpy()[0] > loss_truth.data.numpy()[0]:
            minimal_sample += 1
    """ store data for plot """
    print(str(train_time + 1) + " : minimal ratio: " + str(minimal_sample / sampleNbr))
    loss_plot.append(loss_truth.data.numpy()[0])
    minimalRatio_plot.append(minimal_sample / sampleNbr)

""" plot """
plt.scatter(minimalRatio_plot, loss_plot)
plt.title("sinc function")
plt.xlabel("minimal ratio")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("1_2_3_2.png", dpi=199)
plt.show()
print(minimalRatio_plot)
print(loss_plot)
