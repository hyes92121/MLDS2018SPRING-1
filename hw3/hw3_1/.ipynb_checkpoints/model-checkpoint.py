import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """
        noise size : 100
        """
        self.fc1 = nn.Linear(100, 128*16*16)
        self.convT1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.convT2 = nn.ConvTranspose2d(64, 3, 2, 2, 0)

    def forward(self, x):
        x = F.selu(self.fc1(x))     #(batch, 128*16*16)
        x = x.view(-1, 128, 16, 16) #(batch, 128, 16, 16)
        x = F.selu(self.convT1(x))  #(batch, 64, 32, 32)
        x = F.selu(self.convT2(x))  #(batch, 3, 64, 64)
        return F.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """
        input:(batch, 3, 64, 64)
        """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(256*6*6, 1)

    def forward(self, x):
        x = F.selu(self.conv1(x)) #(batch, 32, 60, 60)
        x = F.selu(self.conv2(x)) #(batch, 64, 30, 30)
        x = F.selu(self.conv3(x)) #(batch, 128, 14, 14)
        x = F.selu(self.conv4(x)) #(batch, 256, 6, 6)
        x = x.view(-1, 256*6*6)
        x = F.selu(self.fc1(x))
        return F.sigmoid(x)




if __name__ == '__main__':
    from dataset import TrainingDataset
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import numpy as np
    import torch
    
    #dataset = TrainingDataset()
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    """
    model = Generator()
    noise = np.random.normal(0, 1, (1, 100))
    noise = Variable(torch.FloatTensor(noise))
    x = model(noise)
    """
    model = Discriminator()
    noise = np.random.normal(0, 1, (30, 3, 64, 64))
    noise = Variable(torch.FloatTensor(noise))
    x = model(noise)
    
    
    """
    for epoch in range(1):
        for batch_n, batch in enumerate(dataloader):
            batch = Variable(batch)
            x = model(batch)
            break
    """
    