import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """
        noise size : 100
        """
        self.f = 64
        self.convT1 = nn.ConvTranspose2d(100     , self.f*8, 4, 1, 0, bias=False)
        self.convT2 = nn.ConvTranspose2d(self.f*8, self.f*4, 4, 2, 1, bias=False)
        self.convT3 = nn.ConvTranspose2d(self.f*4, self.f*2, 4, 2, 1, bias=False)
        self.convT4 = nn.ConvTranspose2d(self.f*2, self.f  , 4, 2, 1, bias=False)
        self.convT5 = nn.ConvTranspose2d(self.f  , 3       , 4, 2, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(self.f*8)
        self.batchnorm2 = nn.BatchNorm2d(self.f*4)
        self.batchnorm3 = nn.BatchNorm2d(self.f*2)
        self.batchnorm4 = nn.BatchNorm2d(self.f)

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.activation(self.batchnorm1(self.convT1(x)))
        x = self.activation(self.batchnorm2(self.convT2(x)))
        x = self.activation(self.batchnorm3(self.convT3(x)))
        x = self.activation(self.batchnorm4(self.convT4(x)))
        x = self.convT5(x)
        return F.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """
        input:(batch, 3, 64, 64)
        """
        self.f = 64
        self.conv1 = nn.Conv2d(3       , self.f  , 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.f  , self.f*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(self.f*2, self.f*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(self.f*4, self.f*8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(self.f*8, 1       , 4, 1, 0, bias=False)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.batchnorm2 = nn.BatchNorm2d(self.f*2)
        self.batchnorm3 = nn.BatchNorm2d(self.f*4)
        self.batchnorm4 = nn.BatchNorm2d(self.f*8)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.batchnorm2(self.conv2(x)))
        x = self.activation(self.batchnorm3(self.conv3(x)))
        x = self.activation(self.batchnorm4(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(-1, 1)
        return F.sigmoid(x)


if __name__ == '__main__':
    '''
    from dataset import TrainingDataset
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import numpy as np
    import torch
    from torchsummary import summary
    
    #dataset = TrainingDataset()
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    
    model = Generator()
    #noise = np.random.normal(0, 1, (1, 100))
    #noise = Variable(torch.FloatTensor(noise))
    #x = model(noise)
    if torch.cuda.is_available():
        model.cuda()
    summary(model, input_size=(100, 1, 1))
    """
    model = Discriminator()
    #noise = np.random.normal(0, 1, (30, 3, 64, 64))
    #noise = Variable(torch.FloatTensor(noise))
    #x = model(noise)
    if torch.cuda.is_available():
        model.cuda()
    summary(model, input_size=(3, 64, 64))
    """
    """
    for epoch in range(1):
        for batch_n, batch in enumerate(dataloader):
            batch = Variable(batch)
            x = model(batch)
            break
    """
    '''
    