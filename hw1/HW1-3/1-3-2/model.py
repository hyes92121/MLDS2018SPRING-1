import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import logging
from MyLogger import logger
from MyLogger import log_method, log_function

logger = logger.getChild(__name__)



class MnistDNN(nn.Module):
    def __init__(self, i_dim=784, o_dim=10, depth=0, width=16):
        # object initialization
        logger.debug("Creating {} instance".format(__class__.__name__))
        super(MnistDNN, self).__init__()

        # layer architecture (input_dim, output_dim)
        # [input_layer, hidden_layers, output_layers]
        #IO = [(i_dim, 64), (64, 16), (16, o_dim)]
        # start layer definition
        self.input  = nn.Linear(i_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for i in range(depth)])
        self.output = nn.Linear(width, o_dim)
        self.depth  = depth


    @log_method(logger)
    def forward(self, x):
        x = F.relu(self.input(x))
        for l in self.layers:
            x = F.relu(l(x))
        x = F.log_softmax(self.output(x), dim=1)

        return x

    # get unique string repr for each different net
    def get_name(self):
        net_name = self.__class__.__name__

        return "{}_{}".format(net_name, self.depth)


    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print("Model Parameters")
        print(self.parameters)
        print("Trainable parameters: {}".format(params))


    def parms_n(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return params


class CifarCNN(nn.Module):
    def __init__(self, depth=5):
        logger.debug("Creating {} instance".format(__class__.__name__))
        super(CifarCNN, self).__init__()

        self.depth = depth

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=depth, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=depth, out_channels=depth//2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=depth//2, out_channels=depth//4, kernel_size=3, stride=2, padding=1)
        self.conv3_drop = nn.Dropout2d()
        self.fc1   = nn.Linear(depth*4, depth)
        self.fc2   = nn.Linear(depth, 10)


    def forward(self, x):
        # 32x32 -> 16x16
        x = self.conv1(x)
        # 16x16 -> 8x8
        x = self.conv2(x)
        # 8x8 -> 4x4
        x = self.conv3(x)
        x = self.conv3_drop(x)
        _, C, H, W = x.shape
        # flatten
        x = x.view(-1, C*H*W)
        x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

    def parms_n(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return params

        # get unique string repr for each different net
    def get_name(self):
        net_name = self.__class__.__name__
        return "{}_{}".format(net_name, self.depth)

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print("Model Parameters")
        print(self.parameters)
        print("Trainable parameters: {}".format(params))

if __name__ ==  "__main__":
    logger.setLevel(logging.DEBUG)
    t = MnistCNN(depth=128)

    a = Variable(torch.rand(2, 3, 32, 32))

    print(t.parms_n())
    print(t(a).shape)
    print(t(a))