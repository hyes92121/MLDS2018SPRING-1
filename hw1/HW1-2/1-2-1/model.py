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
    def __init__(self, i_dim=784, o_dim=10, depth=0):
        # object initialization
        logger.debug("Creating {} instance".format(__class__.__name__))
        super(MnistDNN, self).__init__()

        # layer architecture (input_dim, output_dim)
        # [input_layer, hidden_layers, output_layers]
        #IO = [(i_dim, 64), (64, 16), (16, o_dim)]
        # start layer definition
        self.input  = nn.Linear(i_dim, 64)
        self.layers = nn.ModuleList([nn.Linear(64, 64) for i in range(depth)])
        self.hidden = nn.Linear(64, 16)
        self.output = nn.Linear(16, o_dim)


    @log_method(logger)
    def forward(self, x):
        x = F.relu(self.input(x))
        for l in self.layers:
            x = F.relu(l(x))
        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.output(x), dim=1)

        return x

    # get unique string repr for each different net
    def get_name(self):
        net_name = self.__class__.__name__

        return net_name


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


if __name__ ==  "__main__":
    logger.setLevel(logging.DEBUG)
    t = MnistDNN()

    print(t.get_name())
    t.summary()
    print(t.parms_n())
