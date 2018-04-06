import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from MyLogger import logger
from MyLogger import log_method, log_function

logger = logger.getChild(__name__)

class ShallowNet(nn.Module):
    logger.debug("Creating {} instance".format(__name__))
    def __init__(self, width=100, input_dim=1, output_dim=1):
        super(ShallowNet, self).__init__()

        self.width  = width

        self.input  = nn.Linear(input_dim, width)
        self.output = nn.Linear(width, output_dim)

    #@log_method(logger)
    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.output(x)

        return x

    # get unique string repr for each different net
    def get_name(self):
        net_name = self.__class__.__name__
        return "{}_w{}".format(net_name, self.width)

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



class DeepNet(nn.Module):
    logger.debug("Creating {} instance".format(__name__))
    def __init__(self, width=100, depth=2, input_dim=1, output_dim=1):
        super(DeepNet, self).__init__()

        self.width  = width
        self.depth  = depth

        self.input  = nn.Linear(input_dim, width)
        self.linear = nn.ModuleList([nn.Linear(width, width) for _  in range(depth)])
        self.output = nn.Linear(width, output_dim)

    @log_method(logger)
    def forward(self, x):
        x = F.relu(self.input(x))

        for idx, l in enumerate(self.linear):
            x = F.relu(self.linear[idx](x))

        x = self.output(x)

        return x

    # get unique string repr for each different net
    def get_name(self):
        net_name = self.__class__.__name__
        return "{0}_d{2}w{1}".format(net_name, self.width, self.depth)

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

if __name__ == "__main__":
    s = ShallowNet(200)

    a = Variable(torch.Tensor([1, 2]))

    a.data = a.data.view(len(a.data), 1)

    output = s(a)
    print(output)

    print(s._get_name())











