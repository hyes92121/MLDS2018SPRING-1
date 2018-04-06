import os
import sys
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


import MyLogger
from MyLogger import logger
from MyLogger import log_method, log_function

logger = logger.getChild(__name__)

class Trainer(object):
    logger.debug("Creating {} instance".format(__name__))
    def __init__(self, model, train_dataloader, test_dataloader=None):

        self.train_loader   = train_dataloader
        self.test_loader    = test_dataloader

        # use CUDA if available
        self.__useCUDA__    = torch.cuda.is_available()

        if self.__useCUDA__:
            self.model      = model.cuda()
        else:
            self.model      = model.cpu()

        # define hyper-parameters
        self.parameters = model.parameters()
        self.loss_fn    = F.nll_loss
        self.optimizer  = optim.Adam(self.parameters, lr=3e-4)

        # misc params
        self.start_time = str(datetime.datetime.now())
        self.loss       = None
        self.test_loss  = None


    # define the training function for one epoch
    #@log_method(logger)
    def train(self, epoch):
        # set model to training mode
        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            data, target = batch
            if self.__useCUDA__:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data).view(-1, 784), Variable(target)

            self.optimizer.zero_grad()
            output  = self.model(data)
            loss    = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()


            # print out training info
            if batch_idx % 5 == 0:
                info = self.get_training_info(
                    epoch=epoch,
                    batch_id=batch_idx,
                    batch_size=len(data),
                    total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader),
                    loss=loss.data[0]
                )
                print(info, end='\r')
                sys.stdout.write("\033[K")

        info = self.get_training_info(
            epoch=epoch,
            batch_id=batch_idx,
            batch_size=len(data),
            total_data_size=len(self.train_loader.dataset),
            n_batch=len(self.train_loader),
            loss=loss.data[0]
        )

        logger.info(info)
        # update loss for each epoch
        self.loss = loss.data[0]


    def eval(self, epoch):
        # set model to evaluation(testing) mode
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in self.test_loader:
            data, target = Variable(data).view(-1, 784), Variable(target)
            output = self.model(data)
            test_loss += self.loss_fn(output, target, size_average=False).data[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            test_loss /= len(self.test_loader.dataset)
            self.test_loss = test_loss

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


    #@log_method(logger)
    def get_training_info(self,**kwargs):
        ep = kwargs.pop("epoch", None)
        bID = kwargs.pop("batch_id", None)
        bs = kwargs.pop("batch_size", None)
        tds = kwargs.pop("total_data_size", None)
        nb = kwargs.pop("n_batch", None)
        loss = kwargs.pop("loss", None)
        info = "Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(ep, (bID+1)*bs, tds, 100.*bID/nb, loss)
        return info





