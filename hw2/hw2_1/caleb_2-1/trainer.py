import os
import sys
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from customloss import CustomLoss

import logging
from MyLogger import logger

logger.setLevel(logging.INFO)
logger = logger.getChild(__name__)


class Trainer(object):
    def __init__(self, model, train_dataloader, test_dataloader=None, helper=None):

        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

        # Use cuda is available
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
        else:
            self.model = model.cpu()


        # define hyper parameters
        self.parameters = model.parameters()
        self.loss_fn = CustomLoss()
        self.loss = None
        self.optimizer = optim.Adam(self.parameters, lr=3e-4)

        # TODO: testing only pls remove after testing
        self.helper = helper


    def train(self, epoch, check_result=False):
        self.model.train()

        test_avi, test_truth = None, None

        for batch_idx, batch in enumerate(self.train_loader):
            # prepare data
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # start training process
            self.optimizer.zero_grad()
            seq_logProb, seq_predictions = self.model(avi_feats, target_sentences=ground_truths, mode='train', steps=epoch)
            ground_truths = ground_truths[:, 1:]  # eliminate <SOS>

            if check_result:
                if test_avi is None or test_truth is None:
                    test_avi = avi_feats[:3]
                    test_truth = ground_truths[:3]


            loss = self.loss_fn(seq_logProb, ground_truths, lengths)
            loss.backward()
            self.optimizer.step()

            # print out training info
            if (batch_idx+1):
                info = self.get_training_info(
                    epoch=epoch,
                    batch_id=batch_idx,
                    batch_size=len(lengths),
                    total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader),
                    loss=loss.data[0]
                )
                print(info, end='\r')
                sys.stdout.write("\033[K")

        info = self.get_training_info(
            epoch=epoch,
            batch_id=batch_idx,
            batch_size=len(lengths),
            total_data_size=len(self.train_loader.dataset),
            n_batch=len(self.train_loader),
            loss=loss.data[0]
        )

        logger.info(info)
        # update loss for each epoch
        self.loss = loss.data[0]

        # TODO: testing only pls remove after testing
        if check_result:
            _, test_predictions = self.model(test_avi, mode='train', target_sentences=test_truth ,steps=epoch)
            result = [' '.join(self.helper.index2sentence(s)) for s in test_predictions]
            logger.info('Training Result: \n{} \n{}\n{}\n'.format(result[0], result[1], result[2]))
            truth = [' '.join(self.helper.index2sentence(s)) for s in test_truth]
            logger.info('Ground Truth: \n{} \n{}\n{}\n'.format(truth[0], truth[1], truth[2]))



    def eval(self, check_result=False):
        # set model to evaluation(testing) mode
        self.model.eval()

        test_predictions, test_truth = None, None

        for batch_idx, batch in enumerate(self.test_loader):
            # prepare data
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)


            # start inferencing process
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            ground_truths = ground_truths[:, 1:]

            test_predictions = seq_predictions[:3]
            test_truth = ground_truths[:3]

            break

        # TODO: testing only pls remove after testing
        if check_result:
            result = [' '.join(self.helper.index2sentence(s)) for s in test_predictions]
            logger.info('Testing Result: \n{} \n{}\n{}\n'.format(result[0], result[1], result[2]))
            truth = [' '.join(self.helper.index2sentence(s)) for s in test_truth]
            logger.info('Ground Truth: \n{} \n{}\n{}\n'.format(truth[0], truth[1], truth[2]))

            #logger.info('Testing loss: {}'.format(loss))

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


