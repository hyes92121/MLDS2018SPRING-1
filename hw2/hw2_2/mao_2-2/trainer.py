import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from customloss import CustomLoss


class Trainer(object):
    def __init__(self, model, train_dataloader,  helper=None): # no test set for HW2-2

        self.train_loader = train_dataloader

        # Use cuda if available
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        self.parameters = model.parameters()
        self.loss_fn = CustomLoss()
        self.loss = None
        self.optimizer = optim.Adam(self.parameters, lr=3e-4) # TODO: change optimizer

        # used for printing model output
        self.helper = helper


    def train(self, epoch, check_result=False): # TODO: currently using epoch as steps
        self.model.train()

        test_input, test_truth = None, None

        for batch_idx, batch in enumerate(self.train_loader):
            # prepare data
            a = time.time()
            padded_prev_sentences, padded_curr_sentences, lengths_curr_sentences = batch
            if self.__CUDA__:
                padded_prev_sentences, padded_curr_sentences = padded_prev_sentences.cuda(), padded_curr_sentences.cuda()

            padded_prev_sentences, padded_curr_sentences = Variable(padded_prev_sentences), Variable(padded_curr_sentences)

            # start training process
            self.optimizer.zero_grad()
            seq_Prob, seq_predictions = self.model(prev_sentences=padded_prev_sentences, mode='train', curr_sentences=padded_curr_sentences, steps=epoch)
            padded_curr_sentences = padded_curr_sentences[:, 1:]  # eliminate <SOS>

            if check_result:
                if test_input is None or test_truth is None:
                    test_input = padded_prev_sentences[:3]
                    test_truth = padded_curr_sentences[:3]


            loss = self.loss_fn(seq_Prob, padded_curr_sentences, lengths_curr_sentences)
            loss.backward()
            self.optimizer.step()

            # print out training info
            if (batch_idx+1):
                info = self.get_training_info(
                    epoch=epoch,
                    batch_id=batch_idx,
                    batch_size=len(lengths_curr_sentences),
                    total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader),
                    loss=loss.data[0]
                )
                print('\r', info, '   ', int(time.time()-a), 'seconds/batch', end='') # original: end='\r'
        print()


        if check_result:
            _, test_predictions = self.model(prev_sentences=padded_prev_sentences, mode='train', curr_sentences=padded_curr_sentences, steps=epoch)
            result = [' '.join(self.helper.index2sentence(s)) for s in test_predictions]
            print('Training Result: \n{} \n{}\n{}\n'.format(result[0], result[1], result[2]))
            truth = [' '.join(self.helper.index2sentence(s)) for s in test_truth]
            print('Ground Truth: \n{} \n{}\n{}\n'.format(truth[0], truth[1], truth[2]))



    def eval(self, check_result=False):
        # set model to evaluation(testing) mode
        self.model.eval()

        test_predictions, test_truth = None, None

        for batch_idx, batch in enumerate(self.train_loader):
            # prepare data
            padded_prev_sentences, padded_curr_sentences, lengths_curr_sentences = batch
            if self.__CUDA__:
                padded_prev_sentences, padded_curr_sentences = padded_prev_sentences.cuda(), padded_curr_sentences.cuda()

            padded_prev_sentences, padded_curr_sentences = Variable(padded_prev_sentences), Variable(padded_curr_sentences)


            # start inferencing process
            seq_Prob, seq_predictions = self.model(padded_prev_sentences, mode='inference')
            padded_curr_sentences = padded_curr_sentences[:, 1:]  # eliminate <SOS>

            test_predictions = seq_predictions[:3]
            test_truth = padded_curr_sentences[:3]

            break

        if check_result:
            result = [' '.join(self.helper.index2sentence(s)) for s in test_predictions]
            print('Testing Result: \n{} \n{}\n{}\n'.format(result[0], result[1], result[2]))
            truth = [' '.join(self.helper.index2sentence(s)) for s in test_truth]
            print('Ground Truth: \n{} \n{}\n{}\n'.format(truth[0], truth[1], truth[2]))

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



