import os
import time
import _pickle as pickle
import numpy as np

import torch
#from trainer.trainer import Trainer
#from dataset.dataset import TrainingDataset
#from model.models import ShallowNet, DeepNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.argparser import argument_parser

import logging
import MyLogger


############################
#     for testing only     #
from trainer import Trainer
from model import MnistDNN, CifarCNN
############################


def main():
    # Base Logger setup
    MyLogger.logger.setLevel(logging.INFO)
    logger = MyLogger.logger.getChild(__name__)

    args = argument_parser()
    # params from argparse
    BATCH_SIZE  = args.batch_size
    EPOCHS      = args.epochs
    OUTPKL      = args.outpkl
    MDL_OUTDIR  = args.mdl_outdir
    if not os.path.exists(MDL_OUTDIR):
        os.mkdir(MDL_OUTDIR)

    LOSS_HISTORY = []
    ACC_HISTORY = []
    PARAMS = []
    MODELS = [i for i in range(64, 257, 8)]

    train_dataset = datasets.CIFAR10('data', train=True, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),]))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    for m in MODELS:
        model = CifarCNN(depth=m*2)
        model_name = model.get_name()
        logger.info("Start Training "+model_name)
        logger.info("Model parameter numbers: {}".format(model.parms_n()))

        trainer = Trainer(model=model, train_dataloader=train_loader, test_dataloader=test_loader)

        s = time.time()
        for epoch in range(1, EPOCHS+1):
            trainer.train(epoch)
        trainer.eval(epoch)
        trainer.get_train_acc(epoch)
        LOSS_HISTORY.append((trainer.loss, trainer.test_loss))
        ACC_HISTORY.append((trainer.train_acc, trainer.test_acc))
        PARAMS.append(model.parms_n())
        e = time.time()

        logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format(model_name, e - s))

        break

    np.save('{}/{}'.format(MDL_OUTDIR, 'acc.npy'), ACC_HISTORY)
    np.save('{}/{}'.format(MDL_OUTDIR, 'loss.npy'), LOSS_HISTORY)
    np.save('{}/{}'.format(MDL_OUTDIR, 'params.npy'), PARAMS)

    logger.info("Finished training all models. Dumping loss history")


if __name__ == "__main__":
    #main()

    from tensorboardX import SummaryWriter

    SummaryWriter().add_graph_onnx()

