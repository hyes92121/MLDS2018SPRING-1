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
from model import MnistDNN
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

    LOSS_HISTORY = {}
    ACC_HISTORY = []
    MODELS = []

    model = MnistDNN()


    train_dataset = datasets.MNIST('data', train=True, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),]))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    model_name = model.get_name()
    logger.info("Start Training "+model_name)

    trainer = Trainer(model=model, train_dataloader=train_loader, test_dataloader=test_loader)
    LOSS_HISTORY[model_name] = []

    s = time.time()
    for epoch in range(1, EPOCHS+1):
        trainer.train(epoch)
        trainer.eval(epoch)
        ACC_HISTORY.append(trainer.test_loss)

        if epoch % 3 == 0:
            params = []
            for p in model.parameters():
                params.append(p.data.numpy())
            params = np.array(params)
            np.save('{}/parameters_{}.npy'.format(MDL_OUTDIR, epoch), params)

    e = time.time()

    logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format(model_name, e - s))

    ACC_HISTORY = np.array(ACC_HISTORY)

    np.save('{}/accuracy.npy'.format(MDL_OUTDIR), ACC_HISTORY)

    logger.info("Finished training all models. Dumping loss history")


if __name__ == "__main__":
    main()

