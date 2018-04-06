import os
import time
import _pickle as pickle

import torch
from trainer.trainer import Trainer
from dataset.dataset import TrainingDataset
from model.models import ShallowNet, DeepNet
from torch.utils.data import DataLoader

from utils.argparser import argument_parser

import logging
import MyLogger


def main():
    # Base Logger setup
    MyLogger.logger.setLevel(logging.INFO)
    logger = MyLogger.logger.getChild(__name__)

    args = argument_parser()
    # params from argparse
    csvpath     = args.train_data
    BATCH_SIZE  = args.batch_size
    EPOCHS      = args.epochs
    OUTPKL      = args.outpkl
    MDL_OUTDIR = args.mdl_outdir
    if not os.path.exists(MDL_OUTDIR):
        os.mkdir(MDL_OUTDIR)

    LOSS_HISTORY = {}
    MODELS = []
    # (width, depth)
    DEEP_MODEL = [(18, 1), (9, 4), (8, 5)]
    # (width)
    SHALLOW_MODEL = [128]

    training_dataset    = TrainingDataset(csvpath)
    train_loader        = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # append different models
    for w in SHALLOW_MODEL:
        MODELS.append(ShallowNet(w))
    for w, d in DEEP_MODEL:
        MODELS.append(DeepNet(depth=d, width=w))

    # train EPOCHS epochs for each model in MODELS
    for model in MODELS:
        model_name = model.get_name()
        logger.info("Start training {}".format(model_name))
        trainer = Trainer(model, train_loader)
        # add model loss history into dictionary
        LOSS_HISTORY[model_name] = []

        s = time.time()
        for epoch in range(EPOCHS):
            trainer.train(epoch+1)
            LOSS_HISTORY[model_name].append(trainer.loss)
        e = time.time()

        torch.save(model, "{}/{}.h5".format(MDL_OUTDIR, model_name))

        logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format(model_name, e-s))


    logger.info("Finished training all models. Dumping loss history")

    with open(OUTPKL, 'wb') as f:
        pickle.dump(LOSS_HISTORY, f)

    logger.info("Finished dumping. Existing program...")



if __name__ == "__main__":
    main()

