import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vocabulary import Vocabulary
from dataset import TrainingDataset, collate_fn
from model import VideoCaptionGenerator, EncoderRNN, DecoderRNN
from customloss import CustomLoss
from trainer import Trainer
from checkpoint import *

import logging
from MyLogger import logger


def main():
    training_json_file = 'data/training_label.json'
    training_avi_feats = 'data/training_data/feat'
    testing_json_file = 'data/testing_label.json'
    testing_avi_feats = 'data/testing_data/feat'

    helper = Vocabulary(training_json_file, min_word_count=3)

    dataset = TrainingDataset(
        label_json_file=training_json_file,
        training_data_path=training_avi_feats,
        helper=helper,
        load_into_ram=True
    )

    test_dataset = TrainingDataset(
        label_json_file=testing_json_file,
        training_data_path=testing_avi_feats,
        helper=helper,
        load_into_ram=True
    )


    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=collate_fn)

    INPUT_FEAT_DIM = 4096
    HIDDEN_SIZE = 1000
    WORD_DIM = 1000
    OUTPUT_DIM = helper.vocab_size

    EPOCH = 300
    MDL_OUTDIR = 'model'
    if not os.path.exists(MDL_OUTDIR):
        os.mkdir(MDL_OUTDIR)

    encoder = EncoderRNN(input_size=INPUT_FEAT_DIM, hidden_size=HIDDEN_SIZE)

    decoder = DecoderRNN(hidden_size=HIDDEN_SIZE,
                         output_size=OUTPUT_DIM, # used for transforming hidden state into output softmax
                         vocab_size=OUTPUT_DIM, # used for transforming input one-hot into word2vec
                         word_dim=WORD_DIM
                         )

    model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

    trainer = Trainer(model=model, train_dataloader=dataloader, test_dataloader=test_dataloader, helper=helper)

    s = time.time()

    for epoch in range(EPOCH):
        trainer.train(epoch+1, check_result=True)
        trainer.eval(check_result=True)

    e = time.time()

    torch.save(model, "{}/{}.h5".format(MDL_OUTDIR, 'test'))

    logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))



if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    main()










