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

training_data_path='data/clr_conversation.txt'
helper = Vocabulary(training_data_path)

dataset = TrainingDataset(training_data_path, helper)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=collate_fn)

EPOCH = 300
MDL_OUTDIR = 'model'
if not os.path.exists(MDL_OUTDIR):
    os.mkdir(MDL_OUTDIR)

encoder = EncoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=1024, num_layers=1)
decoder = DecoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=1024, num_layers=1)
model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

trainer = Trainer(model=model, train_dataloader=dataloader, helper=helper)

s = time.time()
print('Start training...')
for epoch in range(EPOCH):
    trainer.train(epoch+1, check_result=True)
    trainer.eval(check_result=True)

e = time.time()

torch.save(model, "{}/{}.h5".format(MDL_OUTDIR, 'test'))

logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))