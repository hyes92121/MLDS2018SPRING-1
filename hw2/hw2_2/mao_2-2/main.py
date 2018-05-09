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

BATCH_SIZE = 80
dataset = TrainingDataset(training_data_path, helper)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn)

encoder = EncoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=256, num_layers=2)
decoder = DecoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=256, num_layers=2)
model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

MDL_OUTDIR = 'two_layer'
if not os.path.exists(MDL_OUTDIR):
    os.mkdir(MDL_OUTDIR)

MDL_PRETRAINED_PATH = ''
if MDL_PRETRAINED_PATH:
	print('Loading pretrained model weights from:', MDL_PRETRAINED_PATH)
	model.load_state_dict(torch.load(MDL_PRETRAINED_PATH))
else:
	print('Initiating new model...')

SKIP_TO_DATA_IDX = 0 # copy this directly from command line
skip_to_batch_idx = SKIP_TO_DATA_IDX // BATCH_SIZE
if skip_to_batch_idx != 0:
	print('Skipping to batch', skip_to_batch_idx)

trainer = Trainer(model=model, train_dataloader=dataloader, helper=helper)

s = time.time()
print('Start training...')

EPOCH = 300
BATCHES_PER_SAVE = 60 # 60 batches/min on GTX1080 (for batch_size = 128) 
for epoch in range(EPOCH):
    trainer.train(epoch+1, batch_size=BATCH_SIZE, check_result=True, model_dir=MDL_OUTDIR, batches_per_save=BATCHES_PER_SAVE,
    				skip_to_batch_idx=skip_to_batch_idx)
    trainer.eval(check_result=True)

e = time.time()

logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))