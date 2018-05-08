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
MDL_OUTDIR = '1505_models/'
MDL_PRETRAINED_PATH = '1505_models/18.pt'
BATCHES_PER_SAVE = 60 # 60 batches/min on GTX1080
SKIP_TO_BATCH_IDX = 19 * BATCHES_PER_SAVE # this will also offset the model filenames (which are numbers)
if not os.path.exists(MDL_OUTDIR):
    os.mkdir(MDL_OUTDIR)

encoder = EncoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=512, num_layers=1)
decoder = DecoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=512, num_layers=1)
model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

if MDL_PRETRAINED_PATH:
	print('Loading pretrained model weights from:', MDL_PRETRAINED_PATH)
	model.load_state_dict(torch.load(MDL_PRETRAINED_PATH))
else:
	print('Initiating new model...')

if SKIP_TO_BATCH_IDX != 0:
	print('Skipping to batch', SKIP_TO_BATCH_IDX)

trainer = Trainer(model=model, train_dataloader=dataloader, helper=helper)

s = time.time()
print('Start training...')
for epoch in range(EPOCH):
    trainer.train(epoch+1, check_result=True, model_dir=MDL_OUTDIR, batches_per_save=BATCHES_PER_SAVE,
    				skip_to_batch_idx=SKIP_TO_BATCH_IDX)
    trainer.eval(check_result=True)

e = time.time()

logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))