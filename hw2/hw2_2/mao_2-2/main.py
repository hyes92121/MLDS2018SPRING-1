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
MDL_PRETRAINED_PATH = 'model_state_dict/18.pt'
if not os.path.exists(MDL_OUTDIR):
    os.mkdir(MDL_OUTDIR)

encoder = EncoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=512, num_layers=1)
decoder = DecoderRNN(word_vec_filepath='word_vectors.npy', hidden_size=512, num_layers=1)
model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

if MDL_PRETRAINED_PATH:
	print('Loading pretrained model weights...')
	model.load_state_dict(torch.load(MDL_PRETRAINED_PATH))
else:
	print('Initiating new model...')

trainer = Trainer(model=model, train_dataloader=dataloader, helper=helper)

s = time.time()
print('Start training...')
for epoch in range(EPOCH):
    trainer.train(epoch+1, check_result=True, model_dir=MDL_OUTDIR, batches_per_save=60) # 60 batches/min
    trainer.eval(check_result=True)

e = time.time()

logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))