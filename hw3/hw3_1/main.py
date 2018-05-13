import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loss import discriminatorLoss, generatorLoss
from dataset import TrainingDataset
from model import Generator, Discriminator
from trainer import Trainer

BATCH_SIZE = 128

dataset = TrainingDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
model_G = Generator()
model_D = Discriminator()

MDL_OUTDIR = 'saved'
if not os.path.exists(MDL_OUTDIR):
    os.mkdir(MDL_OUTDIR)

MDL_PRETRAINED_PATH_G = ''
MDL_PRETRAINED_PATH_D = ''
if MDL_PRETRAINED_PATH_G:
	print('Loading pretrained model_G weights from:', MDL_PRETRAINED_PATH_G)
	model_G.load_state_dict(torch.load(MDL_PRETRAINED_PATH_G))
else:
	print('Initiating new model_G...')
if MDL_PRETRAINED_PATH_D:
	print('Loading pretrained model_D weights from:', MDL_PRETRAINED_PATH_D)
	model_D.load_state_dict(torch.load(MDL_PRETRAINED_PATH_D))
else:
	print('Initiating new model_D...')

trainer = Trainer(model_G, model_D, dataloader)

s = time.time()
print('Start training...')

EPOCH = 2000
EPOCH_PER_SAVE = 15
for epoch in range(EPOCH):
    trainer.train(epoch, not (epoch % EPOCH_PER_SAVE))

e = time.time()

logger.info("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))