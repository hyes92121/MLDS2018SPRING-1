import os
import sys
sys.path.append('./hw3_1')
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from model import Generator
import matplotlib.pyplot as plt
plt.switch_backend('agg')

model_G = Generator()
np.random.seed(50)

MDL_PRETRAINED_PATH_G = 'hw3_1.pt'
if not os.path.exists(MDL_PRETRAINED_PATH_G):
    raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(MDL_PRETRAINED_PATH_G, __name__))

model_G.load_state_dict(torch.load(MDL_PRETRAINED_PATH_G))

def save_imgs(generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    # gen_imgs should be shape (25, 64, 64, 3)
    noise = Variable(torch.FloatTensor(noise))
    gen_imgs = generator(noise).data.numpy()
    gen_imgs = np.rollaxis(gen_imgs, 1, 4)
    gen_imgs = (gen_imgs*127.5)+127.5
    gen_imgs = np.clip(np.rint(gen_imgs), 0, 255)
    gen_imgs = gen_imgs.astype(np.uint8)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("samples/gan.png")
    plt.close()

save_imgs(model_G)
