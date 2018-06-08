import os
import cv2
import sys
import numpy as np
import _pickle as pickle
from PIL import Image

import torch
import torchvision
from torch.autograd import Variable
from model import CGAN

torch.manual_seed(222222)


def save_imgs(img):
    import matplotlib.pyplot as plt
	plt.use('Agg')
    if not os.path.exists('samples'):
    	os.mkdir('samples')

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    # gen_imgs should be shape (25, 64, 64, 3)
    #gen_imgs = generator.predict(noise)
    gen_imgs = img

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("samples/cgan.png")
    plt.close()


def cvt_output(model_output):
	img = model_output.data.numpy()[0]
	img = np.transpose(img, (1, 2, 0))
	img = 0.5*img + 0.5
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#img = np.transpose(img, (2, 0, 1))

	return img


state  = torch.load('model.tar', map_location=lambda storage, loc: storage)
config = state['config']

model  = CGAN(config)
model.load_state_dict(state['state_dict'])
model.eval()


with open('vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)


conditions = []

with open(sys.argv[1]) as f:
	for line in f:
		line = line.split(',')[1]
		line = line.split()
		conditions.append({'hair': [line[0]], 'eyes': [line[2]]})

generated_imgs = []

fixed_noise = [torch.randn(1, 100, 1, 1) for _ in range(5)]


for i, c in enumerate(conditions):
	try:
		noise = fixed_noise[i % 5]
	except Exception:
		raise IndexError
		print(i)
		print(25 % (i+1))
		print(len(fixed_noise))
		exit(-1)

	c = vocab.encode_feature(c)
	c = np.expand_dims(c, 0) * 0.9 + 0.05
	noise, c = Variable(torch.Tensor(noise)), Variable(torch.Tensor(c))

	img_v = model.generator(noise, c)
	img = cvt_output(img_v)
	generated_imgs.append(img)


imgs = np.array(generated_imgs)
save_imgs(imgs)
"""
imgs = Variable(torch.Tensor(imgs))

torchvision.utils.save_image(imgs.data, 'test.jpg', nrow=5)
"""




	 