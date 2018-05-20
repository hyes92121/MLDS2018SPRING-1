import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from model import Generator
import matplotlib.pyplot as plt
import cv2
plt.switch_backend('agg')

model_G = Generator()
np.random.seed(50)

MDL_PRETRAINED_PATH_G = 'saved/epoch' + sys.argv[1] + '_G.pt'
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
    fig.savefig("samples/output" + sys.argv[1] + ".png")
    plt.close()

def detect(filename, cascade_file = "test/lbpcascade_animeface/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    print("Detect {} faces".format(len(faces)))
    if len(faces) >= 20:
        print("Pass !")
    else:
        print("Fail !")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite("test/baseline_result.png", image)

save_imgs(model_G)
detect("samples/output" + sys.argv[1] + ".png")
