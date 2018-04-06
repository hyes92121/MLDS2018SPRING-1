import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import _pickle as pickle
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
import os
from sklearn.decomposition import PCA
import time


acc = np.load('acc.npy')
loss = np.load('loss.npy')
params = np.load('params.npy')

colors = plt.cm.cool(np.linspace(0,1,25))

for i, (loss, pa) in enumerate(zip(acc, params)):
    tr, te = loss
    plt.scatter(pa.reshape(1, 1), tr.reshape(1, 1), marker='o', c='red', s=70,
                linewidth=0.05)
    plt.scatter(pa.reshape(1, 1), te.reshape(1, 1), marker='^', c='blue', s=70,
                linewidth=0.05)

plt.xlabel('Number of paramters')
#plt.ylabel('Loss')
plt.ylabel('Accuracy')
#plt.title('Training/Testing loss comparison')
plt.title('Training/Testing accuracy comparison')
leg = plt.legend(['training', 'testing'], loc='upper right')

plt.show()





