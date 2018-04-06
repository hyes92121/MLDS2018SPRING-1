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


# plot first layer

first_layer_params = []

for iters in range(1, 9):
    root_dir = 'iter_'
    for epoch in range(3, 43, 3):
        pp = np.load('{}{}/parameters_{}.npy'.format(root_dir, iters, epoch))
        f  = pp[0].reshape(-1, pp[0].size).flatten()
        first_layer_params.append(f)


# plot whole model parameters

model_parameters = []

for iters in range(1, 9):
    root_dir = 'iter_'
    for epoch in range(3, 43, 3):
        pp = np.load('{}{}/parameters_{}.npy'.format(root_dir, iters, epoch))
        tmp = np.array([])
        for l in pp:
            l = l.reshape(-1, l.size).flatten()
            tmp = np.concatenate((tmp, l))
        model_parameters.append(tmp)

acc_his = []

for iters in range(1, 9):
    root_dir = 'iter_'
    acc = np.load('{}{}/accuracy.npy'.format(root_dir, iters))
    for i, a in enumerate(acc):
        if (i+1) % 3 == 0:
            acc_his.append(a)

first_layer_params = np.array(first_layer_params)
print(first_layer_params.shape)
model_parameters = np.array(model_parameters)
print(model_parameters.shape)
acc_his = np.array(acc_his)
print(acc_his.shape)

#mdl_params = np.load('{}{}/parameters_{}.npy'.format(root_dir, 1, 3))

# np array with shape (200704, ) (or 256*784, 1)
#layer_1 = mdl_params[0].reshape(-1, mdl_params[0].size)

pca = PCA(n_components=2)


s = time.time()
print('starting pca...')
pca = pca.fit_transform(first_layer_params)
#pca = pca.fit_transform(model_parameters)
e = time.time()

print('Finished PCA. Elapsed time: {} seconds'.format(e-s))

colors = plt.cm.cool(np.linspace(0,1,112))

for i, (acc, (X, Y)) in enumerate(zip(acc_his, pca)):
    plt.scatter(X.reshape(1, 1), Y.reshape(1, 1), marker=r'${:.2f}$'.format(acc*10), c=colors[i], s=700, linewidth=0.05)

plt.title("First Layer Parameters into PCA")
#plt.title("Whole Model Parameters into PCA")
plt.xlabel("w1")
plt.ylabel("w2")

plt.show()

