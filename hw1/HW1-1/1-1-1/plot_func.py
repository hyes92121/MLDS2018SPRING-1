import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import _pickle as pickle
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
import os

MODEL_DIR = "net1"
target_data = pd.read_csv("data/data_train1.csv")
num_plots = len(os.listdir(MODEL_DIR))
colors = plt.cm.cool(np.linspace(0,1,num_plots))

x = target_data.transpose().iloc[0].values
y = target_data.transpose().iloc[1].values

x_v = x.reshape(-1, 1)

fig, ax = plt.subplots(1, 1)
ax.plot(x, y, color='red')

model_name_list = []
for i, mm in enumerate(os.listdir(MODEL_DIR)):
    model = torch.load("{}/{}".format(MODEL_DIR, mm))
    model = model.cuda()
    model_name_list.append(model.get_name())
    predict = Variable(torch.Tensor(x_v).cuda())
    ax.plot(x, model(predict), color=colors[i])


#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_res))
#ax.set_yscale('log')
plt.xlabel("Training epochs")
plt.ylabel("Training loss")
#plt.ylim(ymin=1e-6)
leg = plt.legend(model_name_list, loc='upper right')


for line in leg.get_lines():
    line.set_linewidth(2)

plt.save("recon")

