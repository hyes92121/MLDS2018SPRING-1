import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import _pickle as pickle
import numpy as np

with open("his2.pkl", "rb") as f:
    history = pickle.load(f)

num_plots = len(history)

tick_res = 1000
fig, ax = plt.subplots(1, 1)

colors = plt.cm.cool(np.linspace(0,1,num_plots))

model_name_list = []
for i, (k, v) in enumerate(history.items()):
    model_name_list.append(k)
    #v = v[500:1000]
    epochs, loss = np.array(range(1, len(v)+1)), np.array(v)
    ax.plot(epochs, loss, color=colors[i], linewidth=0.1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_res))
ax.set_yscale('log')
plt.xlabel("Training epochs")
plt.ylabel("Training loss")
#plt.ylim(ymin=1e-6)
leg = plt.legend(model_name_list, loc='upper right')


for line in leg.get_lines():
    line.set_linewidth(2)

#plt.show()
plt.savefig("loss2.png")
