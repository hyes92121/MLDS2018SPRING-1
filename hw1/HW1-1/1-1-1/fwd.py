from model.models import DeepNet, ShallowNet
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# (width, depth)
DEEP_MODEL = [(18, 1), (9, 4), (8, 5)]
# (width)
SHALLOW_MODEL = [128]

MODELS = []

# append different models
for w in SHALLOW_MODEL:
    MODELS.append(ShallowNet(w))
for w, d in DEEP_MODEL:
    MODELS.append(DeepNet(depth=d, width=w))


model = MODELS[0]
print(model.parms_n())
model.summary()
x = Variable(torch.Tensor([1]))
y = model(x)

writer = SummaryWriter()
writer.add_graph(model, y)
writer.close()

exit(-1)


for model in MODELS:
	print(model.parms_n())
	model.summary()
	x = Variable(torch.Tensor([1]))
	y = model(x)

	SummaryWriter().add_graph(model, y)
