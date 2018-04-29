import torch
import torch.nn as nn
from torch.autograd import Variable
input_size = 3
hidden_size = 4

model = nn.GRU(3, 4, batch_first=False).cuda()
x = Variable(torch.randn(5, 4, input_size).cuda())
h = Variable(torch.randn(2, 3, hidden_size).cuda())
model(x, h)
