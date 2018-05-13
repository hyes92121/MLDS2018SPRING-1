import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable


def discriminatorLoss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """
    loss = nn.BCELoss()
    N = logits_real.size() # Batch size.
    
    if torch.cuda.is_available():
        true_labels = Variable(torch.ones(N).cuda()) # True label is 1
        fake_labels = Variable(torch.zeros(N).cuda()) # Fake label is 0
    else:
        true_labels = Variable(torch.ones(N)) # True label is 1
        fake_labels = Variable(torch.zeros(N)) # Fake label is 0
    real_image_loss = loss(logits_real, true_labels)
    fake_image_loss = loss(logits_fake, 1 - true_labels)

    totalLoss = real_image_loss + fake_image_loss 

    return totalLoss

def generatorLoss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """
    loss = nn.BCELoss()
    N = logits_fake.size() # Batch size.
    
    if torch.cuda.is_available():
        true_labels = Variable(torch.ones(N).cuda())
    else:
        true_labels = Variable(torch.ones(N))
    totalLoss = loss(logits_fake, true_labels)

    return totalLoss
