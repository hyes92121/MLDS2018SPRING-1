import numpy as np
import torch
from torch.autograd import Variable
from loss import discriminatorLoss, generatorLoss
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import time


class Trainer(object):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, data_loader):
        super(Trainer, self).__init__()
        self.with_cuda = torch.cuda.is_available()
        print("modelG has", self.count_parameters(model_G), "parameters...")
        print("modelD has", self.count_parameters(model_D), "parameters...")

        if self.with_cuda:
            self.model_G = model_G.cuda()
            self.model_D = model_D.cuda()
        else:
            self.model_G = model_G.cpu()
            self.model_D = model_D.cuda()
        
        self.optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        #self.optimizer_D = torch.optim.SGD(model_D.parameters(), lr=0.005)

        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        
        self.D_iteration = 1
        self.G_iteration = 1
        
        self.log_step = 1

    def train(self, epoch, saveModel=False):
        """
        Training logic for an epoch
        """
        self.model_G.train()
        self.model_D.train()
        a = time.time()
        if epoch > 10:
            self.G_iteration = 2
        
        for batch_idx, real in enumerate(self.data_loader):
            if len(real) != self.batch_size:
                break
            
            for _ in range(self.D_iteration):
                self.optimizer_D.zero_grad()
                if self.with_cuda:
                    real = real.cuda()
                if _ == 0:
                    real = Variable(real)
                real_logits = self.model_D(real)
                
                noise = Variable(self.random_generator((self.batch_size, 100), True))
                fake = self.model_G(noise).detach() # prevent G from updating
                fake_logits = self.model_D(fake)
                
                d_loss = discriminatorLoss(real_logits, fake_logits)
                d_loss.backward()
                #clip_grad_norm(self.model_D.parameters(), 15)
                self.optimizer_D.step()
                
            for _ in range(self.G_iteration):
                self.optimizer_G.zero_grad()
                
                noise = Variable(self.random_generator((self.batch_size, 100), True))
                fake = self.model_G(noise)
                fake_logits = self.model_D(fake)
                
                g_loss = generatorLoss(fake_logits)
                g_loss.backward()
                #clip_grad_norm(self.model_G.parameters(), 30)
                self.optimizer_G.step()
            
            
            if batch_idx % self.log_step == 0:
                info = self.get_training_info(
                    epoch=epoch,
                    batch_id=batch_idx,
                    batch_size=self.batch_size,
                    total_data_size=len(self.data_loader.dataset),
                    n_batch=len(self.data_loader),
                    d_loss=d_loss.data[0],
                    g_loss=g_loss.data[0],
                    g_normD=self.get_gradient_norm(self.model_D),
                    g_normG=self.get_gradient_norm(self.model_G)
                )
                print('\r', info, end='') # original: end='\r'
        if (saveModel):
            print()
            model_dir = "saved"
            print('Saving model', "{}/epoch{}.pt".format(model_dir, epoch))
            torch.save(self.model_G.state_dict(), "{}/epoch{}_G.pt".format(model_dir, epoch))
            #torch.save(self.model_D.state_dict(), "{}/epoch{}_D.pt".format(model_dir, epoch))
        print()
        print("Training time: ", int(time.time()-a), 'seconds/epoch')
    
    def get_training_info(self,**kwargs):
        ep = kwargs.pop("epoch", None)
        bID = kwargs.pop("batch_id", None)
        bs = kwargs.pop("batch_size", None)
        tds = kwargs.pop("total_data_size", None)
        nb = kwargs.pop("n_batch", None)
        d_loss = kwargs.pop("d_loss", None)
        g_loss = kwargs.pop("g_loss", None)
        g_normD = kwargs.pop("g_normD", None)
        g_normG = kwargs.pop("g_normG", None)
        info = "Training Epoch: {} [{}/{} ({:.0f}%)]\tLossD: {:.6f} LossG: {:.6f} GradNormD: {:.0f} GradNormG: {:.0f}    ".format(ep, (bID+1)*bs, tds, 100.*bID/nb, d_loss, g_loss, g_normD, g_normG)
        return info
    
    def get_gradient_norm(self, net):
        grad_all = 0.0
        for p in net.parameters():
            gr = 0.0
            if p.grad is not None:
                gr = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all += gr
        return grad_all ** 0.5
    
    def random_generator(self, shape, isNormal=False):
        if isNormal:
            ranNbr = torch.randn(shape)
        else:
            ranNbr = torch.zeros(shape).uniform_(0, 1)
            
        if self.with_cuda:
            return ranNbr.cuda()
        else:
            return ranNbr
                                 
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    from dataset import TrainingDataset
    from torch.utils.data import DataLoader
    import numpy as np
    from model import Generator, Discriminator
    
    dataset = TrainingDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    
    model_G = Generator()
    model_D = Discriminator()
    
    trainer = Trainer(model_G, model_D, dataloader)
    
    for epoch in range(1):
        trainer.train(epoch, True)

    