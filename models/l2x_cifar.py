
import torch
#from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.weibull import Weibull
import sampling.khot as khs
import sampling.sorting_operator as ss


def get_accuracy(logits, y):
    pred = (logits > 0).int()
    acc = (pred.int()==y.int()).float().mean()
    return acc

class L2XCifar(nn.Module):
    def __init__(self, explainer, q_net, subop=False,  cuda=True, diffk=False):
        super(L2XCifar, self).__init__()
        self.DEVICE = torch.device("cuda" if cuda else "cpu")

        self.diffk = diffk
        self.subop = subop
        self.explainer = explainer
        self.q_net = q_net

        self.size = 32*32*3
        #self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x,y, evaluate=False):
        x = x.reshape((-1,3,32,32))
        explainer_out = self.explainer(x, evaluate=evaluate)
        mask = explainer_out['subset_mask'].reshape((-1,3,32,32))
        #mask_mean = mask.reshape((-1,32*32)).sum(dim=-1).mean()
        #hard_mask = explainer_out['subset_mask_hard'].reshape((-1,3,32,32))

        k_out = explainer_out['k_out']
        subset_logits = explainer_out['subset_logits']

        masked_x = mask*(x+1)/2
        #neg_masked_x = (1-mask)*(x+1)/2

        logits = self.q_net(masked_x)

        #neg_masked_hard_x = (1-hard_mask)*(x+1)/2

        mean_k = k_out.mean()
        min_k = k_out.min()
        max_k = k_out.max()

        if self.subop:
            loss = self.criterion(logits, y)
        else:
            if self.diffk: 
              loss = self.criterion(logits, y)+ 0.01*(mean_k*self.size-100)**2
            else:
              loss = self.criterion(logits, y)#+ 0.01*(mean_k*self.size-100)**2
        #k_ones = torch.ones_like(k_out)
        #k_loss = -Weibull(scale=k_ones, concentration=k_ones*0.5).log_prob(k_out.clamp(min=1e-2)).mean()
        #loss = self.criterion(logits, y)+ (mean_k)**2
        #loss = self.criterion(logits, y)+ k_loss

        pred = logits.data.max(1)[1] # get the index of the max logit
        #accuracy = get_accuracy(logits, y)
        pred_accuracy = pred.eq(y.data).float().mean()

        if self.subop:
            k_ret = k_out
        else:
            if self.diffk:
              k_ret = (k_out.squeeze() * self.size).int()
            else:
              k_ret = k_out

        return {
            'loss': loss,
            'logits': logits,
            'subset_logits': subset_logits,
            'accuracy': pred_accuracy,
            'mask': mask,
            'pred': pred,
            #'masked_input': masked_x,
            #'neg_masked_input': neg_masked_hard_x,
            'k_out': k_ret,
            'mean_k': mean_k * self.size,
            'min_k': min_k * self.size,
            'max_k': max_k * self.size,
            }

class Explainer(nn.Module):
    def __init__(self, subop=False, cuda=True, diffk=False):
        """ stacked hierarchical MLP Encoder"""
        super(Explainer, self).__init__()

        self.diffk = diffk
        if not diffk:
          self.subset_size = 100 #32*32*(0.001)

        self.subop = subop
        if subop:
            self.subop = ss.SubsetOperator(k=self.subset_size, tau=1, hard=True)

        self.pre_explainer = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, stride=1, padding='same'),
                nn.ELU(),
                nn.BatchNorm2d(100),
                nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
                nn.Dropout(0.2),
                nn.ELU(),
                nn.BatchNorm2d(100),
                nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(100),
                nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                )

        self.explainer_out=nn.Sequential(
                nn.Dropout(0.2),
                nn.BatchNorm2d(100),
                nn.Conv2d(in_channels=100, out_channels=3, kernel_size=3, stride=1, padding=1),
                #nn.ELU(),
                #nn.BatchNorm2d(128),
                nn.Flatten(),
                #nn.Linear(128*4*4, 32*32*1)
            )

        self.k_out=nn.Sequential(
            nn.AvgPool2d(8),
            #nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(100*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, input, evaluate=False, debug=False):

        x = input.reshape((-1,3,32,32))
        pre_x = self.pre_explainer(x)
        subset_logits = self.explainer_out(pre_x)
        #print(subset_logits.shape)

        k_out = self.k_out(pre_x).squeeze()
        #subset_logits = self.subset_layer(x)
        #k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, st_prob=True)
        if not evaluate:
          if self.subop:
              k_hot_sample = self.subop(subset_logits)
              k_out = torch.ones_like(k_out)*self.subset_size
          else:
              if self.diffk:
                k_hot_sample, norm_prob = khs.sample_approx_k_hot(k_out*32*32*3, subset_logits, st_prob=True)
              else:
                k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, st_prob=True)
                k_out = torch.ones_like(k_out)*self.subset_size

        else:
          #if not self.diffk:
          if self.subop:
            #works for fixed subset size only
            khot_hard = torch.zeros_like(subset_logits)
            val, ind = torch.topk(subset_logits, self.subset_size, dim=1)
            k_hot_sample_hard = khot_hard.scatter_(1, ind, 1)
            k_hot_sample_hard = k_hot_sample_hard.reshape((-1,3,32,32))
            k_hot_sample = k_hot_sample_hard
          elif self.diffk:
            k_hot_sample, norm_prob = khs.sample_approx_k_hot(k_out*32*32*3, subset_logits, st_prob=True)
          else:
            k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, st_prob=True)

        k_hot_sample = k_hot_sample.reshape((-1,3,32,32))

        return {
            'subset_mask': k_hot_sample,
            #'subset_mask_hard': k_hot_sample_hard,
            'subset_logits': subset_logits,
            'k_out': k_out
        }

class QNet(nn.Module):
    def __init__(self, cuda=True):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.Dropout(0.2),
            #nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            #nn.AvgPool2d(4),
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            #nn.AvgPool2d(4),
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0),
            #nn.ELU(),
            nn.AvgPool2d(2),
            #nn.Flatten(),
            #nn.Dropout(0.2),
            #nn.Linear(64*4*4, 1),
            #nn.Linear(64*2*2, 10),
            #nn.Linear(10*4*4, 10),
            #nn.Linear(5*2*2, 10),
            #nn.Linear(5*16*16, 10),
            #nn.Linear(10*2*2, 10),
        )

    def forward(self, x):
        x = x.reshape((-1,3,32,32))
        logits = self.model(x)

        return logits.squeeze()
