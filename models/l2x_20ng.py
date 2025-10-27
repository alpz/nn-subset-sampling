
import torch
#from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import sampling.khot as khs

import sampling.sorting_operator as ss

class L2X20NG(nn.Module):
    def __init__(self, explainer, q_net, subset_size=50, diffk=True,  cuda=True):
        super(L2X20NG, self).__init__()
        self.DEVICE = torch.device("cuda" if cuda else "cpu")

        self.explainer = explainer
        self.q_net = q_net
        self.diffk = diffk
        self.size = 1000

        self.subset_size = subset_size
        self.diffk = diffk

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x,y, target, evaluation=False):
        #explainer_out = self.explainer(x)
        x_mask = (x != 0).float() * (x != 1).float()

        #5%
        #k = x_mask.sum(dim=-1,keepdim=False)*0.05

        explainer_out = self.explainer(x, x_mask, evaluation=evaluation)
        mask = explainer_out['subset_mask']
        #mask_hard = explainer_out['subset_mask_hard']
        k_out = explainer_out['k_out']
        subset_logits = explainer_out['subset_logits']
        #mask_mean = mask.reshape((-1,32*32)).sum(dim=-1).mean()
            
        logits = self.q_net(x, mask)

        pred = logits.data.max(1)[1] # get the index of the max logit

        pred_accuracy = pred.eq(y.data).float().sum()
        target_accuracy = pred.eq(target.data).float().sum()

        mean_k = k_out.mean()
        min_k = k_out.min()
        max_k = k_out.max()

        if self.diffk:
            # loss = self.criterion(logits, y) + 0.015*(mean_k)
            #loss = self.criterion(logits, y) + 0.0001*(mean_k*1000-50)**2
            loss = self.criterion(logits, y) + 0.0001*(mean_k*1000-self.subset_size)**2
            k_ret = (k_out.squeeze() * self.size).int()
        else:
            loss = self.criterion(logits, y)
            k_ret = k_out

        return {
            'loss': loss,
            'logits': logits,
            'subset_logits': subset_logits,
            'correct': pred_accuracy,
            'target_correct': target_accuracy,
            'predictions': pred,
            #'mask': mask_hard,
            #'k_out': (k_out.squeeze()*1000).int(),
            'k_out': k_ret,
            'mean_k': mean_k*self.size,
            'min_k': min_k*self.size,
            'max_k': max_k*self.size,
        }

class Explainer(nn.Module):
    def __init__(self, embedding_matrix, rr=False, subset_size=50, diffk=True, cuda=True, correct=False):
        """ stacked hierarchical MLP Encoder"""
        super(Explainer, self).__init__()

        #self.subset_size = 25 #32*32*(0.001)
        self.subset_size = subset_size #32*32*(0.001)
        self.diffk = diffk
        self.correct = correct

        self.use_subop = rr
        if(rr):
            print('using subset baseline')
            self.subop = ss.SubsetOperator(k=self.subset_size, tau=1, hard=True)

        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.do1 = nn.Dropout(p=0.2)
        self.pre_1 = nn.Conv1d(100, 128, 5, padding='same')
        self.g_linear = nn.Linear(128, 128)
        #self.pool1 = nn.MaxPool1d(5)
        self.fc_k = nn.Linear(128, 1)

        self.conv1 = nn.Conv1d(128, 128, 5, padding='same')

        self.conv2 = nn.Conv1d(128, 128, 5, padding='same')
        #self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(128, 128, 5, padding='same')
        self.conv4 = nn.Conv1d(2*128, 128, 5, padding='same')
        self.do2 = nn.Dropout(p=0.2)
        self.conv5 = nn.Conv1d(128, 64, 5, padding='same')
        self.conv6 = nn.Conv1d(64, 1, 5, padding='same')


    def forward(self, x, x_mask,  evaluation=False, debug=False):
        #(1000,100)
        x = self.embed(x)
        x = self.do1(x)
        #(100,1000)
        x = x.transpose(1,2)
        pre_x = F.relu(self.pre_1(x))

        global_x = F.relu(self.conv1(pre_x))
        #global max pool
        global_x = F.max_pool1d(global_x, kernel_size=global_x.shape[-1])
        global_x = F.relu(self.g_linear(global_x.squeeze()))

        k_out = torch.sigmoid(self.fc_k(global_x)).squeeze()

        #B, 128, 1000 repeat
        global_x = global_x.unsqueeze(dim=2).expand(-1,-1, pre_x.shape[2])

        x = F.relu(self.conv2(pre_x))
        x = F.relu(self.conv3(x))
        #concat along 10
        x = torch.cat([x, global_x], dim=1)

        x = F.relu(self.conv4(x))
        x = self.do2(x)
        x = F.relu(self.conv5(x))
        #B, 1000
        subset_logits = self.conv6(x).squeeze()

        #-------------- subset sample ---------------
        if self.use_subop and not evaluation:
            k_hot_sample = self.subop(subset_logits)
            k_out = torch.ones_like(k_out)*self.subset_size
        elif self.diffk:
            #k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, x_mask, st_prob=True)
            #k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, st_prob=True)
            k_hot_sample, norm_prob = khs.sample_approx_k_hot(k_out.squeeze()*1000, subset_logits, st_prob=True, correct=self.correct)
        else:
            k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, st_prob=True, correct=self.correct)
            k_out = torch.ones_like(k_out)*self.subset_size

        if self.use_subop and evaluation:
            # evaluation for subop baseline - hard k-hot
            khot_hard = torch.zeros_like(subset_logits)
            val, ind = torch.topk(subset_logits, self.subset_size, dim=1)
            #print(ind[0])
            k_hot_sample_hard = khot_hard.scatter_(1, ind, 1)
            k_hot_sample = k_hot_sample_hard
            #k_hot_sample_hard = k_hot_sample_hard.reshape((-1,3,32,32))

        return {
            'subset_mask': k_hot_sample,
            'subset_logits': subset_logits,
            #'subset_mask_hard': k_hot_sample_hard,
            'k_out': k_out,
        }

class QNet(nn.Module):
    def __init__(self, embedding_matrix, cuda=True):
        super(QNet, self).__init__()

        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.conv1 = nn.Conv1d(100, 100, 5, padding='same')
        self.conv2 = nn.Conv1d(100, 100, 5, padding='same')
        self.fc1 = nn.Linear(100, 100)
        self.do1 = nn.Dropout(p=0.2)
        #20 classes
        self.fc2 = nn.Linear(100, 20)

    def forward(self, x, mask):
        #(1000,100)
        x = self.embed(x)
        #(100,1000)
        x = x.transpose(1,2)

        #mask: (B,1000)
        x = x*mask.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #average along sequence
        x = x.mean(dim=2)
        x = self.do1(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits
