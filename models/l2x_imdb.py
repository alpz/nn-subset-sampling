
import torch
#from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import sampling.khot as khs


class L2XIMDB(nn.Module):
    def __init__(self, explainer, q_net,  cuda=True):
        super(L2XIMDB, self).__init__()
        self.DEVICE = torch.device("cuda" if cuda else "cpu")

        self.explainer = explainer
        self.q_net = q_net

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x,y, target):
        x_mask = (x!=0).float()
        explainer_out = self.explainer(x, x_mask)
        mask = explainer_out['subset_mask']
        #mask_mean = mask.reshape((-1,32*32)).sum(dim=-1).mean()
            
        logits = self.q_net(x, mask)

        pred = logits.data.max(1)[1] # get the index of the max logit

        pred_accuracy = pred.eq(y.data).float().cpu().sum()
        target_accuracy = pred.eq(target.data).float().cpu().sum()

        loss = self.criterion(logits, y)

        return {
            'loss': loss,
            'logits': logits,
            'correct': pred_accuracy,
            'target_correct': target_accuracy,
            'predictions': pred,
            'mask': mask,
        }

class Explainer(nn.Module):
    def __init__(self, embedding_matrix, cuda=True):
        """ stacked hierarchical MLP Encoder"""
        super(Explainer, self).__init__()

        self.subset_size = 10 #32*32*(0.001)
        max_features = 5000
        #maxlen = 400
        #batch_size = 40
        embedding_dims = 50

        self.embed = nn.Embedding(max_features, embedding_dims)
        self.do1 = nn.Dropout(p=0.1)
        self.pre_1 = nn.Conv1d(50, 100, 3, padding='same')
        self.g_linear = nn.Linear(100, 100)
        #self.pool1 = nn.MaxPool1d(5)

        #self.conv1 = nn.Conv1d(128, 128, 5, padding='same')

        self.conv2 = nn.Conv1d(100, 100, 3, padding='same')
        #self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(100, 100, 3, padding='same')
        self.do2 = nn.Dropout(p=0.1)
        self.conv4 = nn.Conv1d(2*100, 100, 1, padding='same')
        self.conv5 = nn.Conv1d(100, 1, 1, padding='same')
        #self.conv6 = nn.Conv1d(64, 1, 5, padding='same')


    def forward(self, x, x_mask, debug=False):
        #(1000,100)
        x = self.embed(x)
        x = self.do1(x)
        #(100,1000)
        x = x.transpose(1,2)
        pre_x = F.relu(self.pre_1(x))

        global_x = pre_x
        #global max pool
        global_x = F.max_pool1d(global_x, kernel_size=global_x.shape[-1])
        global_x = F.relu(self.g_linear(global_x.squeeze()))

        #B, 128, 1000 repeat
        global_x = global_x.unsqueeze(dim=2).expand(-1,-1, pre_x.shape[2])

        #local
        x = F.relu(self.conv2(pre_x))
        x = F.relu(self.conv3(x))
        #concat along channel
        x = torch.cat([x, global_x], dim=1)
        x = self.do2(x)
        x = F.relu(self.conv4(x))
        subset_logits = self.conv5(x).squeeze()
        #B, 1000
        #subset_logits = self.conv6(x).squeeze()

        k_hot_sample, norm_prob = khs.sample_approx_k_hot(self.subset_size, subset_logits, x_mask, st_prob=False)

        return {
            'subset_mask': k_hot_sample,
            'subset_pre_logits': subset_logits
        }

class QNet(nn.Module):
    def __init__(self, embedding_matrix, cuda=True):
        super(QNet, self).__init__()

        max_features = 5000
        #maxlen = 400
        #batch_size = 40
        embedding_dims = 50

        self.embed = nn.Embedding(max_features, embedding_dims)
        #self.conv1 = nn.Conv1d(100, 100, 5, padding='same')
        self.fc1 = nn.Linear(50, 100)
        #20 classes
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x, mask):
        #(1000,100)
        x = self.embed(x)
        #(100,1000)
        x = x.transpose(1,2)

        #mask: (B,1000)
        x = x*mask.unsqueeze(dim=1)
        #x = F.relu(self.conv1(x))
        #average along sequence
        x = x.mean(dim=2)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits
