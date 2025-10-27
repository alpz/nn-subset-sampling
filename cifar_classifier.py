import torch
#from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import sampling.khot as khs
import models.dist as D
from torch.optim import Adam, SGD
import argparse
import numpy as np
import load_data
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MLPVAE')
parser.add_argument('--dataset', default='cifar')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--rho', default=0.9, type=float)
parser.add_argument('--n_samples', default=10, type=int)
parser.add_argument('--save_freq', default=10, type=int)

parser.add_argument('--working_dir', default="out/cifar_full")
args = parser.parse_args()
args_dict = vars(args)

args_dict['experiment_dir'] = "out/cifar_full"

#Cifar with two classes
class CNNCifar(nn.Module):
    def __init__(self, cuda=True):
        """ stacked hierarchical MLP Encoder"""
        super(CNNCifar, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.AvgPool2d(1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(512, 10)
            )

    def forward(self, x):
        x = x.reshape((-1,3,32,32))
        logits = self.model(x)

        return logits.squeeze()

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

input_shape = (32 * 32 * 3,)

train_loader, test_loader = load_data.load_data(args, partial=False)

model = CNNCifar()
model = model.to(DEVICE)
print(model)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
#criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.CrossEntropyLoss()
epochs = 50

def get_accuracy(logits, y):
    pred = (logits > 0).int()
    acc = (pred ==y).float().mean()
    return acc

def train_classifier():
    print("Start training...")
    best_valid_acc = 0.
    for epoch in range(epochs):
        model.train()
        metrics = {'loss': [], 'acc': [] }
        tensor_shape = (-1,) + input_shape
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(tensor_shape)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            #print(x.min(), x.max())
            # binarize
            #x = preprocess(x)
            optimizer.zero_grad()
            logits  = model(x)
            loss = criterion(logits, y)

            pred = logits.data.max(1)[1]  # get the index of the max logit
            # accuracy = get_accuracy(logits, y)
            acc = pred.eq(y.data).float().mean()

            #acc = get_accuracy(logits, y)

            metrics['loss'].append(loss.item())
            metrics['acc'].append(acc.item())

            loss.backward()
            optimizer.step()

        for k in metrics.keys():
            metrics[k] = np.array(metrics[k]).mean()
        # print("\tEpoch", epoch + 1, "", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        print("Epoch {} : Loss {}, acc {}".format(epoch+1, metrics['loss'], metrics['acc']))
        # ------- test --------------
        if (epoch+1)%args.save_freq == 0:
            with torch.no_grad():
                valid_acc = test()
                if valid_acc > best_valid_acc:
                    #torch.save(model, os.path.join(args_dict['experiment_dir'], 'model.pth'))
                    print("saving model")
                    torch.save(model.state_dict(), os.path.join(args_dict['experiment_dir'], 'model.pth'))
                    best_valid_acc= valid_acc

def test():
    model.eval()

    metrics = {'acc': []}
    tensor_shape = (-1,) + input_shape
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.view(tensor_shape)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # binarize
        #x = preprocess(x)
        logits = model(x)

        pred = logits.data.max(1)[1]  # get the index of the max logit
        # accuracy = get_accuracy(logits, y)
        acc = pred.eq(y.data).float().mean()

        #acc = get_accuracy(logits, y)

        metrics['acc'].append(acc.item())

    for k in metrics.keys():
        metrics[k] = np.array(metrics[k])
    # print("\tEpoch", epoch + 1, "", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    print("Test  Acc {}".format(np.mean(metrics['acc'])) )
    print('\n')
    model.train()

    return metrics['acc'].mean()

def save_dataset(loader, prefix):
    print('loading model')
    model.load_state_dict(torch.load(os.path.join(args_dict['experiment_dir'], 'model.pth')))
    print('creating predictions')
    x_list = []
    pred_list = []
    tensor_shape = (-1,) + input_shape
    for batch_idx, (x, y) in enumerate(loader):
        x = x.view(tensor_shape)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        #print(x.min(), x.max())
        # binarize
        #x = preprocess(x)
        logits  = model(x)
        #pred = (logits>0).int()

        pred = logits.data.max(1)[1]  # get the index of the max logit

        x_list.append(x)
        pred_list.append(pred)

    xs = torch.cat(x_list, dim=0)
    preds = torch.cat(pred_list, dim=0)

    print('saving')
    np.save(os.path.join(args_dict['experiment_dir'], f'{prefix}_xs.npy'), xs.cpu().numpy())
    np.save(os.path.join(args_dict['experiment_dir'], f'{prefix}_preds.npy'), preds.cpu().numpy())


train_classifier()
with torch.no_grad():
    print('saving train preds')
    save_dataset(train_loader, prefix='train')
    print('saving test preds')
    save_dataset(test_loader, prefix='test')

#test loading
#train_loader2 = load_data.load_cifar_saved(args, path=args_dict['experiment_dir'])
#(x,y) = next(iter(train_loader2))
#print(x.shape)
#print(y)
