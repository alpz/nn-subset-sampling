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
parser.add_argument('--dataset', default='stl10')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--rho', default=0.9, type=float)
parser.add_argument('--n_samples', default=10, type=int)
parser.add_argument('--save_freq', default=10, type=int)

parser.add_argument('--working_dir', default="out/stl10")
args = parser.parse_args()
args_dict = vars(args)

#save generated predictions here
args_dict['experiment_dir'] = "out/stl10"
args_dict['dataset_path'] = "~/datasets"


class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class STL10Resnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, cuda=True):
        super(STL10Resnet, self).__init__()

        self.conv1 = conv_block(in_channels, 64, pool=True)
        self.conv2 = conv_block(64, 128, pool=True)  
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)  
        self.conv4 = conv_block(256, 512, pool=True)  
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, x):
        x = x.reshape((-1,3,96,96))

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out.squeeze()

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

input_shape = (96 * 96 * 3,)

#load unshuffled testloader and a separate unshuffled train_loader
train_loader, test_loader, train_loader_unshuffled = load_data.load_data(args, dataset_path=args_dict['dataset_path'], partial=False)

model = STL10Resnet()
model = model.to(DEVICE)
print(model)
optimizer = Adam(model.parameters(), lr=args.learning_rate,
                weight_decay=1e-4)
#criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.CrossEntropyLoss()
epochs = 80

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
    #x_list = []
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

        #x_list.append(x)
        pred_list.append(pred)

    #xs = torch.cat(x_list, dim=0)
    preds = torch.cat(pred_list, dim=0)

    print('saving ', prefix, preds.shape)
    #np.save(os.path.join(args_dict['experiment_dir'], f'{prefix}_xs.npy'), xs.cpu().numpy())
    np.save(os.path.join(args_dict['experiment_dir'], f'{prefix}_preds.npy'), preds.cpu().numpy())


train_classifier()
with torch.no_grad():
    print('saving train preds')
    save_dataset(train_loader_unshuffled, prefix='train')
    print('saving test preds')
    save_dataset(test_loader, prefix='test')

#test loading
#train_loader2 = load_data.load_cifar_saved(args, path=args_dict['experiment_dir'])
#(x,y) = next(iter(train_loader2))
#print(x.shape)
#print(y)
