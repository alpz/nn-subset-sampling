# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import os
import sys

import numpy as np

import load_text_dataset
import pickle

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

SAVE_DIR = 'out/20ng'

ret = load_text_dataset.load_text_dataset()
embedding_matrix = ret['embedding_matrix']
train_loader = ret['train_loader']
validation_loader = ret['validation_loader']
test_loader = ret['test_loader']
dictionary= ret['dictionary']
labels_index = ret['labels_index']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix,
                                                  freeze=True)
        self.do1 = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv1d(100, 128, 5)
        self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(128, 128, 5)
        self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(128, 128, 5)
        #check?
        self.pool3 = nn.MaxPool1d(35)
        self.do2 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(128, 128)
        #change for fewer classes
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1,2)
        x = self.do1(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.do2(x)
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        #return F.log_softmax(self.fc2(x), dim=1)
        return self.fc2(x)

model = Net().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

print(model)


def train(epoch):
    model.train()
    epoch_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        epoch_loss += loss.data.item()

        loss.backward()

        optimizer.step()

    epoch_loss /= len(train_loader)
    print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)

def evaluate(loader, epoch=None):
    model.eval()
    loss, correct = 0, 0
    pred_vector = torch.LongTensor()
    pred_vector = pred_vector.to(device)

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss += criterion(output, target).data.item()

        pred = output.data.max(1)[1] # get the index of the max log-probability
        pred_vector = torch.cat((pred_vector, pred))

        correct += pred.eq(target.data).cpu().sum()

    loss /= len(loader.dataset)

    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)

    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader.dataset), accuracy))

    if log is not None and epoch is not None:
        log.add_scalar('val_loss', loss, epoch-1)
        log.add_scalar('val_acc', accuracy, epoch-1)

    #return np.array(pred_vector.cpu())
    return accuracy

def save_dataset(loader, prefix):
    print('loading model')
    #model.load_state_dict(torch.load(os.path.join(args_dict['experiment_dir'], 'model.pth')))
    print('creating predictions')
    data_list = []
    pred_list = []
    target_list = []

    #tensor_shape = (-1,) + input_shape
    for data, target in loader:
        #x = x.view(tensor_shape)
        data = data.to(device)
        target = target.to(device)

        output  = model(data)

        pred = output.data.max(1)[1] # get the index of the max log-probability

        data_list.append(data)
        pred_list.append(pred)
        target_list.append(target)

    xs = torch.cat(data_list, dim=0)
    preds = torch.cat(pred_list, dim=0)
    targets = torch.cat(target_list, dim=0)

    print('saving')
    np.save(os.path.join(SAVE_DIR, f'{prefix}_xs.npy'), xs.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, f'{prefix}_preds.npy'), preds.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, f'{prefix}_targets.npy'), targets.cpu().numpy())


epochs = 50

best_valid_acc = 0
for epoch in range(1, epochs + 1):
    train(epoch)
    with torch.no_grad():
        print('Validation set:')
        valid_acc = evaluate(validation_loader, epoch)
        if valid_acc > best_valid_acc:
            print("saving model")
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model.pth'))


print('loading model')
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.pth')))

with torch.no_grad():
    print('Test set:')
    evaluate(test_loader, None)
save_dataset(train_loader, prefix='train')
save_dataset(validation_loader, prefix='valid')
save_dataset(test_loader, prefix='test')
np.save(os.path.join(SAVE_DIR, 'embedding_matrix.npy'), embedding_matrix.cpu().numpy())
dictionary.save(os.path.join(SAVE_DIR, 'dictionary.pkl'))
with open(os.path.join(SAVE_DIR, 'labels_index.pkl'), 'wb') as f:
    pickle.dump(labels_index, f)
