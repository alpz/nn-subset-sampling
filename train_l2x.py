import argparse

import json
import random
import sys
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.pyplot.switch_backend('agg')

import numpy as np
import subprocess
import datetime
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
# from stability.resnet2 import ResNet18Enc, ResNet18Dec
from torch.optim import Adam, SGD
import os
# from utils import show_gray_image, show_image, show_image_grid
import torchvision
import torchvision.transforms as transforms
import build
from torch import linalg as LA
import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='L2XCifar')
parser.add_argument('--dataset', default='cifar')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

parser.add_argument('--task', default='cifar')
parser.add_argument('--t', default=1)

parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--save_freq', default=10)
parser.add_argument('--generate_freq', default=10, type=int)

parser.add_argument('--eval_freq', default=10)
parser.add_argument('--kl_eval', default=False)

parser.add_argument('--working_dir', default="out/l2x_cifar")
args = parser.parse_args()
args_dict = vars(args)

dataset_path = '~/datasets'
saved_model_path = 'out/cifar_full'
experiment_id = 1

model_map = {
    'L2XCifar': build.l2x_cifar,
    'L2XCifarSubop': build.l2x_cifar_subop,
}
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

batch_size = args.batch_size

assert(args.task == args.dataset)

lr = args.learning_rate
epochs = 600

if args.dataset == 'cifar':
    input_shape = (32*32*3,)

    train_loader, test_loader = load_data.load_cifar_saved(args, path=saved_model_path)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = model_map[args.model](args)
model = model.to(DEVICE)
print(model)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)


#def preprocess(x):
#    return torch.bernoulli(x)



def train():
    time_path = os.path.join(args_dict['experiment_dir'], 'time.txt')
    time_file = open(time_path, 'w')
    print("Start training...")
    model.train()
    best_valid_acc = 0
    start_time = time.time()
    for epoch in range(epochs):
        metrics = {'loss': [], 'accuracy': []}
        tensor_shape = (-1,) + input_shape
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(tensor_shape)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            ret = model(x,y)
            loss = ret['loss']

            metrics['loss'].append(loss.item())
            metrics['accuracy'].append(ret['accuracy'].item())

            loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time
        for k in metrics.keys():
            metrics[k] = np.array(metrics[k]).mean()
        # print("\tEpoch", epoch + 1, "", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        print("Epoch {} t {}: Loss {}, ACC {}".format(epoch+1, args_dict['experiment_id'], metrics['loss'], metrics['accuracy'] ))
        print(f"mean k {ret['mean_k']} min {ret['min_k']} max {ret['max_k']}")
        
        print("{}, {}".format(epoch_time, metrics['accuracy']), file=time_file)
        time_file.flush()

        # ------- test --------------
        if (epoch+1)%args.save_freq == 0:
            with torch.no_grad():
                valid_acc = test()
                if valid_acc > best_valid_acc:
                    #torch.save(model, os.path.join(args_dict['experiment_dir'], 'model.pth'))
                    torch.save(model.state_dict(), os.path.join(args_dict['experiment_dir'], 'model.pth'))
                    best_valid_acc = valid_acc

        # --------generate -----------
        if (epoch+1)%args.generate_freq == 0:
            with torch.no_grad():

                #recon = (ret['masked_input'][0:64]).reshape(-1,3,32,32).permute(0,2,3,1)
                ##recon = (_recon+1)/2
                #recon = recon.contiguous().reshape((-1,3*32*32)).detach().cpu()
                #generate_samples(None, epoch=epoch, generated=recon, reconstruct=True, batch_size=64, task='cifar', prefix="masked")

                subset_logits = ret['subset_logits'][0:64].detach().cpu().numpy()
                k_out = ret['k_out'][0:64].detach().cpu().numpy()
                mask = get_mask_from_logits(subset_logits, k_out)

                mask = torch.tensor(mask).reshape(-1,3,32,32).permute(0,2,3,1).to(DEVICE)

                recon_1 = (x[0:64]).reshape(-1,3,32,32).permute(0,2,3,1)
                recon_1 = (recon_1+1)/2
                neg_recon = (1-mask)*recon_1 #+ mask*0.3*recon_1
                neg_recon = neg_recon.contiguous().reshape((-1,3*32*32)).detach().cpu()

                #neg_recon = (ret['neg_masked_input'][0:64]).reshape(-1,3,32,32).permute(0,2,3,1)
                #neg_recon = (neg_recon+1)/2
                #neg_recon = neg_recon.contiguous().reshape((-1,3*32*32)).detach().cpu()
                generate_samples(None, epoch=epoch, generated=neg_recon, reconstruct=True, batch_size=64, task='cifar', prefix="neg_masked")
                #generate_samples(None, epoch=epoch, generated=recon1, reconstruct=True, batch_size=64, task='cifar', prefix="x")


        sys.stdout.flush()
    print("End")

def get_mask_from_logits(logits, k):
    """  topk from each row of logits set to 1.
        sorting indices
        mask with last k
        unsort mask
    """
    k = k.astype(np.int32)
    n,d = logits.shape
    z = np.zeros_like(logits)
    crange = np.arange(n)[:,np.newaxis]

    sort_idx = logits.argsort(axis=-1)
    unsort_idx = sort_idx.argsort(axis=-1)
    for i in range(n):
        z[i,-k[i]:] = 1
    topk_mask = z[crange,unsort_idx]
    return topk_mask.astype(np.float32)

def test():
    model.eval()

    metrics = {'loss': [], 'accuracy': []}
    tensor_shape = (-1,) + input_shape
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.view(tensor_shape)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # binarize
        #x = preprocess(x)
        ret = model(x,y, evaluate=True)
        loss = ret['loss']

        metrics['loss'].append(loss.item())
        metrics['accuracy'].append(ret['accuracy'].item())

    for k in metrics.keys():
        metrics[k] = np.array(metrics[k]).mean()
    # print("\tEpoch", epoch + 1, "", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    print("Test Loss {}, ACC {}".format(metrics['loss'], metrics['accuracy']))
    print('\n')
    model.train()

    return metrics['accuracy']


def generate_samples(trainer, h=28, w=28, batch_size=128, epoch=0, reconstruct=False, task='mnist', prefix="",
                     generated=None):
    # Test the trained model: generation
    # Sample noise vectors from N(0, 1)

    n = np.sqrt(batch_size).astype(np.int32)
    if task in ['mnist', 'omni', 'sbn']:
        h = w = 28
        I_generated = np.empty((h * n, w * n))
    if task == 'cifar':
        h = w = 32
        c = 3
        I_generated = np.empty((h * n, w * n, c))
    # if z is None:
    #  z = np.random.normal(size=[batch_size, model.hparams.n_latent])
    if reconstruct:
        # x_generated = model.reconstructor(X)
        # x_generated = trainer.model.reconstructed_samples.numpy()
        x_generated = generated.numpy()
    # if generate:
    else:
        x_generated = generated.numpy()

    for i in range(n):
        for j in range(n):
            # I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(h, w)
            if task in ['mnist', 'omni', 'sbn']:
                I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = x_generated[i * n + j, :].reshape(28, 28)
            else:
                I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x_generated[i * n + j, :].reshape(h, w, c)

    plt.figure(figsize=(8, 8))
    if task in ['mnist', 'omni', 'sbn']:
        plt.imshow(I_generated, cmap='gray')
    else:
        plt.imshow(I_generated)

    # plt.show()
    if reconstruct:
        plt.savefig('{}/recon_samples_{}_{}.pdf'.format(args_dict['experiment_dir'], prefix,epoch))
    else:
        plt.savefig('{}/gen_samples_{}.pdf'.format(args_dict['experiment_dir'], epoch))
    plt.close()


def redirect_stdout(outfile):
    class Transcript(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.logfile = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.logfile.write(message)

        def flush(self):
            self.terminal.flush()
            self.logfile.flush()
            # pass

    print("redirecting to ", outfile)
    sys.stdout = Transcript(outfile)
    print("\n\nNew ---- {} -----\n\n".format(datetime.datetime.now()))
    # TODO output command line


def experiment_name():
    """ build experiment dir """
    # global experiment_id
    keys = ['model', 'learning_rate', 'task', 't']
    hparams = {x: args_dict[x] for x in keys}

    if not args.resume:
        # set try number
        dirs = os.listdir(args.working_dir)
        num_list = [int(re.search(r't_(\d+)', d).group(1) or 0) for d in dirs]
        try:
            hparams['t'] = max(num_list) + 1
        except:
            hparams['t'] = 1
    experiment_id = hparams['t']
    args_dict['experiment_id'] = experiment_id

    task_num = int(hparams['t'])
    args_dict['t'] = task_num

    hparams = sorted(hparams.items())
    hparams = (map(str, x) for x in hparams)
    hparams = ('_'.join(x) for x in hparams)
    hparams_str = '.'.join(hparams)
    experiment_dir = os.path.join(args.working_dir, hparams_str)

    if os.path.exists(experiment_dir) and not args.resume:
        print("Directory exists and not resuming")
        sys.exit(0)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    return experiment_dir


def write_source_files(summ_dir):
    stdout_log = os.path.join(summ_dir, 'stdout.txt')
    redirect_stdout(stdout_log)

    # write git diff, commit hash, redirect stdout
    diff = os.path.join(summ_dir, 'git.diff')

    if not os.path.isfile(diff):
        with open(diff, 'w') as fd:
            subprocess.call(['git diff'], stdout=fd, stderr=fd, shell=True)

    # write commit hash
    commit = os.path.join(summ_dir, 'commit.txt')
    if not os.path.isfile(commit):
        with open(commit, 'w') as fd:
            subprocess.call(['git rev-parse HEAD'], stdout=fd, stderr=fd, shell=True)


def main():
    if args.eval and args.resume:
        print('eval only')
        model.load_state_dict(torch.load(os.path.join(args_dict['experiment_dir'], 'model.pth')))
        test()

    else:
        train()


if __name__ == "__main__":
    exp_dir = experiment_name()
    write_source_files(exp_dir)
    args_dict['experiment_dir'] = exp_dir
    print(args_dict)
    main()
