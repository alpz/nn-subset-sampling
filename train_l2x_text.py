import argparse

import json
import random
import sys
import os
import re
import matplotlib
import matplotlib.pyplot as plt

import load_text_dataset

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
from torch.optim import Adam, SGD, RMSprop
import os
# from utils import show_gray_image, show_image, show_image_grid
import torchvision
import torchvision.transforms as transforms
import build
from torch import linalg as LA
import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='L2X20ng')
parser.add_argument('--dataset', default='20ng')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

parser.add_argument('--subset_size', default=50, type=int)
parser.add_argument('--diffk', default=False, action='store_true', help='set differentiable k')
parser.add_argument('--tau', default=1, type=float, help='temperature for Gumbel baseline')
parser.add_argument('--correct', default=False, action='store_true', help='apply correction')

parser.add_argument('--task', default='20ng')
parser.add_argument('--t', default=1)

parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--save_freq', default=10, type=int)
parser.add_argument('--generate_freq', default=10, type=int)

parser.add_argument('--eval_freq', default=10)

parser.add_argument('--working_dir', default="out/l2x_20ng")
args = parser.parse_args()
args_dict = vars(args)

dataset_path = '~/datasets'
experiment_id = 1

model_map = {
    'L2X20ng': build.l2x_20ng,
    'L2X20ngSubop': build.l2x_20ng_rr,
    'L2Ximdb': build.l2x_imdb,
}
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

batch_size = args.batch_size

assert(args.task == args.dataset)

lr = args.learning_rate
epochs =300

if args.dataset == '20ng':
    saved_model_path = 'out/20ng'
    ret = load_text_dataset.load_text_dataset_saved(args, path=saved_model_path)
    embedding_matrix = ret['embedding_matrix']
    train_loader = ret['train_loader']
    validation_loader = ret['validation_loader']
    test_loader = ret['test_loader']
    dictionary = ret['dictionary']
    labels_index = ret['labels_index']

if args.dataset == 'imdb':
    saved_model_path = 'out/imdb'
    ret = load_text_dataset.load_imdb_saved(args, path=saved_model_path)
    embedding_matrix = None
    train_loader = ret['train_loader']
    validation_loader = ret['validation_loader']
    test_loader = ret['test_loader']
    dictionary = None
    labels_index = None


model = model_map[args.model](args, embedding_matrix)
model = model.to(DEVICE)
print(model)
#optimizer = Adam(model.parameters(), lr=lr)
optimizer = RMSprop(model.parameters(), lr=lr)


def train():
    print("Start training...")
    best_valid_acc = 0
    for epoch in range(epochs):
        model.train()
        metrics = {'loss': [], 'correct': 0, 'mean_k':0}
        for (x, y, target) in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            ret = model(x,y,target)
            loss = ret['loss']

            loss.backward()
            optimizer.step()

            metrics['loss'].append(loss.item())
            metrics['correct']+=(ret['correct'].item())

        #for k in metrics.keys():
        metrics['loss'] = np.array(metrics['loss']).mean()
        metrics['accuracy'] = metrics['correct']/len(train_loader.dataset)
        metrics['mean_k'] = ret['mean_k'].item()

        # print("\tEpoch", epoch + 1, "", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        print("Epoch {} t {}: Loss {}, ACC {}".format(epoch+1, args_dict['experiment_id'], metrics['loss'], metrics['accuracy'] ))
        if args.diffk:
          print(f"mean k {metrics['mean_k']} min {ret['min_k']} max {ret['max_k']}")

        # ------- test --------------
        if (epoch+1)%args.save_freq == 0:
            with torch.no_grad():
                valid_acc = test(validation_loader, epoch)
                if valid_acc > best_valid_acc:
                    #torch.save(model, os.path.join(args_dict['experiment_dir'], 'model.pth'))
                    torch.save(model.state_dict(), os.path.join(args_dict['experiment_dir'], 'model.pth'))
                    best_valid_acc = valid_acc

        # --------generate -----------
        #if (epoch+1)%args.generate_freq == 0 and args.dataset in ['20ng']:
        #    with torch.no_grad():
        #        #save subset
        #        gen_save_path = os.path.join(args_dict['experiment_dir'], f'doc_{epoch}.txt')

        #        mask = ret['mask'][0:100].cpu().numpy()
        #        data = x[0:100].cpu().numpy()-2
        #        pred = y[0:100].cpu().numpy()
        #        target = target[0:100].cpu().numpy()
        #        var_pred = ret['predictions'][0:100].cpu().numpy()
        #        with open(gen_save_path, 'w') as f:
        #            f.write(str(labels_index)+"\n\n")
        #            for num, doc in enumerate(data):
        #                text = [dictionary[token]+", "+str(int(mask[num,i])) for i,token in enumerate(doc)  \
        #                                  if token not in [-2,-1,0]]

        #                mask_num = np.array([int(mask[num, i]) for i, token in enumerate(doc) if token not in [-2, -1, 0]]).sum()
        #                f.write(' '.join(text))
        #                f.write('\n\n')
        #                f.write("T: " + str(target[num]) + ", P: " +str(pred[num]) + ", VP: "+ str(var_pred[num]))
        #                f.write(f"selected: {mask_num}")
        #                f.write('\n\n---------------------------\n\n')

        sys.stdout.flush()
    print("End training")

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

def test(loader, epoch=0):
    model.eval()

    metrics = {'loss': [], 'target_correct': 0, 'correct':0}
    for (x, y, target) in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        target = target.to(DEVICE)

        ret = model(x, y, target, evaluation=True)
        loss = ret['loss']

        metrics['loss'].append(loss.item())
        #metrics['target_accuracy'].append(ret['target_accuracy'].item())
        metrics['target_correct'] += (ret['target_correct'].item())
        metrics['correct'] += (ret['correct'].item())

    metrics['loss'] = np.array(metrics['loss'])
    metrics['target_accuracy'] = metrics['target_correct']/len(loader.dataset)
    metrics['accuracy'] = metrics['correct']/len(loader.dataset)

    print("Eval Loss {}, TACC {} ACC {}".format(np.mean(metrics['loss']), np.mean(metrics['target_accuracy']),
                                                metrics['accuracy']))

    #---------------------------- save
    if args.dataset in ['20ng']:
        with torch.no_grad():
            # save subset
            gen_save_path = os.path.join(args_dict['experiment_dir'], f'doc_valid_{epoch}.txt')

            #mask = ret['mask'][0:100].cpu().numpy()
            subset_logits = ret['subset_logits'][0:100].cpu().numpy()
            k_out = ret['k_out'][0:100].cpu().numpy()
            mask = get_mask_from_logits(subset_logits, k_out)
            data = x[0:100].cpu().numpy() - 2
            pred = y[0:100].cpu().numpy()
            target = target[0:100].cpu().numpy()
            var_pred = ret['predictions'][0:100].cpu().numpy()
            with open(gen_save_path, 'w') as f:
                f.write(str(labels_index) + "\n\n")
                for num, doc in enumerate(data):
                    text = [dictionary[token] + ", " + str(int(mask[num, i])) for i, token in enumerate(doc) \
                            if token not in [-2, -1, 0]]

                    mask_num = np.array([int(mask[num, i]) for i, token in enumerate(doc) if token not in [-2, -1, 0]]).sum()
                    total = np.array([1 for i, token in enumerate(doc) if token not in [-2, -1, 0]]).sum()
                    f.write(' '.join(text))
                    f.write('\n\n')
                    f.write("T: " + str(target[num]) + ", P: " + str(pred[num]) + ", VP: " + str(var_pred[num]))
                    f.write(f" selected: {mask_num}/{total}")
                    f.write('\n\n---------------------------\n\n')

        print('Saved sample output in ', args_dict['experiment_dir'])
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

        test_acc = test(test_loader, "test")
        print(f"Test accuracy {test_acc}")

    else:
        train()
        print('testing')
        model.load_state_dict(torch.load(os.path.join(args_dict['experiment_dir'], 'model.pth')))

        test_acc = test(test_loader, "test")
        print(f"Test accuracy {test_acc}")

if __name__ == "__main__":
    exp_dir = experiment_name()
    #write_source_files(exp_dir)
    args_dict['experiment_dir'] = exp_dir
    print(args_dict)
    main()
