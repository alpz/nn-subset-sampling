#The following contains code from
#https://github.com/CSCfi/machine-learning-scripts


# MIT License
#
# Copyright (c) 2019 CSC - IT Center for Science Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from distutils.version import LooseVersion as LV

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import sys

import numpy as np
import pickle

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))

# TensorBoard is a tool for visualizing progress during training.
# Although TensorBoard was created for TensorFlow, it can also be used
# with PyTorch.  It is easiest to use it with the tensorboardX module.

try:
    import tensorboardX
    import datetime
    logdir = os.path.join(os.getcwd(), "logs",
                          "20ng-cnn-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    log = tensorboardX.SummaryWriter(logdir)
except (ImportError, FileExistsError):
    log = None

# ## GloVe word embeddings
#
# Let's begin by loading a datafile containing pre-trained word
# embeddings.  The datafile contains 100-dimensional embeddings for
# 400,000 English words.


def load_text_dataset():
    if 'DATADIR' in os.environ:
        DATADIR = os.environ['DATADIR']
    else:
        #DATADIR = "/scratch/project_2003747/data/"
        DATADIR = "~/datasets/"

    #GLOVE_DIR = os.path.join(DATADIR, "glove.6B")
    GLOVE_DIR = 'embeddings'

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # ## 20 Newsgroups data set
    #
    # Next we'll load the [20 Newsgroups]
    # (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
    # data set.
    #
    # The dataset contains 20000 messages collected from 20 different
    # Usenet newsgroups (1000 messages from each group):
    #
    # | alt.atheism           | soc.religion.christian   | comp.windows.x     | sci.crypt
    # | talk.politics.guns    | comp.sys.ibm.pc.hardware | rec.autos          | sci.electronics
    # | talk.politics.mideast | comp.graphics            | rec.motorcycles    | sci.space
    # | talk.politics.misc    | comp.os.ms-windows.misc  | rec.sport.baseball | sci.med
    # | talk.religion.misc    | comp.sys.mac.hardware    | rec.sport.hockey   | misc.forsale

    TEXT_DATA_DIR = os.path.join(DATADIR, "20_newsgroups")

    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)

    print('Found %s texts.' % len(texts))

    # Tokenize the texts using gensim.

    tokens = list()
    for text in texts:
        tokens.append(simple_preprocess(text))

    # Vectorize the text samples into a 2D integer tensor.

    MAX_NUM_WORDS = 10000 # 2 words reserved: 0=pad, 1=oov
    MAX_SEQUENCE_LENGTH = 1000

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=0, no_above=1.0,
                               keep_n=MAX_NUM_WORDS-2)

    word_index = dictionary.token2id
    print('Found %s unique tokens.' % len(word_index))

    data = [dictionary.doc2idx(t) for t in tokens]

    #print text
    #print([dictionary[i] for i in (data[0]) if i != -1])

    # Truncate and pad sequences.

    data = [i[:MAX_SEQUENCE_LENGTH] for i in data]
    data = np.array([np.pad(i, (0, MAX_SEQUENCE_LENGTH-len(i)),
                            mode='constant', constant_values=-2)
                     for i in data], dtype=int)
    data = data + 2
    #print(data[0])

    print('Shape of data tensor:', data.shape)
    print('Length of label vector:', len(labels))

    print(set(labels))
    #sys.exit(0)
    # Split the data into a training set and a validation set

    VALIDATION_SET, TEST_SET = 1000, 4000

    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=TEST_SET,
                                                        shuffle=True,
                                                        random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=VALIDATION_SET,
                                                      shuffle=False)

    print('Shape of training data tensor:', x_train.shape)
    print('Length of training label vector:', len(y_train))
    print('Shape of validation data tensor:', x_val.shape)
    print('Length of validation label vector:', len(y_val))
    print('Shape of test data tensor:', x_test.shape)
    print('Length of test label vector:', len(y_test))

    # Create PyTorch DataLoaders for all data sets:

    BATCH_SIZE = 128

    print('Train: ', end="")
    train_dataset = TensorDataset(torch.LongTensor(x_train),
                                  torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    print(len(train_dataset), 'messages')

    print('Validation: ', end="")
    validation_dataset = TensorDataset(torch.LongTensor(x_val),
                                       torch.LongTensor(y_val))
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)
    print(len(validation_dataset), 'messages')

    print('Test: ', end="")
    test_dataset = TensorDataset(torch.LongTensor(x_test),
                                 torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)
    print(len(test_dataset), 'messages')

    # Prepare the embedding matrix:

    print('Preparing embedding matrix.')

    EMBEDDING_DIM = 100

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    n_not_found = 0
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS-2:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i+2] = embedding_vector
        else:
            n_not_found += 1

    embedding_matrix = torch.FloatTensor(embedding_matrix)
    print('Shape of embedding matrix:', embedding_matrix.shape)
    print('Words not found in pre-trained embeddings:', n_not_found)

    return{
        'train_loader': train_loader,
        'validation_loader': validation_loader,
        'test_loader': test_loader,
        'embedding_matrix': embedding_matrix,
        'dictionary': dictionary,
        'labels_index': labels_index,
    }

def load_text_dataset_saved(args, path):

    dictionary = Dictionary()
    dictionary = dictionary.load(os.path.join(path, 'dictionary.pkl'))

    x_train  = np.load(os.path.join(path, 'train_xs.npy'))
    pred_train = np.load(os.path.join(path, 'train_preds.npy'))
    target_train = np.load(os.path.join(path, 'train_targets.npy'))

    x_val  = np.load(os.path.join(path, 'valid_xs.npy'))
    pred_val = np.load(os.path.join(path, 'valid_preds.npy'))
    target_val = np.load(os.path.join(path, 'valid_targets.npy'))

    x_test  = np.load(os.path.join(path, 'test_xs.npy'))
    pred_test = np.load(os.path.join(path, 'test_preds.npy'))
    target_test = np.load(os.path.join(path, 'test_targets.npy'))

    embedding_matrix = np.load(os.path.join(path, 'embedding_matrix.npy'))


    #data = next(iter(x_train))-2
    #print([dictionary[i] for i in (data) if i not in [-2,-1,0]])
    #sys.exit(0)
    # Split the data into a training set and a validation set

    #VALIDATION_SET, TEST_SET = 1000, 4000

    #x_train, x_test, y_train, y_test = train_test_split(data, labels,
    #                                                    test_size=TEST_SET,
    #                                                    shuffle=True,
    #                                                    random_state=42)

    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
    #                                                  test_size=VALIDATION_SET,
    #                                                  shuffle=False)

    print('Shape of training data tensor:', x_train.shape)
    print('Length of training label vector:', len(pred_train))
    print('Shape of validation data tensor:', x_val.shape)
    print('Length of validation label vector:', len(target_val))
    print('Shape of test data tensor:', x_test.shape)
    print('Length of test label vector:', len(target_test))

    # Create PyTorch DataLoaders for all data sets:

    BATCH_SIZE = args.batch_size

    print('Train: ', end="")
    train_dataset = TensorDataset(torch.LongTensor(x_train),
                                  torch.LongTensor(pred_train), torch.LongTensor(target_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    print(len(train_dataset), 'messages')

    print('Validation: ', end="")
    validation_dataset = TensorDataset(torch.LongTensor(x_val),
                                       torch.LongTensor(pred_val), torch.LongTensor(target_val))
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)
    print(len(validation_dataset), 'messages')

    print('Test: ', end="")
    test_dataset = TensorDataset(torch.LongTensor(x_test),
                                 torch.LongTensor(pred_test), torch.LongTensor(target_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False,
                             shuffle=False, num_workers=4)
    print(len(test_dataset), 'messages')

    # Prepare the embedding matrix:

    print('Loading embedding matrix.')

    #EMBEDDING_DIM = 100


    embedding_matrix = torch.FloatTensor(embedding_matrix)
    print('Shape of embedding matrix:', embedding_matrix.shape)
    #print('Words not found in pre-trained embeddings:', n_not_found)

    with open(os.path.join(path, 'labels_index.pkl'), 'rb') as f:
        labels_index = pickle.load(f)

    return{
        'train_loader': train_loader,
        'validation_loader': validation_loader,
        'test_loader': test_loader,
        'embedding_matrix': embedding_matrix,
        'dictionary': dictionary,
        'labels_index': labels_index,
    }

def load_imdb_saved(args, path):

    x_train  = np.load(os.path.join(path, 'x_train.npy'))
    pred_train = np.load(os.path.join(path, 'pred_train.npy')).argmax(axis=1)
    target_train = np.load(os.path.join(path, 'y_train.npy')).argmax(axis=1)

    x_val  = np.load(os.path.join(path, 'x_val.npy'))
    pred_val = np.load(os.path.join(path, 'pred_val.npy')).argmax(axis=1)
    target_val = np.load(os.path.join(path, 'y_val.npy')).argmax(axis=1)


    #data = next(iter(x_train))-2
    #print([dictionary[i] for i in (data) if i not in [-2,-1,0]])
    #sys.exit(0)
    # Split the data into a training set and a validation set

    #VALIDATION_SET, TEST_SET = 1000, 4000

    #x_train, x_test, y_train, y_test = train_test_split(data, labels,
    #                                                    test_size=TEST_SET,
    #                                                    shuffle=True,
    #                                                    random_state=42)

    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
    #                                                  test_size=VALIDATION_SET,
    #                                                  shuffle=False)

    print('Shape of training data tensor:', x_train.shape)
    print('Length of training label vector:', len(pred_train))
    print('Shape of validation data tensor:', x_val.shape)
    print('Length of validation label vector:', len(target_val))

    # Create PyTorch DataLoaders for all data sets:

    BATCH_SIZE = args.batch_size

    print('Train: ', end="")
    train_dataset = TensorDataset(torch.LongTensor(x_train),
                                  torch.LongTensor(pred_train), torch.LongTensor(target_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    print(len(train_dataset), 'messages')

    print('Validation: ', end="")
    validation_dataset = TensorDataset(torch.LongTensor(x_val),
                                       torch.LongTensor(pred_val), torch.LongTensor(target_val))
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)
    print(len(validation_dataset), 'messages')

    # Prepare the embedding matrix:


    #EMBEDDING_DIM = 100



    return{
        'train_loader': train_loader,
        'validation_loader': validation_loader,
        'test_loader': None,
        'embedding_matrix': None,
        'dictionary': None,
        'labels_index': None,
    }
