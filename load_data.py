
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import load_stl10


def load_data(args, dataset_path='~/datasets', partial=False):
    batch_size = args.batch_size

    if args.dataset == 'stl10':
        transform = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        kwargs = {'num_workers': 2, 'pin_memory': False}

        train_dataset = torchvision.datasets.STL10(root=dataset_path, split='train',
                                                     download=True, transform=transform)
        test_dataset = torchvision.datasets.STL10(root=dataset_path, split='test',
                                                    download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False,
                                                   shuffle=True, **kwargs)
        train_loader_unshuffled = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False,
                                                   shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=False,
                                                  shuffle=False, **kwargs)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_loader, test_loader, train_loader_unshuffled

    if args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        kwargs = {'num_workers': 2, 'pin_memory': False}

        train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                                     download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                                    download=True, transform=transform_test)
        if partial:
           # select classes you want to include in your subset
            #classes = torch.tensor([0, 1, 2, 3, 4])
            classes = torch.tensor([0, 1])

            # get indices that correspond to one of the selected classes
            indices = (torch.tensor(train_dataset.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
            # subset the dataset
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

            test_indices = (torch.tensor(test_dataset.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
            # subset the dataset
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

            #train_indices = (train_dataset.targets==0) | (train_dataset.targets==1)
            #train_dataset.data = train_dataset.data[train_indices]
            ##train_dataset.targets = train_dataset.targets[train_indices]
            #train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

            #test_indices = (test_dataset.targets==0) | (test_dataset.targets==1)
            #test_dataset.data = test_dataset.data[test_indices]
            #test_dataset.targets = test_dataset.targets[test_indices]


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                                   shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True,
                                                  shuffle=False, **kwargs)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0), std=(1))
        ])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=dataset_path, download=True, train=True, transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=dataset_path, download=True, train=False, transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=False)

    if args.dataset == 'omni':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([28,28]),
            # transforms.Normalize(mean=(0), std=(1))
        ])
        train_d = torchvision.datasets.Omniglot(root=dataset_path, download=True, background=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_d,
            batch_size=batch_size, drop_last=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.Omniglot(root=dataset_path, download=True, background=False, transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, test_loader

def load_cifar_saved(args, path,  partial=False):
    
    kwargs = {'num_workers': 2, 'pin_memory': False}

    train_x = np.load(os.path.join(path,'train_xs.npy'))
    train_preds = np.load(os.path.join(path, 'train_preds.npy'))

    test_x = np.load(os.path.join(path, 'test_xs.npy'))
    test_preds = np.load(os.path.join(path, 'test_preds.npy'))

    batch_size = args.batch_size

    t_x = torch.Tensor(train_x)
    t_y = torch.LongTensor(train_preds)

    test_x = torch.Tensor(test_x)
    test_y = torch.LongTensor(test_preds)

    train_dataset = torch.utils.data.TensorDataset(t_x, t_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                                   shuffle=True, **kwargs)

    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True,
                                               shuffle=False, **kwargs)

    return train_loader, test_loader


def load_stl10_saved(args, dataset_path, saved_path, partial=False):
    kwargs = {'num_workers': 2, 'pin_memory': False}

    #train_x = np.load(os.path.join(path, 'train_xs.npy'))
    train_pred_path = os.path.join(saved_path, 'train_preds.npy')

    #test_x = np.load(os.path.join(path, 'test_xs.npy'))
    test_pred_path = os.path.join(saved_path, 'test_preds.npy')

    batch_size = args.batch_size

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    kwargs = {'num_workers': 2, 'pin_memory': False}

    #our loader with saved labels does not download data. Define dummy dataset to download
    _ = torchvision.datasets.STL10(root=dataset_path, split='train',
                                                 download=True, transform=transform)

    train_dataset = load_stl10.STL10(root=dataset_path, split='train',
                                               download=False, transform=transform,
                                                labels_file_path=train_pred_path)
    train_dataset_untransformed = load_stl10.STL10(root=dataset_path, split='train',
                                     download=False, transform=transform_test,
                                     labels_file_path=train_pred_path)

    test_dataset = load_stl10.STL10(root=dataset_path, split='test',
                                              download=False, transform=transform_test,
                                              labels_file_path=test_pred_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False,
                                               shuffle=True, **kwargs)
    train_loader_untransformed = torch.utils.data.DataLoader(train_dataset_untransformed, batch_size=batch_size, drop_last=True,
                                                          shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True,
                                              shuffle=False, **kwargs)

    return train_loader, test_loader, train_loader_untransformed
