from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
import errno
import codecs
import matplotlib.pyplot as plt
import pickle

from MNIST_utils import my_MNIST, my_FashionMNIST
from CIFAR_utils import *

from funnelNet_utils import *
from timeit import default_timer as timer

import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
parser = argparse.ArgumentParser(description="funnelNet")

# Training settings:
parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training")
parser.add_argument("--test_batch_size", type=int, default=1000, help="input batch size for testing")
parser.add_argument("--epochs", type=int, default=250, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
parser.add_argument("--weight_decay", type=float, default=0, help="L2 penalty on weights (5*1e-3/e-4)")

parser.add_argument("--permute_heads", type=bool, default=False, help="if True, at inference heads are permuted randomly")

# Data settings:
parser.add_argument("--dataset_name", type=str, default='MNIST', help="supported: MNIST, FashionMNIST, CIFAR10, CIFAR100")
parser.add_argument("--percent_corrupt", type=int, default=85, help="percentage of training samples that are corrupt")
parser.add_argument("--aug_data", type=bool, default=False, help="if True, augments the dataset with rotations")
parser.add_argument("--percent_shrink_data", type=int, default=100, help="if <100, decreases the training data size")
parser.add_argument("--mixup_alpha", type=float, default=0, help="if >0, applies mixup with parameter mixup_alpha")

# Model settings:
parser.add_argument("--type_of_model", type=str, default='funnelNet', help="types: funnelNet, funnelNetMesh, baseline")
parser.add_argument("--arch_type", type=str, default='smallNet', help="types: smallNet, AlexNet, denseNet")
parser.add_argument("--CCA", type=bool, default=False, help="if True, the mapping is not linear but CCA-like")
parser.add_argument("--total_instances", type=int, default=4, help="number of instances inside the funnelNet")
parser.add_argument("--bias", type=bool, default=False, help="if True, mapping linear layer learns a bias")
parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout (1D) rate, in [0,1]")
parser.add_argument("--batch_norm", type=bool, default=False, help="if True, applies a BN layer after second conv")

# Mapping strategy:
# Note: to map to 0, simply use N=1 with funnel, and nothing else needs to be replaced
parser.add_argument("--mapweight_start", type=float, default=0.01, help="start value for the mapping weight")
parser.add_argument("--mapweight_thres", type=float, default=5, help="threshold value for the mapping weight")
parser.add_argument("--mapweight_fac1", type=float, default=1.1, help="first growth factor for the mapping weight")
parser.add_argument("--mapweight_fac2", type=float, default=1.01, help="second growth factor for the mapping weight")

# Seeds, cuda and saving:
parser.add_argument("--seed_data", type=float, default=0, help="random seed for NumPy")
parser.add_argument("--seed_model", type=float, default=0, help="random seed for PyTorch")
parser.add_argument("--save_model", type=bool, default=True, help="for saving the current model")
parser.add_argument("--no_cuda", type=bool, default=False, help="disables CUDA training")

opt = parser.parse_args()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def main():
    
    models_dir = '../models_ICASSP'
    
    if opt.arch_type == 'denseNet':
        opt.lr = 0.001
        opt.momentum = 0.01
        opt.weight_decay = 5e-4
    
    use_cuda = (not opt.no_cuda) and (torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    settings_dict = get_settings_dict(opt)
    
    print('Setting the NumPy random seed to %d' %opt.seed_data)
    np.random.seed(opt.seed_data)
    print('Setting the PyTorch random seed to %d' %opt.seed_model)
    torch.manual_seed(opt.seed_model)
    
    
    # Preparing data loaders:
    ####################################################################################################
    classes = dict(
        MNIST=my_MNIST,
        FashionMNIST=my_FashionMNIST,
        CIFAR10=my_CIFAR10,
        CIFAR100=my_CIFAR100)
    normalizations = dict(
        MNIST=[(0.1307,), (0.3081,)],
        FashionMNIST=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        CIFAR10=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        CIFAR100=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)])

    if opt.arch_type == 'smallNet':
        transform = transforms.Compose([ transforms.ToTensor(),
                                   transforms.Normalize(*normalizations[opt.dataset_name]) ])
    elif opt.arch_type == 'AlexNet':
        transform = transforms.Compose([ transforms.Resize(224), transforms.ToTensor(),
                                        transforms.Normalize(*normalizations[opt.dataset_name]) ])

    train_loader = torch.utils.data.DataLoader(
            classes[opt.dataset_name]('../data_{}'.format(opt.dataset_name), train=True, download=True,
                             transform=transform, 
                             percent_randomized=opt.percent_corrupt, 
                             aug_data=opt.aug_data, 
                             percent_shrink_data=opt.percent_shrink_data),
                             batch_size=opt.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(classes[opt.dataset_name]('../data_{}'.format(opt.dataset_name), train=False,
                            transform=transform), batch_size=opt.test_batch_size, shuffle=False, **kwargs)
    ####################################################################################################
        
    print('Initializing ~%s~ with: %d instances, %s architecture' %(opt.type_of_model, opt.total_instances, opt.arch_type))
        
    settings_dict['all_to_all'] = (opt.type_of_model == 'funnelNetMesh')
    
    model_funnel = funnelNet(settings_dict)
    model_funnel.to(device)
    if opt.arch_type == 'denseNet':
        print('Kaiming normalization on denseNet')
        model_funnel.apply(weights_init_kaiming)
        
    params_funnel = model_funnel.parameters()
    optimizer_funnel = optim.SGD(params_funnel, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    mapweight_params = [opt.mapweight_start, opt.mapweight_thres, opt.mapweight_fac1, opt.mapweight_fac2]
    map_weights = get_map_weights(opt.type_of_model, opt.epochs, mapweight_params)

    model_name = get_model_name(settings_dict)
    if (opt.save_model):
        fname = os.path.join(models_dir, model_name, 'settings_dict.pkl')
        if not os.path.isdir(os.path.join(models_dir, model_name)):
            os.makedirs(os.path.join(models_dir, model_name))
        with open(fname, 'wb') as f:
            pickle.dump(settings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('wrote', fname)


    all_train_loss_full = []
    all_train_loss_entropy = []
    all_train_loss_mapping = []
    all_test_loss_full = []
    all_test_loss_entropy = []
    all_test_loss_mapping = []
    all_test_accuracy = []
    all_test_accuracy_ensemble = []

    print('Started training...')
    for epoch in range(opt.epochs):
        
        ##~~ TRAIN ~~##
        start = timer()
        [train_loss_full, train_loss_entropy, train_loss_mapping] = epoch_train_funnelNet(model_funnel, map_weights[epoch], train_loader, optimizer_funnel, opt.mixup_alpha, device)
        all_train_loss_full.append(train_loss_full)
        all_train_loss_entropy.append(train_loss_entropy)
        all_train_loss_mapping.append(train_loss_mapping)
            
        ##~~ TEST ~~##
        [test_loss_full, test_loss_entropy, test_loss_mapping, test_accuracy, test_accuracy_ensemble] = test_funnelNet(model_funnel, map_weights[epoch], test_loader, device)
        all_test_loss_full.append(test_loss_full)
        all_test_loss_entropy.append(test_loss_entropy)
        all_test_loss_mapping.append(test_loss_mapping)
        all_test_accuracy.append(test_accuracy)
        all_test_accuracy_ensemble.append(test_accuracy_ensemble)
        
        end = timer()
        print('    Epoch #%3d/%d: %.4f%%  ML: %.5f  (%.1f sec)' %(epoch, opt.epochs-1, sum(test_accuracy)/len(test_accuracy), np.asarray(train_loss_mapping).mean(), end-start))
    
        ##~~ SAVE ~~##
        if (opt.save_model) and (epoch % 10 == 0):
            torch.save(model_funnel.state_dict(), os.path.join(models_dir, model_name, "epoch_%d.pt" % epoch))


        if (opt.save_model) and (epoch % 50 == 0):
            results_dict = {'all_train_loss_full': np.asarray(all_train_loss_full),
                            'all_train_loss_entropy': np.asarray(all_train_loss_entropy),
                            'all_train_loss_mapping': np.asarray(all_train_loss_mapping),
                            'all_test_loss_full': np.asarray(all_test_loss_full),
                            'all_test_loss_entropy': np.asarray(all_test_loss_entropy),
                            'all_test_loss_mapping': np.asarray(all_test_loss_mapping),
                            'all_test_accuracy': np.asarray(all_test_accuracy),
                            'all_test_accuracy_ensemble': np.asarray(all_test_accuracy_ensemble)}

            fname = os.path.join(models_dir, model_name, 'results_dict.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('wrote', fname)


if __name__ == "__main__":
    main()
    
