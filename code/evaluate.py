#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
evaluate.py: Evaluate the trained models.

Compute the activations for a test dataset, and the similarities (svcca, pwcca, etc.) between different 
representations. This script reads the results of type epoch_{number}.pt. Your need to run the training before using this module.

'''

import os
import sys

import numpy as np
import pandas as pd
import pickle
import torch
from torchvision import datasets, transforms

sys.path.append('../svcca/')
import cca_core

from funnelNet_utils import funnelNet, get_model_name
from MNIST_utils import my_MNIST
from CIFAR_utils import my_CIFAR10


ALL_TO_ALL = False
BATCH_NORM = False
RESULTS_DIR = '../models_ICASSP/'


def get_svcca_similarity(acts1, acts2, K=20, verbose=False, epsilon=None):
    ''' Compute svcca similarity, adapted from tutorial on
    https://github.com/google/svcca/tree/master/tutorials.
    '''
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:K] * np.eye(K), V1[:K])
    svacts2 = np.dot(s2[:K] * np.eye(K), V2[:K])
    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=epsilon, verbose=verbose)
    return svcca_results


def get_similarity(acts1, acts2, verbose=False, epsilon=1e-10, method='mean'):
    import pwcca

    if method == 'all':
        similarity_dict = cca_core.get_cca_similarity(acts1, acts2, verbose=verbose, epsilon=epsilon)
        return similarity_dict['cca_coef1']
    if method == 'mean':
        similarity_dict = cca_core.get_cca_similarity(acts1, acts2, verbose=verbose, epsilon=epsilon)
        return similarity_dict['mean'][0]  # contains two times the same value.
    elif method == 'svcca':
        similarity_dict = get_svcca_similarity(acts1, acts2, K=10, verbose=verbose, epsilon=epsilon)
        return similarity_dict['mean'][0]
    elif method == 'pwcca':
        pwcca_mean, w, __ = pwcca.compute_pwcca(acts1, acts2, epsilon=epsilon)
        return pwcca_mean
    else:
        raise NotImplementedError(method)


def get_activations(type_of_models, opt_dict, test_batch_size, epochs, results_dir=RESULTS_DIR):
    device = torch.device(type='cpu')
    if 'MNIST' in opt_dict['dataset_name']:
        test_loader = torch.utils.data.DataLoader(my_MNIST(
            '../data_MNIST',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307, ), (0.3081, ))])),
                                                  batch_size=test_batch_size,
                                                  shuffle=False)
        size = [50, 4, 4]

    elif opt_dict['dataset_name'] == 'CIFAR10':
        test_loader = torch.utils.data.DataLoader(my_CIFAR10(
            '../data_CIFAR10',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307, ), (0.3081, ))])),
                                                  batch_size=test_batch_size,
                                                  shuffle=False)
        size = [50, 5, 5]
    else:
        raise NotImplementedError(opt_dict['dataset_name'])

    print('Chosen batch size {} results in {} epochs.'.format(test_batch_size, len(test_loader)))

    activations = {m: {} for m in type_of_models}
    mapped_activations = {m: {} for m in type_of_models}
    for this_model_name in type_of_models:
        opt_dict['type_of_model'] = this_model_name

        model_name = get_model_name(opt_dict)
        this_model = funnelNet(opt_dict)
        for epoch in epochs:
            try:
                fname = os.path.join(results_dir, model_name, 'epoch_{}.pt'.format(epoch))
                with open(fname, 'rb') as f:
                    results_dict = torch.load(f, map_location='cpu')
                results_dict.pop('total_instances_const', None)
                this_model.load_state_dict(results_dict)
            except FileNotFoundError:
                print(f'WARNING: did not find model at {fname}. skipping.')
                continue

            this_model.eval()

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    __, this_acts, mapped_acts = this_model(data)
                    # we are just looking at first activation.
                    break

            this_acts = [_.contiguous().view([test_batch_size, *size]) for _ in this_acts]
            mapped_acts = [_.contiguous().view([test_batch_size, *size]) for _ in mapped_acts]

            this_acts = [_.permute(0, 2, 3, 1).contiguous().view([-1, 50]) for _ in this_acts]
            mapped_acts = [_.permute(0, 2, 3, 1).contiguous().view([-1, 50]) for _ in mapped_acts]

            activations[this_model_name][epoch] = this_acts
            mapped_activations[this_model_name][epoch] = mapped_acts

    return activations, mapped_activations


def compute_similarity_measures(models_to_compare, activations, epochs):
    '''  Compute pwcca, svcca, etc. for the given inputs.

    Args:
        * models_to_compare: list of model names to compare. 
        * activations: dictionary of activations.
        * epochs: epochs to use

    Returns:
        dataframe with all similarity measures computed.

    '''
    plot_intermediate = False  #True
    results = pd.DataFrame(
        columns=['epoch', 'this_model', 'other_model', 'this_idx', 'other_idx', 'mean', 'svcca', 'pwcca', 'all_coeffs'])

    shape = activations['baseline'][epochs[0]][0].shape

    for epoch in epochs:
        for i, this_model in enumerate(models_to_compare):
            for other_model in models_to_compare[i:]:

                # model names and indices
                this_name, this_idx = this_model.split(' ')
                other_name, other_idx = other_model.split(' ')
                this_idx = int(this_idx)
                other_idx = int(other_idx)

                this_acts = other_acts = activations

                if this_name in this_acts.keys():
                    this_act = this_acts[this_name][epoch][this_idx].numpy()
                elif this_name == 'random':
                    this_act = np.random.uniform(size=shape)
                else:
                    raise NotImplementedError(this_name)
                if other_name in other_acts.keys():
                    other_act = other_acts[other_name][epoch][other_idx].numpy()
                elif other_name == 'random':
                    other_act = np.random.uniform(size=shape)
                else:
                    raise NotImplementedError(this_name)

                if plot_intermediate:
                    plt.figure()
                    plt.plot(other_act[0, :], color='black')
                    plt.plot(other_act[1, :], color='black')
                    plt.plot(other_act[2, :], color='black')
                    plt.plot(this_act[0, :], color='red')
                    plt.plot(this_act[1, :], color='red')
                    plt.plot(this_act[2, :], color='red')
                    plt.title('{} and {}, epoch {}'.format(this_model, other_model, epoch))
                    plt.show()

                all_sim = get_similarity(this_act.T, other_act.T, verbose=False, epsilon=1e-10, method='all')
                mean_sim = get_similarity(this_act.T, other_act.T, verbose=False, epsilon=1e-10, method='mean')
                svcca_sim = get_similarity(this_act.T, other_act.T, verbose=False, epsilon=1e-10, method='svcca')
                pwcca_sim = get_similarity(this_act.T, other_act.T, verbose=False, epsilon=1e-10, method='pwcca')

                results_dict = dict(epoch=epoch,
                                    this_model=this_name,
                                    other_model=other_name,
                                    this_idx=this_idx,
                                    other_idx=other_idx,
                                    svcca=svcca_sim,
                                    pwcca=pwcca_sim,
                                    mean=[mean_sim],
                                    all_coeffs=[all_sim])
                print(f'epoch {epoch}, {this_model} to {other_model}: {mean_sim:.4f}')
                new_df = pd.DataFrame(results_dict)
                results = results.append(new_df, ignore_index=True, sort=False)
    return results


if __name__ == "__main__":

    out_dir = '../results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ### Choose models to evaluate
    type_of_models = ['funnelNet', 'baseline']
    test_batch_size = 10000
    epochs = range(0, 501, 100)  

    opt_dict = dict(
        type_of_model='',  # will be filled later, by baseline, funnelNet
        total_instances=4,  # always 4
        dataset_name='MNIST',  # always MNIST
        percent_corrupt=75,  # 15,25,50,75,85
        dropout_rate=0,  # dropout rate, 0, 0.5
        bias=False,
        CCA=False,
        arch_type='smallNet',
        batch_norm=False,
        all_to_all=False)

    ### Compute activations (can be commented out, intermediate results are saved.)
    activations, __ = get_activations(type_of_models, opt_dict, test_batch_size, epochs)
    if not any([len(activations[t]) > 0 for t in type_of_models]):
        raise ValueError('Did not find required results files. Make sure to run training first.')

    fname = f'{out_dir}/activations_{test_batch_size}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(activations, f)
    print('saved as', fname)

    ### Compute similarity measures (SVCCA, PWCCA, ...)
    models_to_compare = []
    models_to_compare += ['baseline {}'.format(i) for i in range(4)]
    models_to_compare += ['funnelNet {}'.format(i) for i in range(4)]
    models_to_compare += ['random 0']

    fname = f'{out_dir}/activations_{test_batch_size}.pkl'
    with open(fname, 'rb') as f:
        activations = pickle.load(f)
    print('read', fname)

    fname = f'{out_dir}/results_{test_batch_size}.pkl'
    results = compute_similarity_measures(models_to_compare, activations, epochs)
    results.to_pickle(fname)
    print('saved as', fname)
