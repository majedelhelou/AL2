import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from denseNet_utils import DenseNet121, Identity
from alexnet import get_AlexNet_trunk, get_AlexNet_fcs, get_AlexNet_maps

''' 
class funnelNet
epoch_train_funnelNet
test_funnelNet
get_settings_dict
get_model_name
get_map_weights
test_funnelNet_ablations
'''

class funnelNet(nn.Module):
    '''
    ~~~~ inside settings_dict ~~~~
    total_instances: number of network instances in the funnelNet
    all_to_all: False (mapping loss only cyclical), True (mapping loss is from all instances to all others)
    dropout_rate: probability of an element to be zeroed in Dropout
    batch_norm: False (does not do any batch normalization after the conv), True (applies a batch norm layer)
    dataset_name: name of the dataset used to train the network (see train.py)
    bias: True (linear layer in the mapping learns a bias), False (it does not)
    arch_type: 'smallNet', 'denseNet', determines the architecture model of the trunk & fcs
    CCA: True (uses CCA-like mapping with w,s vectors), False (uses linear mapping)
    
    - trunks: is a list of sub-networks (one for each funnel instance), trunks[instance=0] is a sequence of layers
    - fcs:    is a list of sub-networks (one for each funnel instance),    fcs[instance=0] is a sequence of layers
    ==> the difference between trunks and fcs is that the first creates the representation, the second uses it to
        complete the task. The representations created by trunks are made similar during training.
    - maps:   is a list of fully connected layers that map the representation of one instance to another
    '''
    def __init__(self, settings_dict):
        super(funnelNet, self).__init__()

        self.set = settings_dict

        if self.set['dataset_name'] == 'MNIST' or self.set['dataset_name'] == 'FashionMNIST':
            self.in_channels = 1
            self.num_classes = 10
            self.width = 4  #only needed for smallNet
        elif self.set['dataset_name'] == 'CIFAR10':
            self.in_channels = 3
            self.num_classes = 10
            self.width = 5
        elif self.set['dataset_name'] == 'CIFAR100':
            self.in_channels = 3
            self.num_classes = 100
            self.width = 5
        else:
            raise NotImplementedError('dataset not supported, change dataset_name.')

        self.trunks = nn.ModuleList()
        self.fcs = nn.ModuleList()

        if not self.set['CCA']:
            self.maps = nn.ModuleList()
        else:
            self.w = torch.tensor((), requires_grad=True)
            self.s = torch.tensor((), requires_grad=True)
            if not self.set['all_to_all']:
                self.w = self.w.new_ones(self.set['total_instances'], 50)
                self.s = self.s.new_ones(self.set['total_instances'], 50)
            else:
                self.w = self.w.new_ones(self.set['total_instances'] * (self.set['total_instances'] - 1), 50)

        for idx in range(self.set['total_instances']):
            # SMALL NET
            if self.set['arch_type'] == 'smallNet':
                self.trunks.append(self.get_smallNet_trunk(self.set['batch_norm'], self.set['dropout_rate']))
                self.fcs.append(self.get_smallNet_fcs())
                if not self.set['CCA']:
                    self.maps.append(self.get_smallNet_maps(self.set['bias']))

            # ALEX NET
            elif self.set['arch_type'] == 'AlexNet':
                self.trunks.append(get_AlexNet_trunk(self.in_channels))
                self.fcs.append(get_AlexNet_fcs(self.num_classes))
                if not self.set['CCA']:
                    self.maps.append(
                        get_AlexNet_maps(self.set['bias'], self.set['all_to_all'], self.set['total_instances']))

            # DENSE NET
            elif self.set['arch_type'] == 'denseNet':
                net = DenseNet121(self.in_channels, self.num_classes)
                self.fcs.append(net.linear)
                net.linear = Identity()
                self.trunks.append(net)
                if not self.set['CCA']:
                    self.maps.append(self.get_denseNet_maps(self.set['bias']))
                else:
                    raise RuntimeError('CCA not supported for denseNet architecture')

    def forward(self, x):
        ''' mapped_acts have a different definition (dimensions) when using CCA mapping '''
        activations = []
        mapped_acts = []
        probabilities = []

        activations_vectorized = []
        activations_neurons_stretched = []

        for idx in range(self.set['total_instances']):
            activations.append(self.trunks[idx](x))
            activations_vectorized.append(activations[idx].view(activations[idx].shape[0], -1))
            probabilities.append(self.fcs[idx](activations_vectorized[idx]))

            if not self.set['CCA']:
                if not self.set['all_to_all']:
                    mapped_acts.append(self.maps[idx](activations_vectorized[idx]))
                else:
                    for idx2 in range(self.set['total_instances'] - 1):
                        idx_map = idx * (self.set['total_instances'] - 1) + idx2
                        mapped_acts.append(self.maps[idx_map](activations_vectorized[idx]))

            else:
                if not self.set['all_to_all']:
                    activations_neurons_stretched.append(activations[idx].view(activations[idx].shape[1], -1))
                    product_w = torch.matmul(self.w[idx, :], activations_neurons_stretched[idx])
                    mapped_acts.append(product_w)
                    product_s = torch.matmul(self.s[idx, :], activations_neurons_stretched[idx])
                    mapped_acts.append(product_s)
                else:
                    activations_neurons_stretched.append(activations[idx].view(activations[idx].shape[1], -1))
                    for idx2 in range(self.set['total_instances'] - 1):
                        idx_map = idx * (self.set['total_instances'] - 1) + idx2
                        product = torch.matmul(self.w[idx_map, :], activations_neurons_stretched[idx])
                        mapped_acts.append(product)

        return probabilities, activations_vectorized, mapped_acts

    def get_intermediate_rep(self, x):
        ''' Returns activations_0 (output of the first 3 layers) and activations_1 (output of the full trunk)'''
        activations_1 = []
        for idx in range(self.set['total_instances']):
            activations_1.append(self.trunks[idx](x))

        activations_0 = []
        for idx in range(self.set['total_instances']):
            y = copy.deepcopy(x)
            for layer_idx in range(3):
                y = self.trunks[idx][layer_idx](y)
            activations_0.append(y)

        return activations_0, activations_1

    def get_smallNet_trunk(self, batch_norm, dropout_rate):
        ''' Returns the trunk of the small network version '''
        layers = []
        layers.append(nn.Conv2d(self.in_channels, 20, 5, 1))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Conv2d(20, 50, 5, 1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(50, eps=1e-05, momentum=0.1))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(2, 2))
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        return nn.Sequential(*layers)

    def get_smallNet_fcs(self):
        ''' Returns the fcs of the small network version '''
        layers = []
        layers.append(nn.Linear(self.width * self.width * 50, 500))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Linear(500, self.num_classes))
        layers.append(nn.LogSoftmax(dim=1))
        return nn.Sequential(*layers)

    def get_smallNet_maps(self, bias):
        ''' Returns the maps of the small network version '''
        if not self.set['all_to_all']:
            layers = []
            layers.append(nn.Linear(self.width * self.width * 50, self.width * self.width * 50, bias=bias))
            return nn.Sequential(*layers)
        else:
            # each instance maps to the N-1 other instances
            for idx2 in range(self.set['total_instances'] - 1):
                layers = []
                layers.append(nn.Linear(self.width * self.width * 50, self.width * self.width * 50, bias=bias))
                return nn.Sequential(*layers)

    def get_denseNet_maps(self, bias):
        ''' Returns the maps of the DenseNet network version '''
        num_features = 1024
        if not self.set['all_to_all']:
            layers = []
            layers.append(nn.Linear(num_features, num_features, bias=bias))
            return nn.Sequential(*layers)
        else:
            # each instance maps to the N-1 other instances
            for idx2 in range(self.set['total_instances'] - 1):
                layers = []
                layers.append(nn.Linear(num_features, num_features, bias=bias))
                return nn.Sequential(*layers)


def epoch_train_funnelNet(model, map_weight, train_loader, optimizer, mixup_alpha, device):
    '''
    Operation: runs a full training epoch on the funnelNet model
    
    Args: 
        model: funnelNet network instance
        map_weight: weight of the representation mapping loss term
        train_loader: DataLoader instance for reading training data

        optimizer: torch.optim instance
        mixup_alpha: if >0: alpha parameter for mixup, else: no mixup
        device: torch device, cuda/cpu
        
    Returns: 
        train_loss_full: final loss term per batch               [batches x 1]
        train_loss_entropy: entropy loss per batch and instance  [batches x total_instances]
        train_loss_mapping: mapping loss per batch and instance  [batches x total_instances]
    '''
    model.train()
    train_loss_full = []
    train_loss_entropy = []
    train_loss_mapping = []

    N = int(model.set['total_instances'])

    # idx_act, in [0,N-1], takes as input idx_map and returns the instance ID to which we should map
    # this is only needed with the all_to_all training
    if model.set['all_to_all']:
        if not model.set['CCA']:
            idx_act = [0] * N * (N - 1)
            i = 0
            for idx in range(N):
                for idx2 in range(N):
                    if idx2 != idx:
                        idx_act[i] = idx2
                        i += 1
        else:
            idx_act = [0] * N * (N - 1)
            i = 0
            for idx in range(N):
                for idx2 in range(1, N):
                    if idx2 > idx:
                        idx_act[i] = (N - 1) * (idx2) + idx
                    i += 1

    # Train for one epoch:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            indices = torch.randperm(data.size(0))
            shuffled_data = data[indices]
            shuffled_target = target[indices]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            data = lam * data + (1 - lam) * shuffled_data

        # Run through the funnelNet model:
        if model.set['permute_heads']:
            [probabilities, activations, mapped_acts] = model(data)
            # Re-compute the probabilities from activations but using randomized heads:
            new_probabilities = probabilities.copy()
            randomized_indices = np.random.permutation(model.set['total_instances'])
            for idx in range(model.set['total_instances']):
                new_probabilities[idx] = model.fcs[randomized_indices[idx]](activations[idx])
            probabilities = new_probabilities
        else:
            [probabilities, activations, mapped_acts] = model(data)

        # Compute the loss terms for each of the N instances:
        loss_entropy = [0] * N
        if model.set['CCA'] and model.set['all_to_all']:  #the last instance won't have a pair to make
            loss_mapping = [0] * (N - 1)
        else:
            loss_mapping = [0] * N

        for idx in range(N):
            if mixup_alpha > 0:
                loss_entropy[idx] = lam * F.nll_loss(probabilities[idx], target) + (1 - lam) * F.nll_loss(
                    probabilities[idx], shuffled_target)
            else:
                loss_entropy[idx] = F.nll_loss(probabilities[idx], target)

            if not model.set['CCA']:
                if not model.set['all_to_all']:
                    idx2 = (idx + 1) % N
                    loss_mapping[idx] = ((mapped_acts[idx] - activations[idx2])**2).mean()
                else:
                    for idx2 in range(N - 1):
                        idx_map = idx * (N - 1) + idx2
                        # Divide by (N-1) so we have the same scale as with cyclical training:
                        loss_mapping[idx] += (
                            (mapped_acts[idx_map] - activations[idx_act[idx_map]])**2).mean() / (N - 1)
            else:
                if not model.set['all_to_all']:
                    map_w_idx = 2 * idx
                    idx2 = (idx + 1) % N
                    map_s_idx = 2 * (idx2) + 1
                    A = mapped_acts[map_w_idx]
                    B = mapped_acts[map_s_idx]
                    cca_coeff = torch.matmul(A, B) / ((torch.matmul(A, A) * torch.matmul(B, B))**0.5)
                    loss_mapping[idx] = 1 / cca_coeff - 1
                else:
                    for idx2 in range(idx + 1, N):
                        idx_map = idx * (N - 1) + idx2 - 1
                        A = torch.squeeze(mapped_acts[idx_map])
                        B = torch.squeeze(mapped_acts[idx_act[idx_map]])
                        cca_coeff = torch.matmul(A, B) / ((torch.matmul(A, A) * torch.matmul(B, B))**0.5)
                        # Divide by (N-1)/2 so we have the same scale as with cyclical training:
                        loss_mapping[idx] += (1 / cca_coeff - 1) / ((N - 1) / 2)

        ############################# L2 LOSS OVERWRITING THE MAPPING LOSS #############################
        # If only 1 net, the mapping is to 0:
        if N == 1:
            loss_mapping[0] = ((activations[0] - 0)**2).mean()


#         for idx in range(N):
#             loss_mapping[idx] = ((activations[idx])**2).mean()
############################# L2 LOSS OVERWRITING THE MAPPING LOSS #############################
        if map_weight == 0:
            #             loss = sum(loss_entropy)/len(loss_entropy)
            loss = sum(loss_entropy)
        else:
            #             loss = sum(loss_entropy)/len(loss_entropy) + map_weight * sum(loss_mapping)/len(loss_mapping)
            loss = sum(loss_entropy) + map_weight * sum(loss_mapping)

        # Backpropagation:
        loss.backward()
        optimizer.step()
        train_loss_full.append(loss.item())

        # Logging:
        loss_entropy_vals = [loss_entropy[x].item() for x in range(len(loss_entropy))]
        loss_mapping_vals = [loss_mapping[x].item() for x in range(len(loss_mapping))]
        train_loss_entropy.append(loss_entropy_vals)
        train_loss_mapping.append(loss_mapping_vals)

    return train_loss_full, train_loss_entropy, train_loss_mapping


def test_funnelNet(model, map_weight, test_loader, device):
    '''
    Operation: goes over the entire test set to report evaluation results
    
    Inputs:  - model (funnelNet network instance)
             - map_weight (weight of the mapping loss term)
             - test_loader (DataLoader instance for reading testing data)
             - device (torch device, cuda/cpu)
             
    Outputs: - test_loss_full (full loss = entropy+mapping)              [1]
             - test_loss_entropy (classification entropy loss)           [total_instances] 
             - test_loss_mapping (loss in mapping of test activations)   [total_instances]
             - test_accuracy (percentage of correct test labels)         [total_instances] 
             - test_accuracy_ensemble (ensemble percentage correct)      [total_instances] 
    '''
    model.eval()

    with torch.no_grad():
        N = int(model.set['total_instances'])
        test_loss_entropy = [0] * N
        if model.set['CCA'] and model.set['all_to_all']:
            test_loss_mapping = [0] * (N - 1)
        else:
            test_loss_mapping = [0] * N
        test_accuracy = [0] * N
        test_accuracy_ensemble = 0

        if model.set['all_to_all']:
            if not model.set['CCA']:
                idx_act = [0] * N * (N - 1)
                i = 0
                for idx in range(N):
                    for idx2 in range(N):
                        if idx2 != idx:
                            idx_act[i] = idx2
                            i += 1
            else:
                idx_act = [0] * N * (N - 1)
                i = 0
                for idx in range(N):
                    for idx2 in range(1, N):
                        if idx2 > idx:
                            idx_act[i] = (N - 1) * (idx2) + idx
                        i += 1

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            [probabilities, activations, mapped_acts] = model(data)

            # Collect all predictions of each instance in pred_all:
            pred_all = torch.zeros([target.shape[0], N], dtype=target.dtype).to(device)
            for idx in range(N):
                output = probabilities[idx]

                test_loss_entropy[idx] += F.nll_loss(output, target, reduction='sum').item() / len(test_loader.dataset)
                if not model.set['CCA']:
                    if not model.set['all_to_all']:
                        A = mapped_acts[idx]
                        B = activations[(idx + 1) % N]
                        test_loss_mapping[idx] += ((A - B)**2).mean() / len(test_loader.dataset)
                    else:
                        for idx2 in range(N - 1):
                            idx_map = idx * (N - 1) + idx2
                            A = mapped_acts[idx_map]
                            B = activations[idx_act[idx_map]]
                            test_loss_mapping[idx] += (((A - B)**2).mean() / len(test_loader.dataset)) / (N - 1)
                else:
                    if not model.set['all_to_all']:
                        map_w_idx = 2 * idx
                        idx2 = (idx + 1) % N
                        map_s_idx = 2 * (idx2) + 1
                        A = mapped_acts[map_w_idx]
                        B = mapped_acts[map_s_idx]
                        cca_coeff = torch.matmul(A, B) / ((torch.matmul(A, A) * torch.matmul(B, B))**0.5)
                        test_loss_mapping[idx] = 1 / cca_coeff - 1
                    else:
                        for idx2 in range(idx + 1, N):
                            idx_map = idx * (N - 1) + idx2 - 1
                            A = torch.squeeze(mapped_acts[idx_map])
                            B = torch.squeeze(mapped_acts[idx_act[idx_map]])
                            cca_coeff = torch.matmul(A, B) / ((torch.matmul(A, A) * torch.matmul(B, B))**0.5)
                            test_loss_mapping[idx] += (1 / cca_coeff - 1) / ((N - 1) / 2)

                pred = output.argmax(dim=1, keepdim=True)
                pred_all[:, idx] = pred[:, 0]
                correct = pred.eq(target.view_as(pred)).sum().item()
                test_accuracy[idx] += 100. * correct

            pred_ensemble = pred_all.mode()[0]
            correct = pred_ensemble.eq(target.view_as(pred_ensemble)).sum().item()
            test_accuracy_ensemble += 100. * correct

            # If only 1 net, the mapping is from activations to 0:
            if N == 1:
                test_loss_mapping[0] += ((activations[0] - 0)**2).mean() / len(test_loader.dataset)

    test_loss_entropy = [x / len(test_loader.dataset) for x in test_loss_entropy]
    test_loss_mapping = [x / len(test_loader.dataset) for x in test_loss_mapping]
    test_loss_full = sum(test_loss_entropy) / len(test_loss_entropy) + map_weight * sum(test_loss_mapping) / len(
        test_loss_mapping)

    test_accuracy = [x / len(test_loader.dataset) for x in test_accuracy]
    test_accuracy_ensemble /= len(test_loader.dataset)

    return test_loss_full, test_loss_entropy, test_loss_mapping, test_accuracy, test_accuracy_ensemble


def get_settings_dict(opt):
    '''Get a dictionary from argparser option.

    Args: 
        opt: parameters object returned by ArgumentParser.parse_args(). 
    Returns: 
        dictionary containing as keys and values the parameter names and values.
    '''
    settings_dict = dict([(arg, getattr(opt, arg)) for arg in vars(opt)])
    return settings_dict


def get_model_name(settings_dict):
    '''Get the model name from parameters.

    Args: 
        settings_dict: parameter dictionary returned by get_settings_dict. 

    Returns:
        string used for saving models.
    '''
    model_name = '%s_%d_%s_%d_drop_%.2f' % (settings_dict['type_of_model'], settings_dict['total_instances'],
                                            settings_dict['dataset_name'], settings_dict['percent_corrupt'],
                                            settings_dict['dropout_rate'])

    # val = dict.get('key', default) returns the default value if 'key' is not found
    # in the dictionary.
    if settings_dict.get('batch_norm'):
        model_name += '_BN'
    if settings_dict.get('weight_decay', 0) > 0:
        model_name += '_WD'
    if settings_dict.get('aug_data'):
        model_name += '_Aug'
    if settings_dict.get('percent_shrink_data', 100) < 100:
        model_name += '_Shr%d' % settings_dict['percent_shrink_data']
    if settings_dict.get('mixup_alpha', 0) > 0:
        model_name += '_Mix%.1f' % settings_dict['mixup_alpha']
    if settings_dict.get('batch_size', 64) > 64:
        model_name += '_BS%d' % settings_dict['batch_size']
    if settings_dict.get('CCA', False):  # by default CCA mapping is off.
        model_name += '_CCA'
    if settings_dict.get('map_to_0', False):
        model_name += '_to0'
    if not settings_dict.get('bias', False):  # by default bias is off.
        model_name += '_Boff'
    if settings_dict.get('permute_heads', False):
        model_name += '_PH'
    return model_name


def get_map_weights(type_of_model, epochs, mapweight_params):
    '''
    Operation: calculates the map_weight values to be used in training, for each epoch

    Args: 
        type_of_model: 'funnelNet', 'baseline', or 'funnelNetMesh'
        epochs: the number of training epochs
        mapweight_params: contains the 4 parameters {start value}{threshold value}{stepfactor 1}{stepfactor 2}

    Returns:
        map_weights: list of map_weight values, one for each epoch
    '''

    if (type_of_model == 'funnelNet') or (type_of_model == 'funnelNetMesh'):
        map_weights = [0] * epochs
        map_weights[0] = mapweight_params[0]

        for epoch in range(1, epochs):
            if map_weights[epoch - 1] < mapweight_params[1]:
                map_weights[epoch] = map_weights[epoch - 1] * mapweight_params[2]
            else:
                map_weights[epoch] = map_weights[epoch - 1] * mapweight_params[3]

    elif type_of_model == 'baseline':
        map_weights = [0] * epochs

    else:
        raise NotImplementedError('this type of model is not defined.')

    return map_weights


def test_funnelNet_ablations(model, percent_ablate, test_loader, device):
    '''
    Operation: goes over the entire test set to report evaluation results while activations are
               set to 0 in the learned representation, up to percent_ablate percentage
    
    Inputs:  - model (funnelNet network instance)
             - percent_ablate (percentage of activation points to drop to 0
             - test_loader (DataLoader instance for reading test data)
             - device (torch device, cuda/cpu)
             
    Outputs: - test_loss_entropy (classification entropy loss)           [total_instances] 
             - test_accuracy (percentage of correct test labels)         [total_instances] 
             - test_accuracy_ensemble (ensemble percentage correct)      [total_instances] 
    '''
    model.eval()

    with torch.no_grad():
        N = int(model.set['total_instances'])
        test_loss_entropy = [0] * N
        test_accuracy = [0] * N
        test_accuracy_ensemble = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            [probabilities, activations, mapped_acts] = model(data)

            # Collect all predictions of each instance in pred_all
            pred_all = torch.zeros([target.shape[0], N], dtype=target.dtype).to(device)
            for idx in range(N):
                chosen_indices = np.random.choice([0, 1],
                                                  size=(activations[idx].shape),
                                                  p=[percent_ablate / 100, 1 - percent_ablate / 100])
                activations[idx] = activations[idx] * torch.tensor(chosen_indices,
                                                                   dtype=activations[idx].dtype).to(device)
                ###############################################################################################
                output = model.fcs[idx](activations[idx])

                test_loss_entropy[idx] += F.nll_loss(output, target, reduction='sum').item() / len(test_loader.dataset)

                pred = output.argmax(dim=1, keepdim=True)
                pred_all[:, idx] = pred[:, 0]
                correct = pred.eq(target.view_as(pred)).sum().item()
                test_accuracy[idx] += 100. * correct

            pred_ensemble = pred_all.mode()[0]
            correct = pred_ensemble.eq(target.view_as(pred_ensemble)).sum().item()
            test_accuracy_ensemble += 100. * correct

    test_loss_entropy = [x / len(test_loader.dataset) for x in test_loss_entropy]
    test_accuracy = [x / len(test_loader.dataset) for x in test_accuracy]
    test_accuracy_ensemble /= len(test_loader.dataset)

    return test_loss_entropy, test_accuracy, test_accuracy_ensemble
