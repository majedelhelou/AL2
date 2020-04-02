import torch
import torch.nn as nn

def get_AlexNet_trunk(in_channels=1):
    ''' Returns the trunk of the AlexNet '''
    trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(),
        )
    return trunk

def get_AlexNet_maps(bias=False, all_to_all=False, total_instances=0):
    ''' Returns the maps of the AlexNet
    {total_instances is only needed when all_to_all}
    '''
    num_features = 256*6*6
    if not all_to_all:
        layers = []
        layers.append( nn.Linear(num_features, num_features, bias=bias))
        return nn.Sequential(*layers)
    else:
        for idx2 in range(total_instances - 1):
            layers = []
            layers.append(nn.Linear(num_features, num_features, bias=bias))
            return nn.Sequential(*layers)


def get_AlexNet_fcs(num_classes=10):
    ''' Returns the fcs of the AlexNet '''
    
    fcs = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    return fcs

