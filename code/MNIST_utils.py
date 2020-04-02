import codecs
import errno
import os

import numpy as np
from PIL import Image
import torch.utils.data as data
import torch

'''
shrink_data
augment_data_rot
get_wrong_targets
class my_MNIST
'''


def shrink_data(train_data, train_labels, percent_shrink_data):
    ''' 
    Operation: shrinks the training data (and labels) down to percent_shrink_data of its original size (randomly chosen)
    
    Inputs: 
            - train_data (tensor containing all training samples)
            - train_labels (corresponding classification labels)
            - percent_shrink_data (percentage of training data to be retained)
    Output:
            - train_data_shrink (shrunken training data tensor)
            - train_labels_shrink (shrunken labels tensor corresponding to the train data)
    '''

    chosen_indices = np.random.choice([1, 0],
                                      size=(len(train_labels), ),
                                      p=[percent_shrink_data / 100, 1 - percent_shrink_data / 100])

    # Get the shape of the final tensor:
    final_data_size = sum(chosen_indices)
    for i in range(len(train_data.shape)):
        if i == 0:
            shrink_shape_list = [final_data_size]
        else:
            shrink_shape_list.append(train_data.shape[i])

    train_data_shrink = torch.zeros(shrink_shape_list, dtype=train_data.dtype)
    train_labels_shrink = torch.zeros(final_data_size, dtype=train_labels.dtype)

    shrunk_ID = 0
    for sample in range(len(train_labels)):
        if chosen_indices[sample] == 1:
            train_data_shrink[shrunk_ID, ] = train_data[sample]
            train_labels_shrink[shrunk_ID] = train_labels[sample]
            shrunk_ID += 1

    return train_data_shrink, train_labels_shrink


def augment_data_rot(train_data, train_labels):
    ''' 
    Operation: doubles the number of training samples by randomly choosing
               rotations 90, 180, 270, and adding them to the original data samples
    Inputs: 
            - train_data (tensor containing all training samples)
            - train_labels (corresponding classification labels)
    Output:
            - train_data_aug (tensor containing all training samples with rotation augmentation)
            - train_labels_aug (corresponding classification labels of the augmented dataset)
    '''

    aug_factor = 2  #if !=2: adjust for loop
    chosen_rots = np.random.choice([90, 180, 270], size=(len(train_labels), ))

    for i in range(len(train_data.shape)):
        if i == 0:
            aug_shape_list = [train_data.shape[0] * aug_factor]
        else:
            aug_shape_list.append(train_data.shape[i])

    train_data_aug = torch.zeros(aug_shape_list, dtype=train_data.dtype)
    train_labels_aug = torch.zeros(aug_factor * len(train_labels), dtype=train_labels.dtype)

    # Copy original data:
    train_data_aug[0:train_data.shape[0], ] = train_data
    train_labels_aug[0:len(train_labels)] = train_labels

    # Add another copy made up of rotations:
    for sample in range(len(train_labels)):
        index_aug = train_data.shape[0] + sample
        train_labels_aug[index_aug] = train_labels[sample]
        x = train_data[sample, ]

        if chosen_rots[sample] == 90:
            train_data_aug[index_aug, ] = x.transpose(0, 1).flip(0)

        elif chosen_rots[sample] == 180:
            train_data_aug[index_aug, ] = x.flip(0, 1)

        elif chosen_rots[sample] == 270:
            train_data_aug[index_aug, ] = x.transpose(0, 1).flip(1)

    return train_data_aug, train_labels_aug


def get_wrong_targets(target, percent_randomized, C):
    ''' 
    Operation: given a tensor of targets in {0,C-1}, selects percent_randomized % of values to set to incorrect
               random values (in {0,C-1} too), and returns the new targets that are not fully correct
    
    Inputs: 
            - target contains the class IDs (labels or targets)
            - percent_randomized (corruption percentage for setting wrong labels or targets)
            - C (the total number of classes)
    Output:
            - modified list of class IDs
    '''

    chosen_indices = np.random.choice([1, 0],
                                      size=(len(target), ),
                                      p=[percent_randomized / 100, 1 - percent_randomized / 100])

    for target_id in range(len(target)):
        if chosen_indices[target_id] == 1:
            # Choose a random target different from the correct one:
            target_rand_options = [x for x in range(0, C) if x != target[target_id]]
            target[target_id] = int(np.random.choice(target_rand_options))

    return target


class my_MNIST(data.Dataset):
    '''`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        percent_randomized (float, optional): percentage of labels to be corrupted
        aug_data (bool, optional): If True, augments the data once with rotations
        percent_shrink_data (int, optional): If <100, shrinks the data to that percentage
        
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    '''

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 percent_randomized=0,
                 aug_data=False,
                 percent_shrink_data=100):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

            if percent_shrink_data < 100:
                if (not isinstance(percent_shrink_data,
                                   int)) or (percent_shrink_data > 100) or (percent_shrink_data < 1):
                    raise RuntimeError('Non-valid percentage for shrinking data (must be int in [1,100])')
                else:
                    [self.train_data, self.train_labels] = shrink_data(self.train_data, self.train_labels,
                                                                       percent_shrink_data)
                    print('Shrunk data by random sampling, down to %d%%' % percent_shrink_data)

            if percent_randomized > 0:
                if percent_randomized > 100:
                    raise RuntimeError('Non-valid percentage for randomizing data (must be in [0,100])')
                else:
                    self.train_labels = get_wrong_targets(self.train_labels, percent_randomized, 10)
                    print('Randomized %d %% of training labels to wrong values' % percent_randomized)

            if aug_data:
                [self.train_data, self.train_labels] = augment_data_rot(self.train_data, self.train_labels)
                print('Augmented data (double) using the extra rotations 90, 180, 270')

        else:
            self.test_data, self.test_labels = torch.load(os.path.join(self.root, self.processed_folder,
                                                                       self.test_file))

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        '''
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        '''Download the MNIST data if it doesn't exist in processed_folder already.'''
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
                        read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte')))
        test_set = (read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
                    read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte')))
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class my_FashionMNIST(my_MNIST):
    '''`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    '''
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
