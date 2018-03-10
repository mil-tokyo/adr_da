import sys

sys.path.append('../loader')
from unaligned_data_loader import UnalignedDataLoader
from svhn import load_svhn
from mnist import load_mnist
from usps import load_usps
import numpy as np
import torch


def return_dataset(data, scale=False, usps=False, all_use=False):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(all_use=all_use)
    return train_image, train_label, test_image, test_label


def dataset_read(source, target, batch_size, pixel_norm=True, scale=False, all_use=False):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                            usps=usps, all_use=all_use)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                            all_use=all_use)
    #normalize with mean value of pixels
    if pixel_norm:
        pixel_mean = np.vstack([train_source, train_target]).mean((0,))
        train_source = (train_source - pixel_mean) / float(255)
        train_target = (train_target - pixel_mean) / float(255)
        test_target = (test_target - pixel_mean) / float(255)

    S['imgs'] = torch.from_numpy(train_source).float()
    S['labels'] = torch.from_numpy(s_label_train).long()

    T['imgs'] = torch.from_numpy(train_target).float()
    T['labels'] = torch.from_numpy(t_label_train).long()

    # input target samples for both 
    S_test['imgs'] = torch.from_numpy(test_target).float()
    S_test['labels'] = torch.from_numpy(t_label_test).long()
    T_test['imgs'] = torch.from_numpy(test_target).float()
    T_test['labels'] = torch.from_numpy(t_label_test).long()
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test
