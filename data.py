import torch
import numpy as np
from typing import Callable, List

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NpDataset(Dataset):
    """ Pytorch dataset the right way"""

    def __init__(self, array, labels):
        self.array = torch.from_numpy(array)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        """ Total number of samples """
        return len(self.array)

    # def __add__(self, ds):
        # """ Method to add data to dataset"""
        # return data_utils.TensorDataset(torch.from_numpy(self.array + ds))

    def __getitem__(self, i):
        """ Return single sample given index i.

        Here we are grabbing the sample from a class property. Alternatively,
        given a very large dataset, we can load data on the fly from disk or
        create the data on the fly. See here for example:
        stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
        """
        X = self.array[i]
        y = self.labels[i]

        return X, y

def operator_targets(ops: List[Callable], data: np.ndarray):
    """ Evenly apply operators to the data, prepend with op index and return
    target data (i.e. result of applying ops)

    Each op should accept a 1-D np array.
    """
    chunks = np.array_split(data, len(ops))
    new_x = [None]*len(ops)
    labels = [None]*len(ops)

    # Split and process
    for i, chunk in enumerate(chunks):
        # Apply the function for each row in chunk
        label = np.apply_along_axis(ops[i], axis=1, arr=chunk)
        labels[i] = np.vstack(label)
        # Prepend data with index i
        new_x[i] = np.insert(chunk, 0, i, axis=1)

    # TODO: stack new data and labels
    new_x = np.concatenate(new_x, axis=0)
    labels = np.concatenate(labels, axis=0)
    return new_x, labels

def make_op_dataset(low,high,samples, seq_len, batch_size,ops):
    """
    Generate input of size
    """
    # Generate random samples
    x = np.random.randint(low, high, (samples, seq_len))*1.0

    # Make sure no dupes
    x = np.unique(x, axis=0)
    max_it=100
    for i in range(max_it):
        if len(x) == samples:
            break
        else:
            h = np.random.randint(low, high, (samples-len(x), seq_len))*1.0
            x = np.concatenate([x,h],axis=0)
            x = np.unique(x, axis=0)
    assert (len(x) == samples),"Could not generate a unique dataset"
    np.random.shuffle(x)

    # train/valid/test split
    tr_id = int(samples*0.8)
    valid_id = int(tr_id + samples*0.1)
    inputs = train, valid, test = np.split(x, [tr_id, valid_id])

    # Get targets and loaders
    loaders = []
    for inp in inputs:
        # Get targets and modified inputs
        x_in, y = operator_targets(ops, inp)

        # Make PyTorch dataloaders
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'num_workers': 6}

        x_in, y = torch.from_numpy(x_in), torch.from_numpy(y)
        dataset = torch.utils.data.TensorDataset(x_in, y)
        loader = torch.utils.data.DataLoader(dataset, **params)
        loaders.append(loader)

    return loaders

def get_toy_data(batch_size):
    dataset_settings = {
            'batch_size': batch_size,
            'samples' : 100,
            'seq_len': 10,
            'low' : 0,
            'high' : 100,
            'ops' : [np.max, np.min, np.mean, np.sum]
            }

    train_loader,valid_loader,test_loader = make_op_dataset(**dataset_settings)
    return train_loader, valid_loader, test_loader, dataset_settings

if __name__ == '__main__':

    # Test settings
    dataset_settings = {
            'batch_size': 5,
            'samples' : 100,
            'seq_len': 10,
            'low' : 0,
            'high' : 100,
            'ops' : [np.max, np.min, np.mean, np.sum]
            }

    train_loader,valid_loader,test_loader = make_op_dataset(**dataset_settings)

    pass
