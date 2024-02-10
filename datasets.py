import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IMPALA_dataset(Dataset):
    def __init__(self, X, y_avpu, y_crt):
        self.X = X
        self.y_avpu = y_avpu
        self.y_crt = y_crt

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y_avpu[idx, :], self.y_crt[idx, :]


def create_datasets(dataset, data_split):
    """ Split dataset by patient ids. """

    random_ids = np.random.permutation(list(dataset.keys()))

    # training
    train_size = int(len(random_ids) * data_split[0])
    train_ids = random_ids[:train_size]
    X = np.concatenate([v['X'] for k, v in dataset.items() if k in train_ids], axis=0)
    y_avpu = np.concatenate([v['y'][0] for k, v in dataset.items() if k in train_ids], axis=0)
    y_crt = np.concatenate([v['y'][1] for k, v in dataset.items() if k in train_ids], axis=0)
    train_set = IMPALA_dataset(torch.from_numpy(X), torch.from_numpy(y_avpu), torch.from_numpy(y_crt))

    # validation
    val_size = int(len(random_ids) * data_split[1])
    val_ids = random_ids[train_size:train_size+val_size]
    X = np.concatenate([v['X'] for k, v in dataset.items() if k in val_ids], axis=0)
    y_avpu = np.concatenate([v['y'][0] for k, v in dataset.items() if k in val_ids], axis=0)
    y_crt = np.concatenate([v['y'][1] for k, v in dataset.items() if k in val_ids], axis=0)
    val_set = IMPALA_dataset(torch.from_numpy(X), torch.from_numpy(y_avpu), torch.from_numpy(y_crt))
    
    # test
    test_ids = random_ids[train_size+val_size:]
    X = np.concatenate([v['X'] for k, v in dataset.items() if k in test_ids], axis=0)
    y_avpu = np.concatenate([v['y'][0] for k, v in dataset.items() if k in test_ids], axis=0)
    y_crt = np.concatenate([v['y'][1] for k, v in dataset.items() if k in test_ids], axis=0)
    test_set = IMPALA_dataset(torch.from_numpy(X), torch.from_numpy(y_avpu), torch.from_numpy(y_crt))

    # Show additional information
    total = len(train_set) + len(val_set) + len(test_set)
    print(f'Dataset statistics (data split={data_split}):')
    print(f' Training set size  : {len(train_set)} ({round((len(train_set) / total)*100, 2)}%)')
    print(f' Validation set size: {len(val_set)} ({round((len(val_set) / total)*100, 2)}%)')
    print(f' Test set size      : {len(test_set)} ({round((len(test_set) / total)*100, 2)}%)')

    return train_set, val_set, test_set


def create_dataloaders(dataset, data_split, batch_size, shuffle=True, drop_last=False):
    """ Create separate training, validation and test dataloaders. """

    train_set, val_set, test_set = create_datasets(dataset, data_split)

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last,
                                  pin_memory=True,
                                  num_workers=4)
    
    val_dataloader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                pin_memory=True,
                                num_workers=4)

    test_dataloader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 drop_last=drop_last,
                                 pin_memory=True,
                                 num_workers=4)
    
    return train_dataloader, val_dataloader, test_dataloader

