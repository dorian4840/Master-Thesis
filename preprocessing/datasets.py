import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

class IMPALA_Dataset(Dataset):

    def __init__(self, data_path):
        self.X = torch.load(f"{data_path}_input")
        self.y = torch.load(f"{data_path}_labels")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y[idx, :]


def create_dataloaders(dataset, batch_size=32, shuffle=True, drop_last=False):
    """
    Shuffle the data, split it into train, val and test set and create dataloaders.

    NOTE: Data is not yet seeded. Need to seed data in the training function.
    """

    # Split train/validation/test = 70/10/20
    train_set, valid_set, test_set = random_split(dataset, [0.7, 0.1, 0.2])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_dataloader, val_dataloader, test_dataloader
