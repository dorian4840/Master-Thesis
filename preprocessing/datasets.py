import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Dataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y[idx, :]


def create_dataloaders(X, y, batch_size=32, seed=42):
    """
    Shuffle the data, split it into train, val and test set and create dataloaders.

    NOTE: train, val and test are split, 0.7, 0.1, 0.2.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=seed)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                    test_size=0.66,
                                                    random_state=seed)
    
    train_set = Dataset(X_train, y_train)
    val_set = Dataset(X_val, y_val)
    test_set = Dataset(X_test, y_test)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader
