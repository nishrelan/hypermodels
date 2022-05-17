import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np


class VariableLengthDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

    def __len__(self):
        return 1 if len(self.x_data) > 0 else 0

    def __getitem__(self, idx):
        probs = torch.ones(len(self.x_data)) * 0.1
        selections = torch.bernoulli(probs)
        selections = torch.flatten(torch.nonzero(selections))
        x_points = self.x_data[selections]
        y_points = self.y_data[selections]

        if len(x_points) == 0:
            print("hi")
            return np.expand_dims(self.x_data[0], axis=0), np.expand_dims(self.y_data[0], axis=0)
        else:
            return x_points, y_points


class NumpyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def collate_fn(samples):
    xb, yb = list(zip(*samples))
    xb = np.stack(xb)
    yb = np.array(yb)
    return xb, yb


def variable_collate(samples):
    xb, yb = list(zip(*samples))
    xb = np.stack(xb)
    yb = np.array(yb)
    num_total = len(xb)
    idxs = np.random.choice(num_total, size=10, replace=False)
    return xb[idxs], yb[idxs]


if __name__ == '__main__':
    pass
