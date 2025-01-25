from dataclasses import dataclass

import torch.utils.data
from torch.utils.data import TensorDataset


@dataclass
class DatasetData:
    train_dataset: TensorDataset
    test_dataset: TensorDataset
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
