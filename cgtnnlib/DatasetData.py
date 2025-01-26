from dataclasses import dataclass

import pandas as pd
import torch.utils.data
from torch.utils.data import TensorDataset


@dataclass
class DatasetData:
    df: pd.DataFrame
    train_dataset: TensorDataset
    test_dataset: TensorDataset
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
