## Dataset class v.0.1
## Created at 23 Nov 2024
## Updated at Tue 14 Jan 2025

from dataclasses import dataclass
from typing import Callable, Union

from sklearn.model_selection import train_test_split

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from cgtnnlib.LearningTask import LearningTask

from cgtnnlib.DatasetData import DatasetData
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.nn.NetworkLike import NetworkLike

from cgtnnlib.constants import BATCH_SIZE, RANDOM_STATE, TEST_SAMPLE_SIZE


## Utils


def tensor_dataset_from_dataframe(
    df: pd.DataFrame,
    target: str,
    y_dtype: torch.dtype
) -> TensorDataset:
    X = df.drop(columns=[target]).values
    y = df[target].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=y_dtype)

    return TensorDataset(X_tensor, y_tensor)


def tensorize_and_split(
    df: pd.DataFrame,
    target: str,
    y_dtype: torch.dtype,
    test_size: float,
    random_state: int,
) -> tuple[TensorDataset, TensorDataset]:
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    return (
        tensor_dataset_from_dataframe(
            df=train_df,
            target=target,
            y_dtype=y_dtype,
        ),
        tensor_dataset_from_dataframe(
            df=val_df,
            target=target,
            y_dtype=y_dtype
        )
    )



@dataclass
class Dataset:
    number: int
    name: str
    learning_task: LearningTask
    # ::: can be derived from DatasetData
    classes_count: int
    target: str
    data_maker: Callable[[], pd.DataFrame]

    _data: Union[DatasetData, None] = None

    @property
    def features_count(self) -> int:
        return self.data.train_dataset[1][0].shape[0]

    def model_a_path(self, params: ExperimentParameters) -> str:
        "!!! Please don't use this"

        # ::: PthPath config variable?
        return f'pth/model-{self.number}A-c-P{params.p}_N{params.iteration}.pth'

    def model_b_path(self, params: ExperimentParameters) -> str:
        "!!! Please don't use this"

        return f'pth/model-{self.number}B-c-P{params.p}_N{params.iteration}.pth'

    def model_path(self, params: ExperimentParameters, model: NetworkLike) -> str:
        return f'pth/cgtnn-{self.number}X-{type(model).__name__}-c-P{params.p}_N{params.iteration}.pth'

    @property
    def data(self) -> DatasetData:
        if self._data is None:
            df = self.data_maker()
            train, test = tensorize_and_split(
                df=df,
                target=self.target,
                y_dtype=self.learning_task.dtype,
                test_size=TEST_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
            
            self._data = DatasetData(
                train_dataset=train,
                test_dataset=test,
                train_loader=DataLoader(
                    train,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                ),
                test_loader=DataLoader(
                    test,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                ),
            )
        
        _data: DatasetData = self._data
        return _data
    
    def __repr__(self):
        return ''.join([
            f'Dataset #{self.number} {{', '\n',
            f'  name: "{self.name}"', '\n',
            f'  learning_task: {self.learning_task}', '\n',
            f'  classes_count: {self.classes_count}', '\n',
            f'  target: "{self.target}"', '\n',
            f'  _data: {self._data}', '\n',
            '}',
        ])