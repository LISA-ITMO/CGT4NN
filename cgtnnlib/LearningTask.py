## LearningTask v.0.1
## Updated at Wed 15 Jan 2025

from typing import Literal, TypeAlias
from dataclasses import dataclass

import torch
import torch.nn as nn

LearningTaskName: TypeAlias = Literal['classification', 'regression']

@dataclass
class LearningTask:
    name: LearningTaskName
    criterion: nn.CrossEntropyLoss | nn.MSELoss
    dtype: torch.dtype
    
    def metrics(self):
        match self.name:
            case 'regression':
                return ['r2', 'mse']
            case 'classification':
                return ['f1', 'accuracy', 'roc_auc']
            case _:
                raise TypeError(f'bad task name: {self.name}')


CLASSIFICATION_TASK = LearningTask(
    name='classification',
    criterion=nn.CrossEntropyLoss(),
    dtype=torch.long,
)

REGRESSION_TASK = LearningTask(
    name='regression',
    criterion=nn.MSELoss(),
    dtype=torch.float,
)

def is_classification_task(task: LearningTask) -> bool:
    return task.name == CLASSIFICATION_TASK.name

def is_regression_task(task: LearningTask) -> bool:
    return task.name == REGRESSION_TASK.name