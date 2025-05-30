## LearningTask v.0.1
## Updated at Wed 15 Jan 2025

from typing import Literal, TypeAlias
from dataclasses import dataclass

import torch
import torch.nn as nn

LearningTaskName: TypeAlias = Literal["classification", "regression"]

Criterion = nn.CrossEntropyLoss | nn.MSELoss


@dataclass
class LearningTask:
    """
    Represents a learning task with associated properties and metrics.

        This class encapsulates the name, evaluation criterion, and expected output
        data type for a machine learning task. It also provides a method to determine
        appropriate metrics based on the task type.

        Attributes:
            name: The name of the learning task.
            criterion: The evaluation criterion used for the task.
            y_dtype: The expected data type of the target variable.

        Methods:
            metrics(): Returns a list of appropriate metrics for the task.
    """

    name: LearningTaskName
    criterion: Criterion
    y_dtype: torch.dtype

    @property
    def metrics(self):
        """
        Returns a list of appropriate metrics for the task.

            Args:
                None

            Returns:
                list: A list of strings representing the metrics to be used,
                      depending on the task type (regression or classification).
                      Raises TypeError if an invalid task name is provided.
        """
        match self.name:
            case "regression":
                return [
                    "smape",
                    "r2",
                    "mse",
                ]
            case "classification":
                return ["f1", "accuracy", "roc_auc"]
            case _:
                raise TypeError(f"bad task name: {self.name}")


CLASSIFICATION_TASK = LearningTask(
    name="classification",
    criterion=nn.CrossEntropyLoss(),
    y_dtype=torch.long,
)

REGRESSION_TASK = LearningTask(
    name="regression",
    criterion=nn.MSELoss(),
    y_dtype=torch.float,
)


def is_classification_task(task: LearningTask) -> bool:
    """
    Checks if a given learning task is a classification task.

        Args:
            task: The learning task to check.

        Returns:
            bool: True if the task is a classification task, False otherwise.
    """
    return task.name == CLASSIFICATION_TASK.name


def is_regression_task(task: LearningTask) -> bool:
    """
    Checks if a given learning task is a regression task.

        Args:
            task: The learning task to check.

        Returns:
            bool: True if the task is a regression task, False otherwise.
    """
    return task.name == REGRESSION_TASK.name
