## EvaluationParameters v.0.2
## Created at Thu 28 Nov 2024
## Modified at Sun 1 Dec 2024
## v.0.2 removed inputs_count and outputs_count
##       (they may be derived from dataset)

from dataclasses import dataclass

from cgtnnlib.Dataset import Dataset
from cgtnnlib.LearningTask import LearningTask, is_regression_task
from cgtnnlib.ExperimentParameters import ExperimentParameters


@dataclass
class EvaluationParameters:
    """
    Holds parameters needed for evaluating a model.

        Attributes:
            dataset: The dataset being evaluated.
            model_path: Path to the trained model.
            experiment_parameters: Parameters used during the experiment.
            report_key: Key for identifying this evaluation in reports.

        Methods:
            is_regression: Checks if the task is a regression task.
            task: Returns the learning task associated with this dataset.
    """

    dataset: Dataset
    model_path: str
    experiment_parameters: ExperimentParameters
    report_key: str

    @property
    def is_regression(self) -> bool:
        """
        Checks if the task is a regression task.

            Args:
                None

            Returns:
                bool: True if the task is a regression task, False otherwise.
                      Uses `is_regression_task` to determine this based on the task name.
        """
        return is_regression_task(self.task)

    @property
    def task(self) -> LearningTask:
        """
        Returns the learning task associated with this dataset.

          Args:
            None

          Returns:
            LearningTask: The learning task object.
        """
        return self.dataset.learning_task
