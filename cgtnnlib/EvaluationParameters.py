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
    dataset: Dataset
    model_path: str
    experiment_parameters: ExperimentParameters
    report_key: str
    
    @property
    def is_binary_classification(self) -> bool:
        return self.dataset.classes_count == 2
    
    @property
    def is_regression(self) -> bool:
        return is_regression_task(self.task)
    
    @property
    def task(self) -> LearningTask:
        return self.dataset.learning_task