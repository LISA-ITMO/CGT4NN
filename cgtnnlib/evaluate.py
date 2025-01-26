from IPython.display import clear_output

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score

import torch
import torch.nn.functional as F

from cgtnnlib.Dataset import Dataset
from cgtnnlib.EvaluationParameters import EvaluationParameters
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.LearningTask import is_classification_task, is_regression_task
from cgtnnlib.Report import Report, eval_report_key
from cgtnnlib.ExperimentParameters import iterate_experiment_parameters
from cgtnnlib.constants import NOISE_FACTORS
from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork


def eval_accuracy_f1_rocauc(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
    noise_factor: float,
) -> tuple[float, float, float]:
    evaluated_model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataset.data.test_loader:
            outputs = evaluated_model(
                inputs + torch.randn(inputs.shape) * noise_factor
            )
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    if dataset.classes_count == 2:
        # Binary classification
        np_all_probs = np.array(all_probs)[:, 0]
    else:
        # Non-binary classification
        np_all_probs = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, np_all_probs, multi_class='ovr')

    return float(accuracy), float(f1), float(roc_auc)


def eval_r2_mse(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
    noise_factor: float,
) -> tuple[float, float]:
    evaluated_model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataset.data.test_loader:
            noisy_inputs = inputs + torch.randn(inputs.shape) * noise_factor
            outputs = evaluated_model(noisy_inputs)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    np_all_labels = np.array(all_labels)
    np_all_predictions = np.array(all_predictions)

    r2 = r2_score(np_all_labels, np_all_predictions)
    mse = mean_squared_error(np_all_labels, np_all_predictions)

    return float(r2), float(mse)


def eval_regression_over_noise(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
)-> dict:
    samples = {
        'noise_factor': NOISE_FACTORS,
        'r2': [],
        'mse': [],
    }

    for noise_factor in NOISE_FACTORS:
        r2, mse = eval_r2_mse(
            evaluated_model=evaluated_model,
            dataset=dataset,
            noise_factor=noise_factor,
        )

        samples['r2'].append(r2)
        samples['mse'].append(mse)

    return samples


def evaluate_classification_over_noise(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
)-> dict:
    samples = {
        'noise_factor': NOISE_FACTORS,
        'accuracy': [],
        'f1': [],
        'roc_auc': [],
    }

    for noise_factor in NOISE_FACTORS:
        accuracy, f1, roc_auc = eval_accuracy_f1_rocauc(
            evaluated_model=evaluated_model,
            dataset=dataset,
            noise_factor=noise_factor,
        )

        samples['accuracy'].append(accuracy)
        samples['f1'].append(f1)
        samples['roc_auc'].append(roc_auc)

    return samples


def eval_inner(
    eval_params: EvaluationParameters,
    experiment_params: ExperimentParameters,
    constructor: type,
    report: Report, 
):
    evaluated_model = constructor(
        inputs_count=eval_params.dataset.features_count,
        outputs_count=eval_params.dataset.classes_count,
        p=experiment_params.p
    )

    clear_output(wait=True)
    print(f'Evaluating model at {eval_params.model_path}...')
    evaluated_model.load_state_dict(torch.load(eval_params.model_path))

    if is_classification_task(eval_params.task):
        samples = evaluate_classification_over_noise(
            evaluated_model=evaluated_model,
            dataset=eval_params.dataset,
        )
    elif is_regression_task(eval_params.task):
        samples = eval_regression_over_noise(
            evaluated_model=evaluated_model,
            dataset=eval_params.dataset,
        )
    else:
        raise ValueError(f"Unknown task: {eval_params.task}")

    report.append(
        key=eval_params.report_key,
        data=samples
    )


def evaluate(
    experiment_params: ExperimentParameters,
    datasets: list[Dataset]
):
    """
    Валидирует модель `"B"` (`AugmentedReLUNetwork`) согласно параметрам
    эксперимента `experiment_params` на наборах данных из `datasets`.

    - `constructor` может быть `RegularNetwork` или `AugmentedReLUNetwork`
      и должен соответствовать переданному `model_a_or_b`.
    """

    constructor=AugmentedReLUNetwork

    eval_params_items: list[EvaluationParameters] = [EvaluationParameters(
        dataset=dataset,
        model_path=dataset.model_b_path(experiment_params),
        experiment_parameters=experiment_params,
        report_key=eval_report_key(
            model_name=constructor.__name__,
            dataset_number=dataset.number,
            p=experiment_params.p,
            iteration=experiment_params.iteration,
        )
    ) for dataset in datasets]

    for (i, eval_params) in enumerate(eval_params_items):
        eval_inner(
            eval_params,
            experiment_params,
            constructor
        )


def evaluate_main(
    pp: list[float],
    datasets: list[Dataset],
):
    for experiment_params in iterate_experiment_parameters(pp):
        evaluate(
            experiment_params=experiment_params,
            datasets=datasets,
        )