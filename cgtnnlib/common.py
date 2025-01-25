## COMMON LIBRARY v.0.10
## Created at Sat 23 Nov 2024
## Updated at Wed 15 Jan 2025
## v.0.10 - Explicit datasets and pp parameters for main functions
## v.0.9 - remove DATASETS
## v.0.8 - evaluate_main()
## v.0.7 - training.py
## v.0.6 - even more classes within their own files
## v.0.5 - improved LearningTask interface
## v.0.4 - datasets module
## v.0.3 - more classes within their own files
## v.0.2 - evaluation declarations

## 1.4.-2 Imports

import os

from typing import Any

from IPython.display import clear_output

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error

from cgtnnlib.constants import DRY_RUN, EPOCHS, ITERATIONS, LEARNING_RATE, NOISE_FACTORS, REPORT_DIR
from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.EvaluationParameters import EvaluationParameters
from cgtnnlib.Report import Report, eval_report_key
from cgtnnlib.training import create_and_train_all_models
from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.datasets import datasets
from cgtnnlib.LearningTask import is_classification_task, is_regression_task

## 1.4.-0,5 Initialization

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

report = Report(dir=REPORT_DIR)

def get_pointless_path(filename_without_extension: str) -> str:
    path = os.path.join(REPORT_DIR, f'{filename_without_extension}.png')
    print(f"NOTE: get_pointless_path called but no figure will be saved. Path: {path}")
    plt.close()
    return path

def plot_loss_curve(
    ax, # Axes
    model_name: str,
    dataset_name: str,
    dataset_number: int,
    running_losses: list[float],
    p: float,
    iteration: int,
):
    X = range(len(running_losses))
    
    ax.plot(X, running_losses, label='Running loss', color='orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(f'Loss curve: {model_name} on {dataset_name} (#{dataset_number}), p = {p}, N = {iteration}')
    ax.legend()

def iterate_experiment_parameters(pp: list[float]):
    for iteration in range(0, ITERATIONS):
        for p in pp:
            yield ExperimentParameters(iteration, p)

## 1.4.11 Evaluation

def eval_accuracy_f1_rocauc(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
    noise_factor: float,
    is_binary_classification: bool,
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

    if is_binary_classification:
        np_all_probs = np.array(all_probs)[:, 0]
    else:
        np_all_probs = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, np_all_probs, multi_class='ovr')

    return float(accuracy), float(f1), float(roc_auc)

cool_type = np.ndarray[Any, np.dtype[Any]]

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

def eval_regression_and_report(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
    report_key: str,
)-> pd.DataFrame:
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

    report.append(report_key, samples)

    return pd.DataFrame(samples)

def evaluate_classification_and_report(
    evaluated_model: torch.nn.Module,
    dataset: Dataset,
    report_key: str,
    is_binary_classification: bool,
)-> pd.DataFrame:
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
            is_binary_classification=is_binary_classification,
        )

        samples['accuracy'].append(accuracy)
        samples['f1'].append(f1)
        samples['roc_auc'].append(roc_auc)

    report.append(report_key, samples)

    return pd.DataFrame(samples)

def eval_inner(
    eval_params: EvaluationParameters,
    experiment_params: ExperimentParameters,
    constructor: type,
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
        df = evaluate_classification_and_report(
            evaluated_model=evaluated_model,
            dataset=eval_params.dataset,
            report_key=eval_params.report_key,
            is_binary_classification=eval_params.is_binary_classification
        )
        print('Evaluation of classification (head):')
        print(df.head())
    elif is_regression_task(eval_params.task):
        df = eval_regression_and_report(
            evaluated_model=evaluated_model,
            dataset=eval_params.dataset,
            report_key=eval_params.report_key
        )
        print('Evaluation of regression (head):')
        print(df.head())
    else:
        raise ValueError(f"Unknown task: {eval_params.task}")

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


def train_main(
    pp: list[float],
    datasets: list[Dataset],
):
    create_and_train_all_models(
        datasets=datasets,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        report=report,
        dry_run=DRY_RUN,
        experiment_params_iter=iterate_experiment_parameters(pp)
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

if __name__ == "__main__":
    print('# py')
    print('')
    print('Datasets:')
    for dataset in datasets:
        print(f"{dataset.number}) {dataset.name}: {dataset.features_count} features, {dataset.classes_count} classes")

# if __name__ == '__main__':
#     train_main()
#     evaluate_main()