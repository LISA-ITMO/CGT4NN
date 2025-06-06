## Training module v.0.3
## Created at Tue 26 Nov 2024
## Updated at Wed 4 Dec 2024
## v.0.3 - removed train_model_outer()

import math
import os
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from IPython.display import clear_output

from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.LearningTask import Criterion, is_classification_task
from cgtnnlib.NoiseGenerator import NoiseGenerator, no_noise_generator
from cgtnnlib.Report import (
    KEY_DATASET,
    KEY_LOSS,
    KEY_MODEL,
    KEY_TRAIN_NOISE_GENERATOR,
    KEY_EPOCHS,
    Report,
)
from cgtnnlib.ExperimentParameters import iterate_experiment_parameters
from cgtnnlib.constants import DRY_RUN, EPOCHS, LEARNING_RATE, MODEL_DIR
from cgtnnlib.nn.AugmentedReLUNetworkMultilayer import AugmentedReLUNetworkMultilayer
from cgtnnlib.torch_extras import TORCH_DEVICE

from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork

import cgtnnlib.path


PRINT_TRAINING_SPAN = 499


def init_weights_xavier(m: nn.Module):
    "Usage: `model.apply(init_weights_xavier)`"
    # XXX For ReLU (as opposed to tanh and sigmoid)
    # XXX He initialization is more appropriate
    # XXX
    # XXX <https://arxiv.org/abs/1502.01852>
    # XXX     Delving Deep into Rectifiers: Surpassing
    # XXX     Human-Level Performance on ImageNet Classification
    # XXX
    # XXX     Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def add_noise_to_labels_regression(
    labels: torch.Tensor,
    generate_sample: Callable[[], float],
) -> torch.Tensor:
    """
    Adds noise to regression labels.

        This function adds random noise generated by a provided callable to each element of the input tensor.  The noise is generated independently for each label using the `generate_sample` function.

        Args:
            labels: The original regression labels as a PyTorch tensor.
            generate_sample: A callable that takes no arguments and returns a single float representing the noise to be added.

        Returns:
            torch.Tensor: A new tensor with the added noise, having the same shape as the input `labels` tensor.
    """
    return labels + torch.tensor(
        [generate_sample() for _ in range(labels.numel())]
    ).reshape(labels.shape)


def add_noise_to_labels_classification(
    labels: torch.Tensor, generate_sample: Callable[[], float]
) -> torch.Tensor:
    """
    Adds noise to classification labels.

        This function currently returns the original labels without modification.
        The intended functionality (adding noise based on a regression approach) is
        currently commented out due to issues.

        Args:
            labels: The tensor of classification labels.
            generate_sample: A callable that generates a random sample for noise injection.

        Returns:
            torch.Tensor: The potentially noisy labels (currently returns the original labels).
    """
    return labels
    # XXX Это всё не работает :с
    # t = add_noise_to_labels_regression(
    #     labels,
    #     generate_sample,
    # )
    # a = t.max() - t.min()
    # if a == 0:
    #     # Special case: all classes in labels are same
    #     print("WARNING: a == 0")
    #     return labels
    # n = t.sub(t.min()).div(a)
    # return torch.distributions.Bernoulli(probs=n).sample().type(torch.int64)


def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    iteration: int,
    p: float,
    criterion: Criterion,
    optimizer,
    noise_generator: NoiseGenerator = no_noise_generator,
) -> list[float]:
    """
    Trains a model on the given dataset.

        Args:
            model: The neural network model to train.
            dataset: The dataset used for training.
            epochs: The number of epochs to train for.
            iteration: An identifier for the current iteration (used for printing).
            p: A parameter value (used for printing).
            criterion: The loss function to use during training.
            optimizer: The optimizer to use for updating model parameters.
            noise_generator: An optional noise generator to add noise to labels.

        Returns:
            list[float]: A list of loss values recorded during training.
    """
    # XXX: p and iteration are only used for printing output; very awkward

    losses: list[float] = []
    total_samples = len(dataset.data.train_loader)
    model_name = type(model).__name__

    for epoch in range(epochs):
        model.train()

        inputs: torch.Tensor
        outputs: torch.Tensor
        labels: torch.Tensor

        for i, (inputs, labels) in enumerate(dataset.data.train_loader):
            if is_classification_task(dataset.learning_task):
                pass
                # XXX
                # labels = add_noise_to_labels_classification(
                #     labels=labels,
                #     generate_sample=noise_generator.next_sample,
                # )
            else:
                labels = add_noise_to_labels_regression(
                    labels=labels,
                    generate_sample=noise_generator.next_sample,
                ).to(torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(TORCH_DEVICE)
            # if is_classification_task(dataset.learning_task):
            #     loss = criterion(outputs.softmax(dim=0), labels.softmax(dim=0))
            # else:
            # if is_classification_task(dataset.learning_task):
            #     loss = criterion(torch.argmax().softmax(dim=0), labels.softmax(dim=0))
            # else:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()

            if math.isnan(loss_item):
                raise RuntimeError("nan loss!")

            if i % PRINT_TRAINING_SPAN == 0:
                clear_output(wait=True)
                print(
                    f"N={iteration}",
                    f"#{dataset.number}",
                    f"g{noise_generator.name}",
                    f"p={p}",
                    f"E{epoch}/{epochs}",
                    f"S{total_samples}",
                    f"Loss={loss / 100:.4f}" f"@{model_name}",
                )

            losses.append(loss_item)

    return losses


def create_and_train_model(
    dataset: Dataset,
    epochs: int,
    learning_rate: float,
    dry_run: bool,
    p: float,
    iteration: int,
    noise_generator: NoiseGenerator,
    must_not_exist: bool = False,
):
    """
    Creates and trains an augmented ReLU network model.

        Args:
            dataset: The dataset to use for training.
            epochs: The number of epochs to train the model for.
            learning_rate: The learning rate for the optimizer.
            dry_run: Whether to perform a dry run (no weight updates).
            p: The probability value used in the model and file naming.
            iteration: The iteration number, used in file naming.
            noise_generator: The noise generator to use.
            must_not_exist: If True, skips training if the model file already exists.

        Returns:
            None.  Saves the trained model to a file and saves a report of the training process.
    """
    model_type = AugmentedReLUNetwork
    model_path = cgtnnlib.path.model_path(
        dataset_number=dataset.number,
        model_type=model_type,
        p=p,
        iteration=iteration,
        noise_generator=noise_generator,
    )

    if must_not_exist and os.path.exists(model_path):
        print(f"File already exists at {model_path}. Skipping training.")
        return

    model = model_type(
        inputs_count=dataset.features_count,
        outputs_count=dataset.classes_count,
        p=p,
        softmax_output=is_classification_task(dataset.learning_task),
    )

    model.apply(init_weights_xavier)
    model = model.to(TORCH_DEVICE)

    report_name = cgtnnlib.path.model_name(
        dataset_number=dataset.number,
        model_type=type(model),
        p=p,
        iteration=iteration,
        noise_generator=noise_generator,
    )

    report = Report(dir=MODEL_DIR, filename=report_name + ".json")

    losses: list[float]

    if dry_run:
        print(
            f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1.0s is generated for running_losses."
        )
        losses = [-1.0 for _ in range(epochs)]
    else:
        losses = train_model(
            model=model,
            dataset=dataset,
            epochs=epochs,
            iteration=iteration,
            p=p,
            criterion=dataset.learning_task.criterion,
            optimizer=optim.Adam(
                model.parameters(),
                lr=learning_rate,
            ),
            noise_generator=noise_generator,
        )

    report.set(
        KEY_MODEL,
        {
            "class": model.__class__.__name__,
            "p": p,
        },
    )
    report.set(KEY_DATASET, dataset.to_dict())
    report.set(KEY_TRAIN_NOISE_GENERATOR, {"name": noise_generator.name})
    report.set(KEY_EPOCHS, epochs)
    report.set(KEY_LOSS, losses)

    torch.save(model.state_dict(), model_path)
    print(f"create_and_train_model(): saved model to {model_path}")

    report.save()


def super_train_model(
    make_model: Callable[[], AugmentedReLUNetworkMultilayer],
    model_path: str,
    dataset: Dataset,
    report: Report,
    epochs: int,
    learning_rate: float,
    dry_run: bool,
    iteration: int,
    noise_generator: NoiseGenerator,
):
    """Ultimatge Trainer"""

    must_not_exist = True

    if must_not_exist and os.path.exists(model_path):
        print(f"File already exists at {model_path}. Skipping training.")
        return

    model = make_model()

    model.apply(init_weights_xavier)
    model = model.to(TORCH_DEVICE)

    losses: list[float]

    if dry_run:
        print(
            f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1.0s is generated for running_losses."
        )
        losses = [-1.0 for _ in range(epochs)]
    else:
        losses = train_model(
            model=model,
            dataset=dataset,
            epochs=epochs,
            iteration=iteration,
            p=model.p,
            criterion=dataset.learning_task.criterion,
            optimizer=optim.Adam(
                model.parameters(),
                lr=learning_rate,
            ),
            noise_generator=noise_generator,
        )

    report.set(
        KEY_MODEL,
        {
            "class": model.__class__.__name__,
            "p": model.p,
            "inner_layer_size": model.inner_layer_size,
            "hidden_layers_count": model.hidden_layers_count,
            "iteration": iteration,
        },
    )

    report.set(KEY_DATASET, dataset.to_dict())
    report.set(KEY_TRAIN_NOISE_GENERATOR, {"name": noise_generator.name})
    report.set(KEY_EPOCHS, epochs)
    report.set(KEY_LOSS, losses)

    torch.save(model.state_dict(), model_path)
    print(f"create_and_train_model(): saved model to {model_path}")

    report.save()


def create_and_train_all_models(
    datasets: list[Dataset],
    epochs: int,
    learning_rate: float,
    dry_run: bool,
    experiment_params_iter: Iterable[ExperimentParameters],
):
    """
    Iterates through datasets and experiment parameters to train models.

        This method loops through a list of datasets and a sequence of experiment
        parameters, calling `create_and_train_model` for each combination.

        Args:
            datasets: The list of datasets to use for training.
            epochs: The number of epochs to train each model for.
            learning_rate: The learning rate to use during training.
            dry_run: Whether to perform a dry run (e.g., logging without actual training).
            experiment_params_iter: An iterable yielding ExperimentParameters
                objects, which contain parameters specific to each experiment.

        Returns:
            None
    """
    for experiment_params in experiment_params_iter:
        for dataset in datasets:
            create_and_train_model(
                dataset=dataset,
                epochs=epochs,
                learning_rate=learning_rate,
                dry_run=dry_run,
                p=experiment_params.p,
                iteration=experiment_params.iteration,
                noise_generator=no_noise_generator,
            )


def train_main(
    pp: list[float],
    datasets: list[Dataset],
):
    """
    Trains all models with the given datasets and experiment parameters.

        Args:
            pp: A list of parameter values to iterate through for experiments.
            datasets: A list of Dataset objects used for training.

        Returns:
            None: This function does not return a value; it trains models as a side effect.
    """
    create_and_train_all_models(
        datasets=datasets,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        dry_run=DRY_RUN,
        experiment_params_iter=iterate_experiment_parameters(pp),
    )
