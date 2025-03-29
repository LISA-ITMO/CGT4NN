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
from cgtnnlib.torch_device import TORCH_DEVICE

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
    return labels + torch.tensor(
        [generate_sample() for _ in range(labels.numel())]
    ).reshape(labels.shape)


def add_noise_to_labels_classification(
    labels: torch.Tensor,
    generate_sample: Callable[[], float]
) -> torch.Tensor:
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
                )

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(TORCH_DEVICE)
            # if is_classification_task(dataset.learning_task):
            #     loss = criterion(outputs.softmax(dim=0), labels.softmax(dim=0))
            # else:
            # if is_classification_task(dataset.learning_task):
            #     loss = criterion(torch.argmax().softmax(dim=0), labels.softmax(dim=0))
            # else:
            loss = criterion(
                outputs,
                labels.long() if is_classification_task(dataset.learning_task) else labels
            )
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            
            if math.isnan(loss_item):
                raise RuntimeError('nan loss!')
            
            if i % PRINT_TRAINING_SPAN == 0:
                clear_output(wait=True)
                print(
                    f'N={iteration}',
                    f'#{dataset.number}',
                    f'g{noise_generator.name}',
                    f'p={p}',
                    f'E{epoch}/{epochs}',
                    f'S{total_samples}',
                    f'Loss={loss / 100:.4f}'
                    f'@{model_name}',
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
    model_type = AugmentedReLUNetwork
    model_path = cgtnnlib.path.model_path(
        dataset_number=dataset.number,
        model_type=model_type,
        p=p,
        iteration=iteration,
        noise_generator=noise_generator
    )

    if must_not_exist and os.path.exists(model_path):
        print(f'File already exists at {model_path}. Skipping training.')
        return

    model = model_type(
        inputs_count=dataset.features_count,
        outputs_count=dataset.classes_count,
        p=p,
        softmax_output=is_classification_task(dataset.learning_task)
    )

    model.apply(init_weights_xavier)
    model = model.to(TORCH_DEVICE)
    
    report_name = cgtnnlib.path.model_name(
        dataset_number=dataset.number,
        model_type=type(model),
        p=p,
        iteration=iteration,
        noise_generator=noise_generator
    )
 
    report = Report(
        dir=MODEL_DIR,
        filename=report_name + '.json'
    )

    losses: list[float]

    if dry_run:
        print(f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1.0s is generated for running_losses.")
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


    report.set(KEY_MODEL, {
        "class": model.__class__.__name__,
        "p": p,
    })
    report.set(KEY_DATASET, dataset.to_dict())
    report.set(KEY_TRAIN_NOISE_GENERATOR, { "name": noise_generator.name })
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
    experiment_params_iter: Iterable[ExperimentParameters]
):
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
    create_and_train_all_models(
        datasets=datasets,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        dry_run=DRY_RUN,
        experiment_params_iter=iterate_experiment_parameters(pp)
    )