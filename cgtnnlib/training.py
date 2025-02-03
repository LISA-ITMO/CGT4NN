## Training module v.0.3
## Created at Tue 26 Nov 2024
## Updated at Wed 4 Dec 2024
## v.0.3 - removed train_model_outer()

from typing import Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from IPython.display import clear_output

from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.NoiseGenerator import NoiseGenerator, no_noise_generator
from cgtnnlib.Report import KEY_LOSS, Report
from cgtnnlib.ExperimentParameters import iterate_experiment_parameters
from cgtnnlib.constants import DRY_RUN, EPOCHS, LEARNING_RATE
from cgtnnlib.torch_device import TORCH_DEVICE

from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork

import cgtnnlib.path


PRINT_TRAINING_SPAN = 499

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def add_noise_to_tensor(
    input: torch.Tensor,
    generate_sample: Callable[[], float],
) -> torch.Tensor:
    return input + torch.tensor(
        [generate_sample() for _ in range(input.numel())],
        dtype=input.dtype
    ).reshape(input.shape)


def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    iteration: int,
    p: float,
    criterion,
    optimizer,
    noise_generator: NoiseGenerator,
) -> list[float]:
    losses: list[float] = []
    total_samples = len(dataset.data.train_loader)
    model_name = type(model).__name__

    for epoch in range(epochs):
        model.train()

        inputs: torch.Tensor
        outputs: torch.Tensor
        labels: torch.Tensor

        for i, (inputs, labels) in enumerate(dataset.data.train_loader):
            inputs = inputs.to(TORCH_DEVICE)
            labels = add_noise_to_tensor(
                input=labels,
                generate_sample=noise_generator.next_sample,
            ).to(TORCH_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(TORCH_DEVICE)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            
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
    noise_generator: NoiseGenerator
):
    model = AugmentedReLUNetwork(
        inputs_count=dataset.features_count,
        outputs_count=dataset.classes_count,
        p=p
    )

    model_path = cgtnnlib.path.model_path(
        dataset_number=dataset.number,
        model_type=type(model),
        p=p,
        iteration=iteration,
        noise_generator=noise_generator
    )

    model.apply(init_weights)
    model = model.to(TORCH_DEVICE)
    
    report_name = cgtnnlib.path.model_name(
        dataset_number=dataset.number,
        model_type=type(model),
        p=p,
        iteration=iteration,
        noise_generator=noise_generator
    )
 
    report = Report(
        dir='pth/',
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