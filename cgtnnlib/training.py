## Training module v.0.3
## Created at Tue 26 Nov 2024
## Updated at Wed 4 Dec 2024
## v.0.3 - removed train_model_outer()

from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from IPython.display import clear_output

from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.Report import Report
from cgtnnlib.ExperimentParameters import iterate_experiment_parameters
from cgtnnlib.constants import DRY_RUN, EPOCHS, LEARNING_RATE
from cgtnnlib.report_instance import report
from cgtnnlib.torch_device import TORCH_DEVICE

from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork


PRINT_TRAINING_SPAN = 499

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    iteration: int,
    p: float,
    criterion,
    optimizer,
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
            inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

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
                    f'N={iteration} #{dataset.number}',
                    f'p={p}',
                    f'E{epoch}/{epochs}',
                    f'S{total_samples}',
                    f'Loss={loss / 100:.4f}'
                    f'@{model_name}',
                )

            losses.append(loss_item)

    return losses

def create_and_train_model(
    model_path: str,
    dataset: Dataset,
    epochs: int,
    learning_rate: float,
    report: Report,
    dry_run: bool,
    p: float,
    iteration: int,    
):
    model = AugmentedReLUNetwork(
        inputs_count=dataset.features_count,
        outputs_count=dataset.classes_count,
        p=p
    )

    model.apply(init_weights)
    model = model.to(TORCH_DEVICE)

    running_losses: list[float]

    if dry_run:
        print(f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1.0s is generated for running_losses.")
        running_losses = [-1.0 for _ in range(epochs)]
    else:
        running_losses = train_model(
            model=model,
            dataset=dataset,
            epochs=epochs,
            iteration=iteration,
            p=p,
            criterion=dataset.learning_task.criterion,
            optimizer=optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )
        )

    report.record_running_losses(
        running_losses,
        model,
        dataset,
        p,
        iteration,
    )

    torch.save(model.state_dict(), model_path)
    print(f"create_and_train_model(): saved model to {model_path}")


def create_and_train_all_models(
    datasets: list[Dataset],
    epochs: int,
    learning_rate: float,
    report: Report,
    dry_run: bool,
    experiment_params_iter: Iterable[ExperimentParameters]
):
    for experiment_params in experiment_params_iter:
        for dataset in datasets:
            create_and_train_model(
                model_path=dataset.model_b_path(experiment_params),
                dataset=dataset,
                epochs=epochs,
                learning_rate=learning_rate,
                report=report, 
                dry_run=dry_run,
                p=experiment_params.p,
                iteration=experiment_params.iteration, 
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