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
from cgtnnlib.torch_device import TORCH_DEVICE

from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork


PRINT_TRAINING_SPAN = 500

def print_progress(
    p: float,
    iteration: int,
    epoch: int,
    total_epochs: int,
    total_samples: int,
    running_loss: float,
    dataset_number: int,
    model_name: str,
):
    clear_output(wait=True)
    print(
        f'N={iteration} #{dataset_number} p={p} E{epoch}/{total_epochs} S{total_samples} Loss={running_loss / 100:.4f} @{model_name}'
    )

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    experiment_params: ExperimentParameters,
    criterion,
    optimizer,
) -> list[float]:
    model.apply(init_weights)
    model = model.to(TORCH_DEVICE)

    running_losses: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0

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
            running_loss += loss.item()

            if i % PRINT_TRAINING_SPAN == 0:
                print_progress(
                    p=experiment_params.p,
                    iteration=experiment_params.iteration,
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    total_samples=len(dataset.data.train_loader),
                    running_loss=running_loss,
                    dataset_number=dataset.number,
                    model_name=type(model).__name__,
                )

            running_losses.append(running_loss)
            running_loss = 0.0

    return running_losses

def create_and_train_all_models(
    datasets: list[Dataset],
    epochs: int,
    learning_rate: float,
    report: Report,
    dry_run: bool,
    experiment_params_iter: Iterable[ExperimentParameters]
):
    for experiment_params in experiment_params_iter:
        # fig, axs = plt.subplots(2, 3, sharey='col', figsize=(10, 12))
        # fig.set_size_inches(35, 20)

        for dataset in datasets:
            inputs_count = dataset.features_count
            outputs_count = dataset.classes_count

            for (model, path) in [
                ## Uncomment to train RegularNetwork
                # (RegularNetwork(
                #     inputs_count=inputs_count,
                #     outputs_count=outputs_count,
                #     p=experiment_params.p
                # ), training_params.model_a_path),
                (AugmentedReLUNetwork(
                    inputs_count=inputs_count,
                    outputs_count=outputs_count,
                    p=experiment_params.p
                ), dataset.model_b_path(experiment_params))
            ]:
                running_losses: list[float]

                if dry_run:
                    print(f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1s is generated for running_losses.")
                    running_losses = [-1.0 for _ in range(epochs)]
                else:
                    running_losses = train_model(
                        model=model,
                        dataset=dataset,
                        epochs=epochs,
                        experiment_params=experiment_params,
                        criterion=dataset.learning_task.criterion,
                        optimizer=optim.Adam(
                            model.parameters(),
                            lr=learning_rate,
                        )
                    )
                
                save_model_to_path(model, path)

                report.record_running_losses(running_losses,
                                             model,
                                             dataset,
                                             experiment_params)
                

def save_model_to_path(
    model,
    path,
):
    print(f"train_model_outer(): saved model to {path}")
    torch.save(model.state_dict(), path)