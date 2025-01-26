## Path generation, dict key generation, etc. - the Hub of Identity

from cgtnnlib.Dataset import Dataset
from cgtnnlib.NoiseGenerator import NoiseGenerator
from cgtnnlib.nn.NetworkLike import NetworkLike


def model_name(
    dataset_number: int,
    model: NetworkLike,
    p: float,
    iteration: int,
    noise_generator: NoiseGenerator,
):
    return f'cgtnn-{dataset_number}Y-{type(model).__name__}-g{noise_generator.name}-P{p}_N{iteration}'

def model_path(
    dataset_number: int,
    model: NetworkLike,
    p: float,
    iteration: int,
    noise_generator: NoiseGenerator,
):
    return f'pth/{model_name(dataset_number, model, p, iteration, noise_generator)}.pth'


def loss_report_key(
    model: NetworkLike,
    dataset: Dataset,
    p: float,
    iteration: int,
) -> str:
    return f'loss_{type(model).__name__}_{dataset.number}_p{p}_N{iteration}'


def eval_report_key(
    model_name: str,
    dataset_number: int,
    p: float,
    iteration: int,
) -> str:
    return f'evaluate_{model_name}_{dataset_number}_p{p}_N{iteration}'