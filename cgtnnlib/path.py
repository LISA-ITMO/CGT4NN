## Path generation, dict key generation, etc. - the Hub of Identity
from typing import Type

from cgtnnlib.NoiseGenerator import NoiseGenerator
from cgtnnlib.constants import MODEL_DIR
from cgtnnlib.nn.NetworkLike import NetworkLike


def model_name(
    dataset_number: int,
    model_type: Type[NetworkLike],
    p: float,
    iteration: int,
    noise_generator: NoiseGenerator,
):
    return f'cgtnn-{dataset_number}Y-{model_type.__name__}-g{noise_generator.name}-P{p}_N{iteration}'


def model_path(
    dataset_number: int,
    model_type: Type[NetworkLike],
    p: float,
    iteration: int,
    noise_generator: NoiseGenerator,
):
    return f'{MODEL_DIR}/{model_name(dataset_number, model_type, p, iteration, noise_generator)}.pth'


def eval_report_key(
    model_name: str,
    dataset_number: int,
    p: float,
    iteration: int,
) -> str:
    return f'evaluate_{model_name}_{dataset_number}_p{p}_N{iteration}'