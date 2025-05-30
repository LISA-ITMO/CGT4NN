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
    """
    Generates a string representing the model's name.

        Args:
            dataset_number: The number of the dataset used for training.
            model_type: The type of neural network model used.
            p: The probability value used in the model.
            iteration: The iteration number during training.
            noise_generator: The noise generator used in the model.

        Returns:
            str: A formatted string representing the model's name,
                 following the pattern 'cgtnn-{dataset_number}Y-{model_type.__name__}-g{noise_generator.name}-P{p}_N{iteration}'.
    """
    return f"cgtnn-{dataset_number}Y-{model_type.__name__}-g{noise_generator.name}-P{p}_N{iteration}"


def model_path(
    dataset_number: int,
    model_type: Type[NetworkLike],
    p: float,
    iteration: int,
    noise_generator: NoiseGenerator,
):
    """
    Constructs the file path for a trained model.

        Args:
            dataset_number: The number of the dataset used for training.
            model_type: The type of neural network model.
            p: The probability value used during training.
            iteration: The iteration number of the trained model.
            noise_generator: The noise generator used during training.

        Returns:
            str: The full file path to the saved model, including the '.pth' extension.
    """
    return f"{MODEL_DIR}/{model_name(dataset_number, model_type, p, iteration, noise_generator)}.pth"


def eval_report_key(
    model_name: str,
    dataset_number: int,
    p: float,
    iteration: int,
) -> str:
    """
    Generates a unique key for an evaluation report.

        Args:
            model_name: The name of the model being evaluated.
            dataset_number: The number of the dataset used for evaluation.
            p: The p-value used in the evaluation.
            iteration: The iteration number of the evaluation.

        Returns:
            str: A string representing the unique key for the report.
    """
    return f"evaluate_{model_name}_{dataset_number}_p{p}_N{iteration}"
