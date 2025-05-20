from dataclasses import dataclass

from cgtnnlib.constants import ITERATIONS


@dataclass
class ExperimentParameters:
    """
    Encapsulates parameters used for running an experiment.

        This class stores configuration values related to the experimental setup,
        such as iteration number and a parameter 'p'.
    """

    iteration: int
    p: float


def iterate_experiment_parameters(pp: list[float]):
    """
    Iterates through experiment parameters.

        Generates a sequence of ExperimentParameters objects, cycling through each
        parameter value in the input list for each iteration up to ITERATIONS.

        Args:
            pp: A list of parameter values to iterate over.

        Returns:
            ExperimentParameters: A generator yielding ExperimentParameters instances.
    """
    for iteration in range(0, ITERATIONS):
        for p in pp:
            yield ExperimentParameters(iteration, p)
