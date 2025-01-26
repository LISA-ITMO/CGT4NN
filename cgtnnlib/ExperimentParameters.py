from dataclasses import dataclass

from cgtnnlib.constants import ITERATIONS


@dataclass
class ExperimentParameters:
    iteration: int
    p: float


def iterate_experiment_parameters(pp: list[float]):
    for iteration in range(0, ITERATIONS):
        for p in pp:
            yield ExperimentParameters(iteration, p)