from dataclasses import dataclass


@dataclass
class ExperimentParameters:
    iteration: int
    p: float