from dataclasses import dataclass
from typing import Callable

import numpy as np

from cgtnnlib.Dataset import Dataset

@dataclass
class NoiseGenerator:
    name: str
    next_sample: Callable[[], float]

no_noise_generator = NoiseGenerator(
    name="NoNoise",
    next_sample=lambda: 0,
)

def target_dispersion_scaled_noise(
    dataset: Dataset,
    factor: float,
    random_seed: int,
) -> NoiseGenerator:
    scale = factor * (dataset.data.df[dataset.target].std() ** 2)
    rng = np.random.default_rng(random_seed)
    
    return NoiseGenerator(
        name=f"TDS{dataset.number}",
        next_sample=lambda: rng.uniform(-1.0, 1.0) * scale,
    )