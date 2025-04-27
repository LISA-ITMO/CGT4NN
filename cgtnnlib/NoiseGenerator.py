from dataclasses import dataclass
from typing import Callable

import numpy as np

from cgtnnlib.Dataset import Dataset


def stable_noise_func(alpha, beta, size=1):
  """
  Generates stable noise samples using the Chambers-Mallows-Stuck (CMS) algorithm.

  Args:
    alpha: Stability parameter (0 < alpha <= 2).  alpha=2 corresponds to Gaussian. alpha=1 Cauchy.
    beta: Skewness parameter (-1 <= beta <= 1). beta=0 is symmetric.
    size: Number of samples to generate.

  Returns:
    A numpy array of stable noise samples.

  Raises:
    ValueError: If alpha or beta are outside the allowed ranges.
  """

  if not 0 < alpha <= 2:
    raise ValueError("alpha must be in the range (0, 2]")
  if not -1 <= beta <= 1:
    raise ValueError("beta must be in the range [-1, 1]")

  U = np.random.uniform(-np.pi/2, np.pi/2, size=size)
  E = np.random.exponential(1, size=size)

  if alpha != 1:
    term1 = np.sin(alpha * (U + beta * np.pi / 2))
    term2 = np.cos(U)**(-1/alpha)
    term3 = np.cos(U - alpha * (U + beta * np.pi / 2))
    
    # Cast to complex to avoid nans due to exponentiation
    X = term1 * term2 * abs(
      np.cfloat(E / term3)
      ** ((1-alpha)/alpha)
    )

  else:
      X = (2/np.pi) * ( ( (np.pi/2) + beta*U )*np.tan(U) - beta*np.log( (E * np.cos(U) ) / ((np.pi/2)+ beta*U) )  )

  return X


@dataclass
class NoiseGenerator:
    name: str
    description: str
    next_sample: Callable[[], float]

no_noise_generator = NoiseGenerator(
    name="NoNoise",
    description="Нет шума",
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
        description=f"Белый шум с амплитудой {factor} от дисперсии целевой переменной",
        next_sample=lambda: rng.uniform(-1.0, 1.0) * scale,
    )

def stable_noise(
    dataset: Dataset,
    factor: float,
    alpha: float,
    beta: float,
) -> NoiseGenerator:
    scale = factor * (dataset.data.df[dataset.target].std() ** 2)
    
    return NoiseGenerator(
        name=f"Stable{dataset.number}A{alpha}B{beta}F{factor}",
        description=f"Стабильный {factor} от дисперсии целевой переменной, ɑ = {alpha}, β = {beta}",
        next_sample=lambda: stable_noise_func(alpha, beta, 1)[0] * scale
    )