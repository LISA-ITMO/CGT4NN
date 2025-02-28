from typing import Callable
import torch
import torch.nn as nn

from cgtnnlib.NoiseGenerator import stable_noise
from cgtnnlib.training import add_noise_to_labels_regression

import cgtnnlib.datasets as ds

def do(
    labels: torch.Tensor,
    generate_sample: Callable[[], float]
):
    t = add_noise_to_labels_regression(
        labels,
        generate_sample,
    )
        
    print('t =', t)
    
    n = t.sub(t.min()).div(t.max() - t.min())


    # Alternatively: Use argmax if you're going to use more classes later
    # noisy_targets = torch.argmax(torch.stack([1 - noisy_probabilities, noisy_probabilities], dim=1), dim=1)
    # noisy_targets = noisy_targets.long()

    print(f"Noisy probabilities: {n}")
    print(f"Original targets: {labels}")

    return torch.distributions.Bernoulli(probs=n).sample()



if __name__ == '__main__':
    
    ng = stable_noise(
        dataset=ds.datasets[0],
        factor=0.03,
        alpha=1,
        beta=1,
    )
    for i in range(0, 100):
        baps = do(
            torch.tensor([0., 0., 0., 1., 1., 1, 1., 1., 1., 0., 0., 1.]),
            generate_sample=ng.next_sample
        )
        print(f"baps: {baps}")