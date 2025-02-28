import torch
from typing import Callable

from cgtnnlib.NoiseGenerator import stable_noise
import cgtnnlib.datasets as ds

def add_noise_to_labels_classification(
    labels: torch.Tensor,
    num_classes: int,
    generate_sample: Callable[[], float]
) -> torch.Tensor:
    """
    Добавляет шум к меткам классов для многоклассовой классификации.

    Args:
        labels: Tensor с метками классов (например, [1, 2, 0, 2, 3, 0]).
        num_classes: Количество классов в задаче классификации.
        generate_sample: Функция, которая генерирует случайное число, определяющее
                        величину шума для каждой метки.  Эта функция должна возвращать float.

    Returns:
        Tensor с измененными метками классов.
    """
    noisy_labels = labels.clone()

    for i in range(labels.numel()):
        noise = generate_sample()
        probs = torch.ones(num_classes) * noise / (num_classes - 1)
        probs[labels[i]] = 1 - noise

        probs = probs / probs.sum()


        noisy_labels[i] = torch.multinomial(probs, 1)[0]

    return noisy_labels


ng = stable_noise(
    dataset=ds.datasets[0],
    factor=0.03,
    alpha=1,
    beta=1,
)

for i in range(0, 100):
    print(
        add_noise_to_labels_classification(torch.tensor([1, 2, 0, 2, 3, 0]), 4, ng.next_sample)
    )