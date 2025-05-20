from cgtnnlib.LearningTask import LearningTask
from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.training import train_model
from cgtnnlib.datasets import datasets

import torch.optim as optim

p = 0.5
dataset = datasets[0]
model = AugmentedReLUNetwork(
    inputs_count=dataset.features_count, outputs_count=dataset.classes_count, p=p
)

train_model(
    model=model,
    dataset=dataset,
    epochs=1,
    iteration=1,
    p=p,
    criterion=dataset.learning_task.criterion,
    optimizer=optim.Adam(
        model.parameters(),
        lr=0.05,
    ),
)
