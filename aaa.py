from cgtnnlib.constants import LEARNING_RATE, RANDOM_STATE
import cgtnnlib.training as tr
import cgtnnlib.datasets as ds
from cgtnnlib.NoiseGenerator import (
    target_dispersion_scaled_noise,
    stable_noise,
    no_noise_generator,
)

iterations = 10
epochs = 10
pp = [0.0, 0.5, 0.9]
inner_layer_size = 150

# datasets = ds.datasets

datasets = [
    # ds.datasets[0], # 1
    ds.datasets["StudentPerformanceFactors"],  # 3
    # ds.datasets['allhyper'], # 4
    # ds.datasets['wine_quality'], # 6
]

ng_makers = [
    lambda _: no_noise_generator,
    lambda dataset: target_dispersion_scaled_noise(
        dataset=dataset,
        factor=0.03,
        random_seed=RANDOM_STATE + 1,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=1,
        beta=0,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=1.12,
        beta=0,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=2.0,
        beta=1,
    ),
]

import os

from cgtnnlib.Report import Report
from cgtnnlib.nn.AugmentedReLUNetworkMultilayer import AugmentedReLUNetworkMultilayer

for i, dataset in enumerate(datasets):
    for ng_maker in ng_makers:
        for p in pp:
            noise_generator = ng_maker(dataset)
            for iteration in range(iterations):
                os.makedirs(
                    f"rev9/dataset{dataset.number}_p{p}_noise{noise_generator.name}/",
                    exist_ok=True,
                )
                report = Report(
                    dir="rev9",
                    filename=f"dataset{dataset.number}_p{p}_noise{noise_generator.name}/report.json",
                )
                tr.super_train_model(
                    make_model=lambda: AugmentedReLUNetworkMultilayer(
                        inputs_count=dataset.features_count,
                        outputs_count=dataset.classes_count,
                        p=p,
                        inner_layer_size=inner_layer_size,
                        hidden_layers_count=3,
                    ),
                    model_path=f"rev9/dataset{dataset.number}_p{p}_noise{noise_generator.name}/{iteration}.pth",
                    dataset=dataset,
                    report=report,
                    epochs=epochs,
                    learning_rate=LEARNING_RATE,
                    dry_run=False,
                    iteration=iteration,
                    noise_generator=noise_generator,
                )
