from cgtnnlib.constants import LEARNING_RATE, RANDOM_STATE
import cgtnnlib.training as tr
import cgtnnlib.datasets as ds
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.NoiseGenerator import target_dispersion_scaled_noise


iterations = 10
epochs = 10
dataset = ds.datasets[2]
p = 0.9
for iteration in range(iterations):
    experiment_params = ExperimentParameters(iteration, p=p)

    tr.create_and_train_model(
        dataset,
        epochs,
        learning_rate=LEARNING_RATE,
        dry_run=False,
        p=p,
        iteration=iteration,
        noise_generator=target_dispersion_scaled_noise(
            dataset=dataset,
            factor=0.03,
            random_seed=RANDOM_STATE + 1,
        )
    )