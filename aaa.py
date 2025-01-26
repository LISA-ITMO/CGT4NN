
from cgtnnlib.EvaluationParameters import EvaluationParameters
from cgtnnlib.Report import Report
from cgtnnlib.constants import LEARNING_RATE, RANDOM_STATE
from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork
import cgtnnlib.evaluate as ev
import cgtnnlib.datasets as ds
import cgtnnlib.path as ph
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.NoiseGenerator import target_dispersion_scaled_noise


iterations = 10
epochs = 10
dataset = ds.datasets[2]
p = 0.9
for iteration in range(iterations):
    experiment_params = ExperimentParameters(iteration, p=p)
    noise_generator = target_dispersion_scaled_noise(
        dataset=dataset,
        factor=0.03,
        random_seed=RANDOM_STATE + 1,
    )
    
    eval_params = EvaluationParameters(
        dataset=dataset,
        model_path=ph.model_path(
            dataset_number=dataset.number,
            model_type=AugmentedReLUNetwork,
            p=p,
            iteration=iteration,
            noise_generator=noise_generator,
        ),
        experiment_parameters=experiment_params,
        report_key=ph.eval_report_key(
            model_name=AugmentedReLUNetwork.__name__,
            dataset_number=dataset.number,
            p=p,
            iteration=iteration,
        )
    )
    
    report_name = ph.model_name(
        dataset_number=dataset.number,
        model_type=AugmentedReLUNetwork,
        p=p,
        iteration=iteration,
        noise_generator=noise_generator
    )
    report = Report(
        dir='pth/',
        filename=report_name + '.json',
        must_exist=True,
    )
    
    ev.eval_inner(
        eval_params=eval_params,
        constructor=AugmentedReLUNetwork,
        report=report
    )
    
    report.save()

    # tr.create_and_train_model(
    #     dataset,
    #     epochs,
    #     learning_rate=LEARNING_RATE,
    #     dry_run=False,
    #     p=p,
    #     iteration=iteration,
    #     noise_generator=target_dispersion_scaled_noise(
    #         dataset=dataset,
    #         factor=0.03,
    #         random_seed=RANDOM_STATE + 1,
    #     )
    # )