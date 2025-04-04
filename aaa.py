from cgtnnlib.constants import LEARNING_RATE, RANDOM_STATE
import cgtnnlib.training as tr
import cgtnnlib.datasets as ds
from cgtnnlib.NoiseGenerator import target_dispersion_scaled_noise, stable_noise, no_noise_generator


iterations = 10
epochs = 10
pp = [0.0, 0.5, 0.9]

double_epochs_datasets = [
    ds.datasets[0], # 1
    ds.datasets[1], # 2
    ds.datasets[4], # 5
    ds.datasets[6], # 7
    ds.datasets[7], # 8
]

# datasets = ds.datasets

datasets = [
    ds.datasets[0], # 1
    ds.datasets['StudentPerformanceFactors'], # 3
    ds.datasets['allhyper'], # 4
    ds.datasets['wine_quality'], # 6
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
        beta=1,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=1.12,
        beta=1,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=1.25,
        beta=1,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=1.5,
        beta=1,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=1.75,
        beta=1,
    ),
    lambda dataset: stable_noise(
        dataset=dataset,
        factor=0.03,
        alpha=2.0,
        beta=1,
    ),
]

## Analysis

import json

import pandas as pd

import matplotlib.pyplot as plt

from cgtnnlib.LearningTask import is_classification_task
from cgtnnlib.analyze import plot_deviant_curves_on_ax_or_plt
from cgtnnlib.constants import NOISE_FACTORS
from cgtnnlib.evaluate import eval_report_at_path
from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork



def read_json(path: str) -> dict:
    with open(path) as file:
        return json.load(file)

def summarize_series_list(series_list: list[pd.Series]):
    df = pd.DataFrame(series_list).T

    summary_df = pd.DataFrame({
        0.25: df.quantile(0.25, axis=1),
        0.75: df.quantile(0.75, axis=1),
        'mean': df.mean(axis=1),
    })

    return summary_df


def make_ax_drawer(
    read_json,
    dataset,
    ng_maker,
    p,
):
    noise_generator = ng_maker(dataset)
    prefix = (
        f'cgtnn-{dataset.number}Y-AugmentedReLUNetwork'
        +f'-g{noise_generator.name}-P{p}_'
    )

    def report_path(n):
        return f'pth/{prefix}N{n}.json'

    def model_path(n):
        return f'pth/{prefix}N{n}.pth'

    def read_eval_from_iteration(n) -> pd.DataFrame:
        path = report_path(n)
        eval_report_at_path(
                    report_path=path,
                    model_path=model_path(n),
                    constructor=AugmentedReLUNetwork,
                    dataset=dataset,
                    p=p,
                )
        print('read_eval_from_iteration', path, n)
        return pd.DataFrame(read_json(path)['eval'])
    
    def read_loss_from_iteration(n) -> pd.DataFrame:
        path = report_path(n)
        json = read_json(path)
        return pd.DataFrame({ 'loss': json['loss'] })

    metric = 'loss'

    files = [
        read_loss_from_iteration(n)
        for n in range(iterations)
    ]
            
    print(report_path(0))

    curve = summarize_series_list([file[metric] for file in files])

    def draw_ax(ax):
        plot_deviant_curves_on_ax_or_plt(
            ax_or_plt=ax,
            models=[{
                'curve': curve,
                'color': 'purple',
                'label': 'Среднее',
                'quantiles_color': 'pink',
                'quantiles_label': 'Квартили 0,25; 0,75', 
            }],
            title='\n'.join([
                f'{noise_generator.name}, p = {p}',
            ]),
            xlabel='Шум на входе',
            ylabel=metric,
            quantiles_alpha=0.5,
        )
    
    return draw_ax

ax_drawers = [
    [
        [make_ax_drawer(read_json, dataset, ng_maker, p) for p in pp]
        for ng_maker in ng_makers
    ]
    for dataset in datasets
]

print(ax_drawers)