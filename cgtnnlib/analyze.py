## Result data analysis routines v.0.3
## Created at Thu 28 Nov 2024
## Updated at Wed 15 Jan 2025
## v.0.3 support for more datasets
## v.0.2 search_plot_data raises IndexError on failed search

import os

from typing import Any, TypeAlias, TypedDict

import matplotlib.pyplot as plt
import pandas as pd

from cgtnnlib.Dataset import Dataset
from cgtnnlib.PlotModel import PlotModel, Measurement, Metric
from cgtnnlib.Report import make_search_index, search_plot_data, load_raw_report, SearchIndex, RawReport
from cgtnnlib.plt_extras import set_title, set_xlabel, set_ylabel

def df_head_fraction(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    df_head_fraction(df, frac=0.15)
    """
    n_rows = int(len(df) * frac)
    return df.head(n_rows)

Color: TypeAlias = str 
"Like 'lightblue', 'lightgray', etc"

class DeviantCurvePlotModel(TypedDict):
    curve: pd.DataFrame
    color: Color
    label: str
    quantiles_color: Color
    quantiles_label: str
    pass

# 3. Вывод графиков
def plot_deviant_curves_on_ax_or_plt(
    ax_or_plt,
    models: list[DeviantCurvePlotModel],
    title: str,
    xlabel: str,
    ylabel: str,
    quantiles_alpha: float,
):
    if len(models) == 0:
        raise TypeError("models should not be empty")
    
    X: pd.Index = models[0]['curve'].index
    
    for model in models:
        expected_columns = [0.25, 0.75, 'mean']
        columns = model['curve'].columns.to_list()
        
        assert columns == expected_columns, f"Bad value of curve_df.columns: should be {expected_columns}, instead got {columns}"

        ax_or_plt.plot(
            X,
            model['curve']['mean'],
            label=model['label'],
            color=model['color'],
        )

        ax_or_plt.fill_between(
            X,
            model['curve'][0.25],
            model['curve'][0.75],
            color=model['quantiles_color'],
            alpha=quantiles_alpha,
            label=model['quantiles_label'],
        )

    set_xlabel(ax_or_plt, xlabel)
    set_ylabel(ax_or_plt, ylabel)
    set_title(ax_or_plt, title)

    ax_or_plt.legend()

def extract_values_from_search_results(
    search_results: pd.DataFrame,
    raw_report: dict[str, Any],
    measurement: Measurement,
    metric: str,
):
    if measurement == 'loss':
        return pd.DataFrame([raw_report[str(row.Key)] for row in search_results.itertuples()])
    else:
        cols = []
        
        for row in search_results.itertuples():
            report_data: dict = raw_report[str(row.Key)]

            cols.append(report_data[metric])
            
        return pd.DataFrame(cols)

class AnalysisParams(TypedDict):
    measurement: Measurement
    dataset_number: int
    xlabel: str
    frac: float
    metrics: list[Metric]


def search_curve(
    search_index: SearchIndex,
    plot_params: PlotModel,
    raw_report: RawReport,
) -> pd.DataFrame:
    values = extract_values_from_search_results(
        search_results=search_plot_data(
            search_index=search_index,
            plot_params=plot_params,
        ),
        raw_report=raw_report,
        measurement=plot_params.measurement,
        metric=plot_params.metric,
    )
    result = values.quantile([0.25, 0.75]).transpose()
    result['mean'] = values.mean()

    return df_head_fraction(
        df=result,
        frac=plot_params.frac
    )    

def plot_analysis_fig(
    search_index: SearchIndex,
    raw_report: RawReport,
    analysis_params_list: list[AnalysisParams],
    pp: list[float],
) -> None:
    for analysis_params in analysis_params_list:
        measurement = analysis_params['measurement']
        dataset_number = analysis_params['dataset_number']
        xlabel = analysis_params['xlabel']
        frac = analysis_params['frac']
        metrics = analysis_params['metrics']

        # Plot grid
        nrows = len(metrics)
        ncols = len(pp)
        fig, axs = plt.subplots(nrows, ncols, figsize=(24, nrows * ncols))

        for (i, metric) in enumerate(metrics):
            def make_curve_for_p(p: float) -> pd.DataFrame:
                return search_curve(
                    search_index=search_index,
                    plot_params=PlotModel(
                        measurement=measurement,
                        dataset_number=dataset_number,
                        model_name='AugmentedReLUNetwork',
                        metric=metric,
                        p=p,
                        frac=frac,
                    ),
                    raw_report=raw_report
                )

            reference_curve: pd.DataFrame = make_curve_for_p(0)

            for (j, p) in enumerate(pp):
                plot_deviant_curves_on_ax_or_plt(
                    ax_or_plt=axs[i, j] if len(metrics) > 1 else axs[j],
                    models=[{
                        'curve': reference_curve,
                        'color': 'lightblue',
                        'label': 'Mean of p = 0',
                        'quantiles_color': 'lightgray',
                        'quantiles_label': '0.25 to 0.75 Quantiles',
                    }, {
                        'curve': make_curve_for_p(p),
                        'color': 'blue',
                        'label': f'Mean of p = {p}',
                        'quantiles_color': 'gray',
                        'quantiles_label': '0.25 to 0.75 Quantiles',
                    },],
                    title=f'p = {p}',
                    xlabel=xlabel,
                    ylabel=metric,
                    quantiles_alpha=0.1,
                )
        fig.suptitle(f'Dataset #{dataset_number}, zoom factor: {frac}')
        plt.tight_layout()
        path = os.path.join('report/', f'{measurement}_{dataset_number}_f{frac:.02f}.png')
        plt.savefig(path)
        plt.close()

# default_analysis_params_list: list[AnalysisParams] = [
#     {'measurement': 'loss', 'dataset_number': 1,
#      'xlabel': 'iteration', 'frac': 1 },
#     {'measurement': 'loss', 'dataset_number': 1,
#      'xlabel': 'iteration', 'frac': 0.02},
#     {'measurement': 'loss', 'dataset_number': 1,
#      'xlabel': 'iteration', 'frac': 0.1},
#     {'measurement': 'loss', 'dataset_number': 1,
#      'xlabel': 'iteration', 'frac': 0.2},
#     {'measurement': 'evaluate', 'dataset_number': 1,
#      'xlabel': 'noise factor', 'frac': 1},
#     {'measurement': 'loss', 'dataset_number': 2,
#      'xlabel': 'iteration', 'frac': 1},
#     {'measurement': 'loss', 'dataset_number': 2,
#      'xlabel': 'iteration', 'frac': 0.02},
#     {'measurement': 'loss', 'dataset_number': 2,
#      'xlabel': 'iteration', 'frac': 0.1},
#     {'measurement': 'loss', 'dataset_number': 2,
#      'xlabel': 'iteration', 'frac': 0.2},
#     {'measurement': 'evaluate', 'dataset_number': 2,
#      'xlabel': 'noise factor', 'frac': 1},
#     {'measurement': 'loss', 'dataset_number': 3,
#      'xlabel': 'iteration', 'frac': 1},
#     {'measurement': 'loss', 'dataset_number': 3,
#      'xlabel': 'iteration', 'frac': 0.02},
#     {'measurement': 'loss', 'dataset_number': 3,
#      'xlabel': 'iteration', 'frac': 0.1},
#     {'measurement': 'loss', 'dataset_number': 3,
#      'xlabel': 'iteration', 'frac': 0.2},
#     {'measurement': 'evaluate', 'dataset_number': 3,
#      'xlabel': 'noise factor', 'frac': 1},
# ]

def analyze_main(
    report_path: str,
    pp: list[float],
    datasets: list[Dataset],
) -> None:
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)
    
    datasets[0].learning_task
    analysis_params_list = [{
        'measurement': 'loss',
        'dataset_number': dataset.number,
        'xlabel': 'iteration',
        'frac': 0.01,
        'metrics': ['loss'],
    } for dataset in datasets] + [{
        'measurement': 'evaluate',
        'dataset_number': dataset.number,
        'xlabel': 'noise factor',
        'frac': 1,
        'metrics':  dataset.learning_task.metrics(),
    } for dataset in datasets]

    plot_analysis_fig(
        search_index=search_index,
        raw_report=raw_report,
        analysis_params_list=analysis_params_list,
        pp=pp,
    )

def search_curve_in_report(
    report_path: str,
    model: PlotModel,
):
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)
    
    return search_curve(search_index, model, raw_report)