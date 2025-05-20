## Result data analysis routines v.0.3
## Created at Thu 28 Nov 2024
## Updated at Wed 15 Jan 2025
## v.0.3 support for more datasets
## v.0.2 search_plot_data raises IndexError on failed search

import json
import os

from typing import Any, TypeAlias, TypedDict

import matplotlib.pyplot as plt
import pandas as pd

from cgtnnlib.Dataset import Dataset
from cgtnnlib.PlotModel import PlotModel, Measurement, Metric
from cgtnnlib.Report import (
    Report,
    get_reports_list,
    make_search_index,
    search_plot_data,
    load_raw_report,
    SearchIndex,
    RawReport,
)
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
    """
    Represents a curve to be plotted, potentially with quantiles."""

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
    X,
    title: str,
    xlabel: str,
    ylabel: str,
    quantiles_alpha: float,
):
    """
    Plots deviant curves with quantiles on a given axes or pyplot.

        Args:
            ax_or_plt: The matplotlib Axes object or pyplot instance to plot on.
            models: A list of dictionaries, where each dictionary represents a model
                and contains 'label', 'color', 'quantiles_color', 'quantiles_label' and 'curve'.
                The 'curve' key should contain a Pandas DataFrame with columns 0.25, 0.75, and mean.
            X: The x-axis values to use for plotting. If None, the index of the first model's curve is used.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            quantiles_alpha: The transparency (alpha) value for the quantile shading.

        Returns:
            None
    """
    if len(models) == 0:
        raise TypeError("models should not be empty")

    if X is None:
        X = models[0]["curve"].index

    for model in models:
        expected_columns = [0.25, 0.75, "mean"]
        columns = model["curve"].columns.to_list()

        assert (
            columns == expected_columns
        ), f"Bad value of curve_df.columns: should be {expected_columns}, instead got {columns}"

        ax_or_plt.plot(
            X,
            model["curve"]["mean"],
            label=model["label"],
            color=model["color"],
        )

        ax_or_plt.fill_between(
            X,
            model["curve"][0.25],
            model["curve"][0.75],
            color=model["quantiles_color"],
            alpha=quantiles_alpha,
            label=model["quantiles_label"],
        )

    set_xlabel(ax_or_plt, xlabel)
    set_ylabel(ax_or_plt, ylabel)
    set_title(ax_or_plt, title)

    ax_or_plt.legend()

    if isinstance(ax_or_plt, plt.Axes):
        ax_or_plt.grid(True)
    else:
        plt.grid(True)


def extract_values_from_search_results(
    search_results: pd.DataFrame,
    raw_report: dict[str, Any],
    measurement: Measurement,
    metric: str,
):
    """
    Extracts values from a raw report based on search results.

        Args:
            search_results: DataFrame containing search result rows with a 'Key' column.
            raw_report: Dictionary holding the raw report data, keyed by string representation of keys.
            measurement: Measurement object (used to determine extraction logic).
            metric: The metric name to extract from the raw report.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted values.  If measurement is 'loss',
                          returns a DataFrame with each row representing the value associated with the key in search_results.
                          Otherwise, returns a DataFrame with a single column of extracted metric values.
    """
    if measurement == "loss":
        return pd.DataFrame(
            [raw_report[str(row.Key)] for row in search_results.itertuples()]
        )
    else:
        cols = []

        for row in search_results.itertuples():
            report_data: dict = raw_report[str(row.Key)]

            cols.append(report_data[metric])

        return pd.DataFrame(cols)


class AnalysisParams(TypedDict):
    """
    Stores parameters for performing data analysis."""

    measurement: Measurement
    dataset_number: int
    xlabel: str
    frac: float
    metrics: list[Metric]


def search_curve(
    plot_params: PlotModel,
    raw_report: RawReport,
) -> pd.DataFrame:
    """
    Calculates and returns a DataFrame representing the curve data for plotting.

        This method extracts values from search results based on provided plot parameters
        and raw report data, calculates quantiles and mean, and then returns a fraction
        of the resulting DataFrame for zooming purposes.

        Args:
            plot_params: Parameters defining the plot characteristics (measurement, metric, frac).
            raw_report: The raw report containing search results.

        Returns:
            pd.DataFrame: A DataFrame with quantiles and mean values representing the curve,
                          potentially truncated to a specified fraction for zooming.
    """
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
    result["mean"] = values.mean()

    # for zooming
    return df_head_fraction(df=result, frac=plot_params.frac)


def plot_analysis_fig_from_file(
    dataset: Dataset,
    filenames: list[str],
    key: str,
    xlabel: str,
    frac: float,
) -> None:
    """
    Plots analysis figures from data loaded from files.

        Args:
            dataset: The dataset being analyzed.
            filenames: A list of filenames containing the report data.
            key: The key to access the relevant data within the JSON file.
            xlabel: The label for the x-axis of the plot.
            frac:  A zoom factor used in the figure title and potentially plotting.

        Returns:
            None. This function saves the generated plot to a file.
    """
    measurement_keys[KEY_EVAL, KEY_LOSS]
    dataset_number: int = analysis_params["dataset_number"]
    xlabel: str = analysis_params["xlabel"]
    frac: float = analysis_params["frac"]
    metrics: list[Metric] = analysis_params["metrics"]

    # Plot grid
    pp = [0.9]
    nrows = len(metrics)
    ncols = len(pp)

    fig, axs = plt.subplots(nrows, ncols, figsize=(24, nrows * ncols))

    for filename in get_reports_list():
        report = Report(filename=filename)
        with open(filename) as json:
            json = json.load(json)
            for key in json:
                df = pd.DataFrame(json[key])

        def make_curve_for_p(p: float) -> pd.DataFrame:
            raise RuntimeError("broken")
            # return search_curve(
            #     search_index=search_index,
            #     plot_params=PlotModel(
            #         measurement=measurement,
            #         dataset_number=dataset_number,
            #         model_name='AugmentedReLUNetwork',
            #         metric=metric,
            #         p=p,
            #         frac=frac,
            #     ),
            #     raw_report=raw_report
            # )

    reference_curve: pd.DataFrame = make_curve_for_p(0)

    for j, p in enumerate(pp):
        plot_deviant_curves_on_ax_or_plt(
            ax_or_plt=axs[i, j] if len(metrics) > 1 else axs[j],
            models=[
                {
                    "curve": reference_curve,
                    "color": "lightblue",
                    "label": "Mean of p = 0",
                    "quantiles_color": "lightgray",
                    "quantiles_label": "0.25 to 0.75 Quantiles",
                },
                {
                    "curve": make_curve_for_p(p),
                    "color": "blue",
                    "label": f"Mean of p = {p}",
                    "quantiles_color": "gray",
                    "quantiles_label": "0.25 to 0.75 Quantiles",
                },
            ],
            title=f"p = {p}",
            xlabel=xlabel,
            ylabel=metric,
            quantiles_alpha=0.1,
        )

    fig.suptitle(f"Dataset #{dataset_number}, zoom factor: {frac}")
    plt.tight_layout()
    path = os.path.join("report/", f"{measurement}_{dataset_number}_f{frac:.02f}.png")
    plt.savefig(path)
    plt.close()


def plot_analysis_fig(
    search_index: SearchIndex,
    raw_report: RawReport,
    analysis_params_list: list[AnalysisParams],
    pp: list[float],
) -> None:
    """
    Plots analysis figures based on provided parameters.

        This method iterates through a list of analysis parameter sets and generates
        plots comparing curves for different values of 'p'.  Each plot displays
        curves for a specified measurement, dataset number, and fraction, with
        subplots for each metric and value of 'p'. The plots are saved as PNG files.

        Args:
            search_index: The search index object used to retrieve data curves.
            raw_report: The raw report object containing the original data.
            analysis_params_list: A list of dictionaries, where each dictionary
                contains parameters for a single analysis (measurement, dataset_number,
                xlabel, frac, and metrics).
            pp: A list of float values representing the 'p' values to be plotted.

        Returns:
            None
    """
    for analysis_params in analysis_params_list:
        measurement = analysis_params["measurement"]
        dataset_number = analysis_params["dataset_number"]
        xlabel = analysis_params["xlabel"]
        frac = analysis_params["frac"]
        metrics = analysis_params["metrics"]

        # Plot grid
        nrows = len(metrics)
        ncols = len(pp)
        fig, axs = plt.subplots(nrows, ncols, figsize=(24, nrows * ncols))

        for i, metric in enumerate(metrics):

            def make_curve_for_p(p: float) -> pd.DataFrame:
                return search_curve(
                    search_index=search_index,
                    plot_params=PlotModel(
                        measurement=measurement,
                        dataset_number=dataset_number,
                        model_name="AugmentedReLUNetwork",
                        metric=metric,
                        p=p,
                        frac=frac,
                    ),
                    raw_report=raw_report,
                )

            reference_curve: pd.DataFrame = make_curve_for_p(0)

            for j, p in enumerate(pp):
                plot_deviant_curves_on_ax_or_plt(
                    ax_or_plt=axs[i, j] if len(metrics) > 1 else axs[j],
                    models=[
                        {
                            "curve": reference_curve,
                            "color": "lightblue",
                            "label": "Mean of p = 0",
                            "quantiles_color": "lightgray",
                            "quantiles_label": "0.25 to 0.75 Quantiles",
                        },
                        {
                            "curve": make_curve_for_p(p),
                            "color": "blue",
                            "label": f"Mean of p = {p}",
                            "quantiles_color": "gray",
                            "quantiles_label": "0.25 to 0.75 Quantiles",
                        },
                    ],
                    title=f"p = {p}",
                    xlabel=xlabel,
                    ylabel=metric,
                    quantiles_alpha=0.1,
                )
        fig.suptitle(f"Dataset #{dataset_number}, zoom factor: {frac}")
        plt.tight_layout()
        path = os.path.join(
            "report/", f"{measurement}_{dataset_number}_f{frac:.02f}.png"
        )
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
    """
    Analyzes a report and generates plots based on provided datasets.

        This method loads a raw report, creates a search index for it,
        defines analysis parameters for loss curves and evaluation metrics
        for each dataset, and then generates a plot using these components.

        Args:
            report_path: The path to the report file.
            pp: A list of float values (purpose not specified in code).
            datasets: A list of Dataset objects to analyze.

        Returns:
            None: This method does not return any value; it generates a plot as a side effect.
    """
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)

    datasets[0].learning_task
    analysis_params_list = [
        {
            "measurement": "loss",
            "dataset_number": dataset.number,
            "xlabel": "iteration",
            "frac": 0.01,
            "metrics": ["loss"],
        }
        for dataset in datasets
    ] + [
        {
            "measurement": "evaluate",
            "dataset_number": dataset.number,
            "xlabel": "noise factor",
            "frac": 1,
            "metrics": dataset.learning_task.metrics(),
        }
        for dataset in datasets
    ]

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
    """
    Searches for a curve within a report based on a given model.

        Loads the raw report data, creates a search index, and then uses the
        index and model to find the corresponding curve in the report.

        Args:
            report_path: The path to the report file.
            model: The PlotModel representing the curve to search for.

        Returns:
            The result of searching for the curve within the report.  The specific
            return type depends on the implementation of `search_curve`.
    """
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)

    return search_curve(search_index, model, raw_report)
