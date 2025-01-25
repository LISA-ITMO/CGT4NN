## Report v.0.7
## Created at Tue 26 Nov 2024
## Modified at Wed 15 Jan 2025
## v.0.7 - Load report data from file on initialization
## v.0.6 - .record_running_losses() now accepts a Dataset
##          instead of a TrainingParameters (which is gone now)
## v.0.5 - class Report: report_running_losses()
## v.0.4 - RawReport, SearchIndex
## v.0.3 - eval_report_key()
## v.0.2 - .path, .filename properties; .see() method

from typing import TypeAlias
from datetime import datetime
import os
import json

import torch
import numpy as np
import pandas as pd

from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.PlotModel import PlotModel

from cgtnnlib.nn.NetworkLike import NetworkLike


SearchIndex: TypeAlias = pd.DataFrame
RawReport: TypeAlias = dict[str, dict | list | str]


def now_isoformat() -> str:
    return datetime.now().isoformat()


def see_value(value) -> str:
    if isinstance(value, (list, np.ndarray, torch.Tensor)):
        num_items = len(value) if isinstance(value, list) else value.size
        return f"{type(value).__name__}({num_items} items)"
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (int, float, bool)):
        return str(value)
    else:
        return f"{type(value).__name__}(...)"


class Report:
    dir: str
    raw: RawReport = {
        'started': now_isoformat()
    }
    
    def __init__(self, dir: str):
        self.dir = dir
        if os.path.exists(self.path):
            print(f"Report found at {self.path}. Loading...")
            self.raw = load_raw_report(self.path)
            print("Report loaded.")
    
    @property 
    def filename(self):
        return 'report.json'
    
    @property
    def path(self):
        return os.path.join(self.dir, self.filename)

    def append(self, key: str, data: dict | list):
        self.raw[key] = data

    def save(self):
        self.append('saved', now_isoformat())
        with open(self.path, 'w') as file:
            json.dump(self.raw, file, indent=4)
        print(f"Report saved to {self.path}.")
    
    def see(self):
        title = f"Report {self.path}"
        print(title)
        print(''.join(['=' for _ in range(0, len(title))]))
        
        for key in self.raw:
            value = self.raw[key]
            print(f"{key}: {see_value(value)}")
    
    def record_running_losses(
        self,
        running_losses: list[float],
        model: NetworkLike,
        dataset: Dataset,
        experiment_params: ExperimentParameters,
    ):
        key = f'loss_{type(model).__name__}_{dataset.number}_p{experiment_params.p}_N{experiment_params.iteration}'
        self.append(key, running_losses)
        

def eval_report_key(
    model_name: str,
    dataset_number: int,
    p: float,
    iteration: int,
) -> str:
    return f'evaluate_{model_name}_{dataset_number}_p{p}_N{iteration}'


def load_raw_report(path: str) -> RawReport:
    with open(path) as fd:
        return json.load(fd)


def make_search_index(raw_report: RawReport) -> SearchIndex:
    df = pd.DataFrame([[key] + key.split('_') for key in raw_report.keys()])

    df.columns = ['Key', 'Measurement', 'Network', 'Dataset', 'P', 'N']

    # Remove metadata
    df = df[df['Key'] != 'started']
    df = df[df['Key'] != 'saved']
    df = df[df['Key'] != 'comment']

    df.Dataset = df.Dataset.apply(lambda x: int(x))

    df.P = df.P.apply(lambda x: float(x[1:]))
    df.N = df.N.apply(lambda x: int(x[1:]))

    return df


def search_plot_data(
    search_index: pd.DataFrame,
    plot_params: PlotModel,
) -> pd.DataFrame:
    search_results: pd.DataFrame = (
        search_index
            .loc[search_index.Measurement == plot_params.measurement]
            .loc[search_index.Dataset == plot_params.dataset_number]
            .loc[search_index.Network == plot_params.model_name]
            .loc[search_index.P == plot_params.p]
    )

    if search_results.empty:
        print("Search index:")
        print(search_index)
        raise IndexError(f"Search failed: {plot_params}")

    return search_results