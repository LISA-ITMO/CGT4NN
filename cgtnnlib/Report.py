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
import glob

import torch
import numpy as np
import pandas as pd

from cgtnnlib.PlotModel import PlotModel
from cgtnnlib.constants import MODEL_DIR, REPORT_DIR



SearchIndex: TypeAlias = pd.DataFrame
ReportEntry = dict | list | int | float | str
RawReport: TypeAlias = dict[str, ReportEntry]

KEY_EVAL = 'eval'
KEY_LOSS = 'loss'
KEY_DATASET = 'dataset'
KEY_MODEL = 'model'
KEY_TRAIN_NOISE_GENERATOR = 'train_noise_generator'
KEY_EPOCHS = 'epochs'

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


def get_reports_list() -> list[str]:
  """
  Возвращает список имен всех отчётов в указанной папке.

  Args:
    folder_path: Путь к папке.

  Returns:
    Список строк, каждая из которых является именем JSON-файла.
  """
  pattern = os.path.join(MODEL_DIR, "*.json")
  full_paths = glob.glob(pattern)
  file_names = [os.path.basename(path) for path in full_paths]
  return file_names


class Report:
    dir: str
    filename: str
    raw_report: RawReport = {}
    
    def __init__(
        self,
        dir: str = REPORT_DIR,
        filename: str = 'report.json',
        must_exist: bool = False,
    ):
        self.dir = dir
        self.filename = filename
        self.set('started', now_isoformat())
        if os.path.exists(self.path):
            print(f"Report found at {self.path}. Loading...")
            self.raw_report = load_raw_report(self.path)
            print("Report loaded.")
        elif must_exist:
            raise LookupError(f'Report at {self.path} must exist, but it doesn\'t')
    
    @staticmethod
    def from_path(
        path: str,
        must_exist: bool = False,
    ) -> 'Report':
        [dir, filename] = path.split('/', maxsplit=1)
        return Report(dir, filename, must_exist)
    
    @property
    def path(self):
        return os.path.join(self.dir, self.filename)

    def set(self, key: str, data: ReportEntry):
        self.raw_report[key] = data
        
    def has(self, key: str) -> bool:
        return key in self.raw_report

    def get(self, key: str):
        return self.raw_report[key]

    def save(self):
        self.set('saved', now_isoformat())
        with open(self.path, 'w') as file:
            json.dump(self.raw_report, file, indent=4)
        print(f"Report saved to {self.path}.")

    def see(self):
        title = f"Report {self.path}"
        print(title)
        print(''.join(['=' for _ in range(0, len(title))]))
        
        for key in self.raw_report:
            value = self.raw_report[key]
            print(f"{key}: {see_value(value)}")


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