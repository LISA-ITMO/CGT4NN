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

KEY_EVAL = "eval"
KEY_LOSS = "loss"
KEY_DATASET = "dataset"
KEY_MODEL = "model"
KEY_TRAIN_NOISE_GENERATOR = "train_noise_generator"
KEY_EPOCHS = "epochs"


def now_isoformat() -> str:
    """
    Returns the current time as an ISO 8601 formatted string.

        Args:
            None

        Returns:
            str: The current date and time in ISO 8601 format (e.g., '2023-10-27T10:30:00.123456').
    """
    return datetime.now().isoformat()


def see_value(value) -> str:
    """
    Returns a string representation of the given value.

        Args:
            value: The value to represent as a string.

        Returns:
            str: A user-friendly string describing the value, including its type and size if applicable.
    """
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
    """
    A class for managing and accessing report data.

    This class provides methods for creating, loading, saving, and querying
    report information stored in a JSON-like format."""

    dir: str
    filename: str
    raw_report: RawReport = {}

    def __init__(
        self,
        dir: str = REPORT_DIR,
        filename: str = "report.json",
        must_exist: bool = False,
    ):
        """
        Initializes a new report instance.

            Args:
                dir: The directory to store the report in.
                filename: The name of the report file.
                now_isoformat: A callable that returns the current time in ISO format.
                must_exist: Whether the report file must already exist.  If True and the
                    file does not exist, a LookupError is raised.

            Returns:
                None
        """
        self.dir = dir
        self.filename = filename
        self.set("started", now_isoformat())
        if os.path.exists(self.path):
            print(f"Report found at {self.path}. Loading...")
            self.raw_report = load_raw_report(self.path)
            print("Report loaded.")
        elif must_exist:
            raise LookupError(f"Report at {self.path} must exist, but it doesn't")

    @staticmethod
    def from_path(
        path: str,
        must_exist: bool = False,
    ) -> "Report":
        [dir, filename] = path.split("/", maxsplit=1)
        return Report(dir, filename, must_exist)

    @property
    def path(self):
        """
        Returns the file path of the report."""
        return os.path.join(self.dir, self.filename)

    def set(self, key: str, data: ReportEntry):
        """
        Checks if a key exists in the raw report.

            Args:
                key: The key to check for existence.

            Returns:
                bool: True if the key is present in the raw report, False otherwise.
        """
        self.raw_report[key] = data

    def has(self, key: str) -> bool:
        """
        Checks if a key exists in the raw report.

            Args:
                key: The key to check for existence.

            Returns:
                bool: True if the key exists in the raw report, False otherwise.
        """
        return key in self.raw_report

    def get(self, key: str):
        """
        Retrieves the value associated with a given key from the cache.

          Args:
            key: The key whose value is to be retrieved.

          Returns:
            The value associated with the key, or None if the key is not found.
        """
        return self.raw_report[key]

    def save(self):
        """
        Saves the raw report data to a JSON file.

            Args:
                file: The file object to write the JSON data to.

            Returns:
                None
        """
        self.set("saved", now_isoformat())
        with open(self.path, "w") as file:
            json.dump(self.raw_report, file, indent=4)
        print(f"Report saved to {self.path}.")

    def see(self):
        """
        Prints the raw report data to the console.

            Iterates through the keys and values of the raw report dictionary
            and prints each key-value pair, using the `see_value` function to
            format the value for display.

            Args:
                None

            Returns:
                None
        """
        title = f"Report {self.path}"
        print(title)
        print("".join(["=" for _ in range(0, len(title))]))

        for key in self.raw_report:
            value = self.raw_report[key]
            print(f"{key}: {see_value(value)}")


def load_raw_report(path: str) -> RawReport:
    """
    Creates a search index from a raw report.

        Args:
            raw_report: The raw report to create the search index from.

        Returns:
            SearchIndex: A SearchIndex object representing the searchable data
                         from the raw report.
    """
    with open(path) as fd:
        return json.load(fd)


def make_search_index(raw_report: RawReport) -> SearchIndex:
    """
    Searches the provided index for data matching plot parameters.

        Args:
            search_index: The DataFrame representing the search index.
            plot_params: An object containing the desired plot parameters to filter by.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered data from the search index
                          that matches the provided plot parameters.
    """
    df = pd.DataFrame([[key] + key.split("_") for key in raw_report.keys()])

    df.columns = ["Key", "Measurement", "Network", "Dataset", "P", "N"]

    # Remove metadata
    df = df[df["Key"] != "started"]
    df = df[df["Key"] != "saved"]
    df = df[df["Key"] != "comment"]

    df.Dataset = df.Dataset.apply(lambda x: int(x))

    df.P = df.P.apply(lambda x: float(x[1:]))
    df.N = df.N.apply(lambda x: int(x[1:]))

    return df


def search_plot_data(
    search_index: pd.DataFrame,
    plot_params: PlotModel,
) -> pd.DataFrame:
    """
    Searches the search index DataFrame for data matching the given plot parameters.

        Args:
            search_index: The DataFrame containing indexed plot data.
            plot_params: An object containing the measurement, dataset number,
                model name, and p value to search for.

        Returns:
            pd.DataFrame: A DataFrame containing the search results that match all provided criteria.
                Raises an IndexError if no matching data is found.
    """
    search_results: pd.DataFrame = (
        search_index.loc[search_index.Measurement == plot_params.measurement]
        .loc[search_index.Dataset == plot_params.dataset_number]
        .loc[search_index.Network == plot_params.model_name]
        .loc[search_index.P == plot_params.p]
    )

    if search_results.empty:
        print("Search index:")
        print(search_index)
        raise IndexError(f"Search failed: {plot_params}")

    return search_results
