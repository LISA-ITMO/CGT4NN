## Dataset utilities v.0.4
## Created at Tue 26 Nov 2024
## Updated at Tue 14 Jan 2024
## v.0.4 - new datasets: eye_movements, wine_quality,
##                       Hill_Valley_with_noise, Hill_Valley_without_noise
## v.0.3 - Drop make_datasetN() functions in favor of datasets list
## v.0.2 - sha1 hash checking to avoid duplicate downloads

import os
import urllib.request
import hashlib

import pandas as pd


from cgtnnlib.DatasetCollection import DatasetCollection
from cgtnnlib.LearningTask import REGRESSION_TASK, CLASSIFICATION_TASK
from cgtnnlib.Dataset import Dataset
from cgtnnlib.fn import compose
from cgtnnlib.constants import DATA_DIR, PMLB_TARGET_COL
from cgtnnlib.preprocess import (
    preprocess_breast_cancer,
    preprocess_car_evaluation,
    preprocess_student_performance_factors,
)

# ::: Here can't be a good place for this
os.makedirs(DATA_DIR, exist_ok=True)

## CSV sources


def download_csv(
    url: str, saved_name: str, sha1: str, features: list[str] | None = None
) -> pd.DataFrame:
    """
    Downloads a CSV file from a `url` and saves it to the default data
    directory. Before downloading, `download_csv` checks if there's already
    a file with SHA-1 hash `sha1`. Returns the file's contents
    as Pandas `DataFrame`.

    If `columns` is supplied, it's interpreted as a list of column names
    for tables with no headers.
    """
    file_path = os.path.join(DATA_DIR, saved_name)

    def calculate_sha1(file_path):
        hasher = hashlib.sha1()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    if os.path.exists(file_path):
        file_sha1 = calculate_sha1(file_path)
        if file_sha1 != sha1:
            raise ValueError(
                f"SHA1 mismatch for existing file: {file_path}. Expected {sha1}, got {file_sha1}"
            )
        else:
            print(f"File {file_path} exists and SHA1 matches, skipping download.")
            if features is None:
                return pd.read_csv(file_path)
            else:
                return pd.read_csv(file_path, header=None, names=features)
    else:
        print(f"Downloading {url} to {file_path}")
        urllib.request.urlretrieve(url, file_path)
        downloaded_sha1 = calculate_sha1(file_path)
        if downloaded_sha1 != sha1:
            os.remove(file_path)
            raise ValueError(
                f"SHA1 mismatch for downloaded file: {file_path}. Expected {sha1}, got {downloaded_sha1}"
            )

    if features is None:
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, header=None, names=features)


def download_pmlb(dataset_name: str) -> pd.DataFrame:
    """
    Downloads a dataset from the PML Benchmark datasets.
    See: <https://epistasislab.github.io/pmlb/>

        Args:
            dataset_name: The name of the dataset to download.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the downloaded dataset.
    """
    import pmlb

    return pmlb.fetch_data(dataset_name, return_X_y=False, local_cache_dir=DATA_DIR)


datasets: DatasetCollection = DatasetCollection(
    [
        Dataset(
            number=1,
            name="wisc_bc_data",
            learning_task=CLASSIFICATION_TASK,
            classes_count=2,
            target="diagnosis",
            load_data=compose(
                preprocess_breast_cancer,
                lambda: download_csv(
                    url="https://raw.githubusercontent.com/dataspelunking/MLwR/refs/heads/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2003/wisc_bc_data.csv",
                    saved_name="wisc_bc_data.csv",
                    sha1="3b75f889e7e8d140b9eb28df39556b94b4331e33",
                ),
            ),
        ),
        Dataset(
            number=2,
            name="car_evaluation",
            learning_task=CLASSIFICATION_TASK,
            classes_count=4,
            target="class",
            load_data=compose(
                preprocess_car_evaluation,
                lambda: download_csv(
                    url="https://raw.githubusercontent.com/mragpavank/car-evaluation-dataset/refs/heads/master/car_evaluation.csv",
                    saved_name="car_evaluation.csv",
                    sha1="985852bc1bb34d7cb3c192d6b8e7127cc743e176",
                    features=[
                        "buying",
                        "maint",
                        "doors",
                        "persons",
                        "lug_boot",
                        "safety",
                        "class",
                    ],
                ),
            ),
        ),
        Dataset(
            number=3,
            name="StudentPerformanceFactors",
            learning_task=REGRESSION_TASK,
            classes_count=1,
            target="Exam_Score",
            # StudentPerformanceFactors.csv is stored in the repo
            load_data=compose(
                preprocess_student_performance_factors,
                lambda: pd.read_csv(
                    "data/StudentPerformanceFactors.csv",
                ),
            ),
        ),
        Dataset(
            number=4,
            name="allhyper",
            learning_task=CLASSIFICATION_TASK,
            classes_count=4,
            target=PMLB_TARGET_COL,
            load_data=lambda: download_pmlb("allhyper"),
        ),
        Dataset(
            number=5,
            name="eye_movements",
            learning_task=CLASSIFICATION_TASK,
            classes_count=2,
            target="label",
            load_data=lambda: download_csv(
                url="https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/dabc0f5cea2459217a54bf275227e68cda218e9d/clf_cat/eye_movements.csv",
                saved_name="eye_movements.csv",
                sha1="4ed08bb19912a220a18fa0399821e3ee57dc1094",
            ),
        ),
        Dataset(
            number=6,
            name="wine_quality",
            learning_task=REGRESSION_TASK,
            classes_count=1,
            target="quality",
            load_data=lambda: download_csv(
                url="https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/dabc0f5cea2459217a54bf275227e68cda218e9d/reg_num/wine_quality.csv",
                saved_name="wine_quality.csv",
                sha1="83caedd8c35eba2146ea8eaf9f1d1dfa208f50ec",
            ),
        ),
        Dataset(
            number=7,
            name="Hill_Valley_with_noise",
            learning_task=CLASSIFICATION_TASK,
            classes_count=2,
            target=PMLB_TARGET_COL,
            load_data=lambda: download_pmlb("Hill_Valley_with_noise"),
        ),
        Dataset(
            number=8,
            name="Hill_Valley_without_noise",
            learning_task=CLASSIFICATION_TASK,
            classes_count=2,
            target=PMLB_TARGET_COL,
            load_data=lambda: download_pmlb("Hill_Valley_without_noise"),
        ),
        Dataset(
            number=9,
            name="294_satellite_image",
            learning_task=REGRESSION_TASK,
            classes_count=1,
            target=PMLB_TARGET_COL,
            load_data=lambda: download_pmlb("294_satellite_image"),
        ),
        Dataset(
            number=10,
            name="1030_ERA",
            learning_task=REGRESSION_TASK,
            classes_count=1,
            target=PMLB_TARGET_COL,
            load_data=lambda: download_pmlb("1030_ERA"),
        ),
    ]
)
"""
Datasets collection.

NOTE: https://jamesmccaffrey.wordpress.com/2024/11/01/i-get-tricked-by-the-pollen-dataset/

Please do not add PMLB's "529_pollen" into the collection.
"""
