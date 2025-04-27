from cgtnnlib.Dataset import Dataset


from typing import Iterable


class DatasetCollection(Iterable):
    def __init__(self, datasets: list[Dataset]):
        self._datasets: list[Dataset] = datasets

        self._name_to_index: dict[str, int] = {
            ds.name: i for i, ds in enumerate(datasets)
        }

    def __getitem__(self, key: str | int) -> Dataset:
        if isinstance(key, str):
            index = self._name_to_index.get(key)
            if index is None:
                raise KeyError(
                    f"Dataset with name '{key}' not found."
                )
            return self._datasets[index]
        elif isinstance(key, int):
            return self._datasets[key]
        else:
            raise TypeError(
                "Key must be either an integer or a string (dataset name)."
            )

    def __iter__(self):
        return iter(self._datasets)

    def __len__(self):
        return len(self._datasets)