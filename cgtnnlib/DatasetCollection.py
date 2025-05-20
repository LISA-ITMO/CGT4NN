from cgtnnlib.Dataset import Dataset


from typing import Iterable


class DatasetCollection(Iterable):
    """
    A collection of datasets, allowing access by index or name."""

    def __init__(self, datasets: list[Dataset]):
        """
        Initializes the DatasetCollection with a list of datasets.

            Args:
                datasets: A list of Dataset objects to be included in the collection.

            Returns:
                None
        """
        self._datasets: list[Dataset] = datasets

        self._name_to_index: dict[str, int] = {
            ds.name: i for i, ds in enumerate(datasets)
        }

    def __getitem__(self, key: str | int) -> Dataset:
        """
        Retrieves a dataset by either its index or name.

            Args:
                key: The index (int) or name (str) of the desired dataset.

            Returns:
                Dataset: The requested dataset.

            Raises:
                KeyError: If the dataset with the given name is not found.
                TypeError: If the key is neither an integer nor a string.
        """
        if isinstance(key, str):
            index = self._name_to_index.get(key)
            if index is None:
                raise KeyError(f"Dataset with name '{key}' not found.")
            return self._datasets[index]
        elif isinstance(key, int):
            return self._datasets[key]
        else:
            raise TypeError("Key must be either an integer or a string (dataset name).")

    def __iter__(self):
        """
        Returns an iterator for the underlying datasets.

          Args:
            None

          Returns:
            iterator: An iterator that yields elements from the _datasets collection.
        """
        return iter(self._datasets)

    def __len__(self):
        """
        Returns the number of datasets in this object.

          Args:
            None

          Returns:
            int: The number of datasets currently stored.
        """
        return len(self._datasets)
