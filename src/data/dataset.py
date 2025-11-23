import numpy as np
import numpy.typing as npt
import data.dataframe as df
import math as m
from typing import Optional


class Dataset:
    xs: npt.NDArray
    ys: npt.NDArray

    def __init__(self, xs: npt.NDArray, ys: npt.NDArray):
        self.xs = xs
        self.ys = ys

    @staticmethod
    def train_test_split(
        data: list[df.DataFrame], train_ratio: float, test_ratio: Optional[float] = None
    ) -> tuple["Dataset", "Dataset"]:
        # split dataframes into train and test
        train_idx = m.floor(len(data) * train_ratio)
        train_dfs = data[:train_idx]
        test_dfs = []
        if test_ratio != None:
            test_idx = m.floor(len(data) * test_ratio)
            test_dfs = data[train_idx : train_idx + test_idx]
        else:
            test_dfs = data[train_idx:]
        train_dataset = Dataset(
            np.array(df.DataFrame.get_images(train_dfs), dtype=bytes),
            np.array(df.DataFrame.get_labels(train_dfs), dtype=bool),
        )

        test_dataset = Dataset(
            np.array(df.DataFrame.get_images(test_dfs), dtype=bytes),
            np.array(df.DataFrame.get_labels(test_dfs), dtype=bool),
        )

        return (train_dataset, test_dataset)
