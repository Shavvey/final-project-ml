import numpy as np
import data.dataframe as df
from data.dataset import Dataset

COMPRESSED_TRAIN_FILE = "ACRIMA_data/serialized/train.npz"
COMPRESSED_TEST_FILE = "ACRIMA_data/serialized/test.npz"


@staticmethod
def compressed_train_test():
    dfs = df.DataFrame.collect(df.DATA_DIRECTORY)
    train, test = Dataset.train_test_split(dfs, 0.75)
    # serialize train
    np.savez_compressed(COMPRESSED_TRAIN_FILE, xs=train.xs, ys=train.ys)
    # serialize test
    np.savez_compressed(COMPRESSED_TEST_FILE, xs=test.xs, ys=test.ys)


def get_train_test() -> tuple[Dataset, Dataset]:
    # load compressed numpy array from file
    loaded_train_data = np.load(COMPRESSED_TRAIN_FILE)
    train = Dataset(loaded_train_data["xs"], loaded_train_data["ys"])
    # load compressed numpy array from file
    loaded_test_data = np.load(COMPRESSED_TEST_FILE)
    test = Dataset(loaded_test_data["xs"], loaded_test_data["ys"])

    return (train, test)
