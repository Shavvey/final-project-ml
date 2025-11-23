from data.dataset import Dataset
from data.dataframe import DataFrame, DATA_DIRECTORY


def main():
    dfs = DataFrame.collect(DATA_DIRECTORY)
    train, test = Dataset.train_test_split(dfs, 0.75)


if __name__ == "__main__":
    main()
