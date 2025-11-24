from data.dataset import Dataset
from data.dataframe import DataFrame, DATA_DIRECTORY


def main():
    dfs = DataFrame.collect(DATA_DIRECTORY)
    imgs = DataFrame.get_images(dfs)


if __name__ == "__main__":
    main()
