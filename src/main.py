from data.dataset import ACRIMA, Dataset
from data.dataframe import DataFrame, DATA_DIRECTORY
from PIL import Image


def main():
    dfs = DataFrame.collect(DATA_DIRECTORY)
    imgs = DataFrame.get_images(dfs)
    labels = DataFrame.get_labels(dfs)
    im = Image.fromarray(imgs[0])
    im.show()
    dataset = ACRIMA(imgs, labels)


if __name__ == "__main__":
    main()
