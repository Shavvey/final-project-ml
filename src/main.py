from data.dataset import Dataset
from data.dataframe import DataFrame, DATA_DIRECTORY
from PIL import Image
import base64 as b
from io import BytesIO


def main():
    dfs = DataFrame.collect(DATA_DIRECTORY)
    train, test = Dataset.train_test_split(dfs, 0.75)
    decoded_bytes = b.b64decode(train.xs[0])
    i = Image.open(BytesIO(decoded_bytes))
    i.show()


if __name__ == "__main__":
    main()
