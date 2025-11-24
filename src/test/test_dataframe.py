import unittest
from src.data.dataframe import DataFrame, DATA_DIRECTORY
from PIL import Image

class TestDataFrame(unittest.TestCase):
    def test_dataframe_collect(self):
        dfs = DataFrame.collect(DATA_DIRECTORY)
        imgs = DataFrame.get_images(dfs)
        im = Image.fromarray(imgs[0])
        im.show()

if __name__ == "__main__":
    unittest.main()
