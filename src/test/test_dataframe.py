import unittest
from src.data.dataframe import DataFrame, DATA_DIRECTORY
from PIL import Image


class TestDataFrame(unittest.TestCase):
    def test_dataframe_collect(self):
        """Test to see if image data is presevered inside of the dataframes"""
        # since we launch a new thread for the image.show(), unittest will warn of resource leak
        dfs = DataFrame.collect(DATA_DIRECTORY)
        imgs = DataFrame.get_images(dfs)
        im1 = Image.fromarray(imgs[0])
        # open with stored absolute path
        with Image.open(DATA_DIRECTORY + "/" + dfs[0].filename) as im2:
            assert im1.__hash__ == im2.__hash__

if __name__ == "__main__":
    unittest.main()
