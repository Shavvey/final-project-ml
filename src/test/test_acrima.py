import unittest
import torchvision.transforms as transforms
import data.make_data as make
import numpy as np
import matplotlib.pyplot as pp
import torch
import torchvision


class TestACRIMADataset(unittest.TestCase):
    def test_acrima_dataset_with_loader(self):
        EXAMPLE_TRANSFORM = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # retreive dataset, do a basic transform
        dataset = make.get_ACRIMA(EXAMPLE_TRANSFORM)
        # show one of theimages from dataset, for fun
        img = dataset.images[0]
        pp.imshow(np.transpose(img, (0, 1, 2)))
        pp.show()


if __name__ == "__main__":
    unittest.main()
