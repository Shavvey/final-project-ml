import unittest
import torchvision.transforms as transforms
from data.make_data import get_ACRIMA
import numpy as np
import matplotlib.pyplot as pp


class TestACRIMADataset(unittest.TestCase):
    def test_acrima_dataset_with_loader(self):
        EXAMPLE_TRANSFORM = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # retreive dataset, do a basic transform
        dataset = get_ACRIMA(EXAMPLE_TRANSFORM)
        # show one of theimages from dataset, for fun
        img = dataset.images[0]
        npimg = img.numpy()
        pp.imshow(np.transpose(npimg, (1, 2, 0)))
        pp.show()


if __name__ == "__main__":
    unittest.main()
