import unittest
import torchvision.transforms as transforms
import data.make_data as make
import numpy as np
import matplotlib.pyplot as pp
import torch


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
        data_loader = torch.utils.data.DataLoader(dataset)
        dataiter = iter(data_loader)
        image, _ = next(dataiter)
        # take image out of batch should get (C x H x W) numpy arr
        npimg = image[0].numpy()
        npimg = npimg / 2 + 0.5  # un-normalize, range [0,1]
        pp.imshow(np.transpose(npimg, (1, 2, 0)))  # tranpose to (W x H X C)
        pp.show()


if __name__ == "__main__":
    unittest.main()
