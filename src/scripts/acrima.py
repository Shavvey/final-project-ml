import torchvision
import data.make_data as make
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as pp
import numpy as np


def imshow(img):
    npimg = img.numpy()
    pp.imshow(np.transpose(npimg, (1, 2, 0)))
    pp.show()


def make_grid_of_acrima():
    EXAMPLE_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = make.get_ACRIMA(EXAMPLE_TRANSFORM)
    data_loader = torch.utils.data.DataLoader(dataset, 25, shuffle=True)
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images, 5))
    pp.show()
