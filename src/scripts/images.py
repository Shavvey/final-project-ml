import torchvision
import data.make_data as make
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as pp
import numpy as np
import torch


def image_color_histogram():
    TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = make.get_ACRIMA(TRANSFORM)
    data_loader = torch.utils.data.DataLoader(dataset, len(dataset), shuffle=True)
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    print(images.shape)
    np_imgs = np.array(images)
    mean, std = 0.0, 0.0
    for image in images:
        mean += image.mean([1, 2])
        std += image.std([1, 2])

    mean /= len(images)
    std /= len(images)
    print(mean, std)

    pp.hist(np_imgs.ravel(), bins=50, density=True)
    pp.xlabel("RGB (0-255) Pixle Values")
    pp.ylabel("Frequency")
    pp.show()


def image_color_histogram_normalized():
    TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.7264, 0.3760, 0.0987], [0.0994, 0.1069, 0.0509]),
        ]
    )
    dataset = make.get_ACRIMA(TRANSFORM)
    data_loader = torch.utils.data.DataLoader(dataset, len(dataset), shuffle=True)
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    np_imgs = np.array(images)
    pp.hist(np_imgs.ravel(), bins=50, density=True)
    pp.xlabel("RGB (0-255) Pixle Values")
    pp.ylabel("Frequency")
    pp.show()
