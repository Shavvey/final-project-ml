from torch.utils.data import Dataset, Subset
import torch


class ACRIMA(Dataset):
    """
    Pytorch Dataset subclass for our ACRIMA dataset.
    Constructor can take lists or numpy arrays, so type annotations have been left out.
    """

    def __init__(self, images, labels, transform=None):
        """Construct dataset from original numpy arrays of images and labels.
        Optionally, we can also store and use a transform."""
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Dunder method that will return length of dataset.
        This simply just defers to numpy implementation of `len` method."""
        return len(self.labels)

    def __getitem__(self, index):
        """Gets items and applys the transform if we have one"""
        if torch.is_tensor(index):
            index = index.tolist()
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            # apply the transform to image if provided one
            image = self.transform(image)
        # give back image, label pair
        return image, label


@staticmethod
def calc_mean_std(subset: Subset) -> tuple[float, float]:
    """Calculate mean and standard deviation for slice of dataset"""
    data_loader = torch.utils.data.DataLoader(subset, len(subset))
    dataiter = iter(data_loader)
    images, _ = next(dataiter)
    mean, std = 0.0, 0.0
    for image in images:
        mean += image.mean([1, 2])
        std += image.std([1, 2])

    mean /= len(images)
    std /= len(images)
    return mean, std
