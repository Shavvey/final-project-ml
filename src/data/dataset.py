from torch.utils.data import Dataset
import torch


class ACRIMA(Dataset):
    """
    Pytorch Dataset subclass for our ACRIMA dataset.
    Constructor can take lists or numpy arrays, so type annotations have been left out.
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            # apply the transform to image if provided one
            image = self.transform(image)
        # give back image, label pair
        return image, label
