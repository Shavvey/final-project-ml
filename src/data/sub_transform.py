from torch.utils.data import Dataset


class SubsetTransform(Dataset):
    """Simple subclass that allows use to apply transforms to subsets.
    This is useful when calculating the mean and std, 
    and apply a normalization later after we split data."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
