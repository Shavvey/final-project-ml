from torch.utils.data import Dataset
import torch


class ACRIMA(Dataset):
    def __init__(self, data: list, targets: list, transform=None):
        self.data = torch.Tensor(data)  # change data to tensor
        self.targets = torch.Tensor(targets)  # change targets/labels in tensor
        self.transform = transform
