import torch
import torch.optim as optim
import torch.nn as nn

from data.make_data import get_ACRIMA
from network.nn import CNN

def train(epochs: int):
    # use gpu CUDA API if able, otherwise just use the regular cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    # create model
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    full_dataset = get_ACRIMA()
    train_dataset, test_dataset  = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    for epoch in range(epochs):
        for batch_idx, (image, label) in enumerate(train_loader):
            # do the training
            pass
