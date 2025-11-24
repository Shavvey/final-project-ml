import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional

from data.make_data import get_ACRIMA
from network.nn import CNN


# NOTE: we may need to implement are own transform for ACRIMA class later?
IMAGE_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def init_device(verbose: Optional[bool] = None):
    # use gpu CUDA API if able, otherwise just use the regular cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose == True:
        print(device)


def train(epochs: int):
    init_device()
    criterion = nn.BCEWithLogitsLoss()
    # create model
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optionally apply transforms here
    full_dataset = get_ACRIMA(transform = IMAGE_TRANSFORM)
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.8, 0.2]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=20, shuffle=True
    )
    for epoch in range(epochs):
        running_loss = 0.0
        # do the training optimization step for the number of batches we have
        for batch_idx, (images, labels) in enumerate(train_loader):
            #print(image[0])
            # zero out param gradients
            optimizer.zero_grad()
            # use image to obtain outputs
            outputs = model(images)
            # compute the loss
            loss = criterion(outputs, labels)
            # backprop
            loss.backward()
            # making a optimizaton step
            optimizer.step()
            # find the running loss
            running_loss += loss.item()
            if batch_idx % 5 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
