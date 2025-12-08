import torch
from torch.utils.data import Subset, Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional
import sklearn.metrics as metrics
import data.sub_transform as st
import data.dataset as dataset

from torchvision.transforms.transforms import Compose

from data.make_data import get_ACRIMA
from network.cnn_gray import CNNGray

# basic image transform, throws pixel data into tensor then normalizes rgb channel
BASE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5),
    ]
)


def init_device(verbose: Optional[bool] = None):
    # use gpu CUDA API if able, otherwise just use the regular cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: implement GPU device copy when we have gpu available
    if verbose == True:
        print(device)


def train_test_split(
    split_ratios: list[float], transform: Optional[Compose] = None
) -> tuple[Subset, Subset]:
    full_dataset = get_ACRIMA(transform=transform)
    # get random split of the initial dataset into two subsets, one test and one train
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, split_ratios
    )
    return train_dataset, test_dataset


def train_network(
    model: CNNGray,
    train_dataset: Subset | Dataset,
    epochs: int,
    num_batches: int,
    save_path: Optional[str] = None,
) -> CNNGray:
    # use cpu, or CUDA api with compatible GPU with available
    init_device()
    # TODO: revisit the loss and optimizer functions later
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=num_batches, shuffle=True
    )
    for epoch in range(epochs):
        running_loss = 0.0
        # do the training optimization step for the number of batches we have
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            if batch_idx % 5 == 0:  # print every 5 mini_batches
                print(
                    f"[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / num_batches:.3f}"
                )
                running_loss = 0.0
    # save the trained network if optional flag set
    if save_path != None:
        torch.save(model.state_dict(), save_path)
    return model


def test_network(
    model: CNNGray, test_dataset: Subset | Dataset, test_log: Optional[bool] = None
) -> tuple[list[int], list[int]]:
    # set model into eval mode
    model.eval()
    # create test loader from the dataset
    test_loader = torch.utils.data.DataLoader(test_dataset)
    num_correct, num_samples = 0, 0
    preds: list[int] = []
    actuals: list[int] = []
    with torch.no_grad():
        for train_idx, (image, label) in enumerate(test_loader):
            outputs = model(image)
            # return (value, index) tuple, only need the 2nd member
            _, predicted = torch.max(outputs, 1)  # return max, reduce to one dim
            # use argmax to convert onehot representation to regular integer label
            int_label = torch.argmax(label)
            preds.append(int(predicted.item()))
            actuals.append(int(int_label.item()))
            if test_log == True:
                if train_idx == 0:
                    print("======")
                print(f"[PRED {train_idx+1}]: {predicted.item()}")
                print(f"[LABEL {train_idx+1}]: {int_label}")
                print("======")
            num_samples += label.size(0)
            num_correct += (predicted == int_label).sum().item()
    print(f"NUMBER OF TEST SAMPLES: {num_samples}")
    print(f"NUMBER OF CORRECT PREDICTIONS: {num_correct}")
    acc = (num_correct) / (num_samples)
    print(f"ACCURACY ON TEST SAMPLES: {acc* 100:.3f}%")
    return preds, actuals


def train_test(split_ratio: list[float], epochs: int, batch_size: int):
    model = CNNGray()
    train_dataset, test_dataset = train_test_split(split_ratio, BASE_TRANSFORM)
    mean, std = dataset.calc_mean_std(train_dataset)
    print(mean, std)
    # apply normalization to train and test, after computing normal and std using train
    NORMAL_TRANSFORM = transforms.Compose([transforms.Normalize(mean, std)])
    train = st.SubsetTransform(train_dataset, transform=NORMAL_TRANSFORM)
    test = st.SubsetTransform(test_dataset, transform=NORMAL_TRANSFORM)
    model = train_network(model, train, epochs, batch_size)
    preds, actuals = test_network(model, test)
    return preds, actuals
