import torch
import torch.optim as optim
import torch.nn as nn

from network.nn import CNN

# use gpu CUDA API if able, otherwise just use the regular cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

# create model
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epochs: int):
    for epoch in range(epochs):
        for i, data in
