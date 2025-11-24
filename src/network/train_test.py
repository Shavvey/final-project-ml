import torch
import torch.optim as optim

# use gpu CUDA API if able, otherwise just use the regular cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
