import unittest
import data.dataset as dataset
import torch
import network.cnn as cnn
import network.resnet as resnet
import network.cnn_train_test as tt
import network.resnet_train_test as rtt
import torchvision.transforms as transforms
import data.sub_transform as st


class TestLoadModel(unittest.TestCase):
    def test_load_model(self):
        model_path = "models/CNN_RGB_97.pth"
        model = cnn.CNN()
        # NOTE: don't store state dict into intermediate var
        model.load_state_dict(torch.load(model_path))
        # test the loaded model
        train_dataset, test_dataset = tt.train_test_split([0.8, 0.2], tt.BASE_TRANSFORM)
        mean, std = dataset.calc_mean_std(train_dataset)
        print(mean, std)
        # apply normalization to train and test, after computing normal and std using train
        NORMAL_TRANSFORM = transforms.Compose([transforms.Normalize(mean, std)])
        train = st.SubsetTransform(train_dataset, transform=NORMAL_TRANSFORM)
        test = st.SubsetTransform(test_dataset, transform=NORMAL_TRANSFORM)
        preds, actuals = tt.test_network(model, test)
