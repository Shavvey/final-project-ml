import network.cnn_train_test as ttcnn
import network.cnn as cnn
import torch


def main():
    # make train and test data with 80:20 ratio
    # apply usual transform
    train_dataset, test_dataset = ttcnn.train_test_split(
        [0.8, 0.2], ttcnn.IMAGE_TRANSFORM
    )
    model = cnn.CNN()
    # NOTE: saving state dict as local vars will return invalid keys
    model.load_state_dict(torch.load("./models/CNN_model1.pth", weights_only=True))
    ttcnn.test_network(model, test_dataset, test_log=True)


if __name__ == "__main__":
    main()
