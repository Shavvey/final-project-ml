import network.cnn_train_test as ttcnn
import network.cnn as cnn
import torch


def main():
    train_dataset, test_dataset = ttcnn.train_test_split(
        [0.8, 0.2], ttcnn.IMAGE_TRANSFORM
    )
    model = cnn.CNN()
    model = ttcnn.train_network(model, train_dataset, 10, 24)
    ttcnn.test_network(model, test_dataset)
    name = input("Name the model:")
    torch.save(model.state_dict(), "models/" + name + ".pth")


if __name__ == "__main__":
    main()
