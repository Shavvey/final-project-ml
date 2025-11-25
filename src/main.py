import network.cnn_train_test as ttcnn
import network.cnn as cnn


def main():
    # make train and test data with 80:20 ratio
    # apply usual transform
    train_dataset, test_dataset = ttcnn.train_test_split(
        [0.8, 0.2], ttcnn.IMAGE_TRANSFORM
    )
    # instantiate model
    model = cnn.CNN()
    # train the model
    model = ttcnn.train_network(model, train_dataset, 1, 24)
    # test the model
    ttcnn.test_network(model, test_dataset)


if __name__ == "__main__":
    main()
