import network.resnet_train_test as ttres
import network.resnet as resnet
import torch


def main():
    train_dataset, test_dataset = ttres.train_test_split(
        [0.8, 0.2], ttres.IMAGE_TRANSFORM
    )
    model = resnet.ResNet101(2)
    model = ttres.train_network(model, train_dataset, 3, 12)
    ttres.test_network(model, test_dataset)
    name = input("Name the model:")
    torch.save(model.state_dict(), "models/" + name + ".pth")


if __name__ == "__main__":
    main()
