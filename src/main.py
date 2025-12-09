import network.resnet_train_test as tt



def main():
    tt.train_test([0.8, 0.2], 5, 24)

if __name__ == "__main__":
    main()
