import network.resnet_train_test as tt
import scripts.stats as stats



def main():
    preds, actuals = tt.train_test([0.8, 0.2], 5, 24, "/content/drive/MyDrive/ML-Final-Project/Models/resnet2.pth")
    stats.classification_report(preds, actuals)
    stats.confusion_matrix(preds, actuals, "/content/drive/MyDrive/ML-Final-Project/Figs/resnet")

if __name__ == "__main__":
    main()
