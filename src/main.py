import network.resnet_train_test as tt
import scripts.stats as stats



def main():
    preds, actuals = tt.train_test([0.8, 0.2], 5, 24)
    stats.classification_report(preds, actuals)
    stats.confusion_matrix(preds, actuals)

if __name__ == "__main__":
    main()
