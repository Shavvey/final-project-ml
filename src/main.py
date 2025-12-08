import scripts.stats as stats
import network.resnet_train_test as tt



def main():
    preds, actuals = tt.train_test([0.8, 0.2], 3, 6)
    stats.classification_report(preds, actuals)
    stats.confusion_matrix(preds, actuals)

if __name__ == "__main__":
    main()
