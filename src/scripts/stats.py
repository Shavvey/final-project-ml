import network.cnn_train_test as ctt
import network.cnn as cnn
import sklearn.metrics as metrics
import matplotlib.pyplot as pp
import data.dataset as dataset
import data.sub_transform as st
import torchvision.transforms as transforms


def confusion_matrix_rgb_cnn():
    model = cnn.CNN()
    train_dataset, test_dataset = ctt.train_test_split([0.8, 0.2], ctt.BASE_TRANSFORM)
    mean, std = dataset.calc_mean_std(train_dataset)
    print(mean, std)
    # apply normalization to train and test, after computing normal and std using train
    NORMAL_TRANSFORM = transforms.Compose([transforms.Normalize(mean, std)])
    train = st.SubsetTransform(train_dataset, transform=NORMAL_TRANSFORM)
    test = st.SubsetTransform(test_dataset, transform=NORMAL_TRANSFORM)
    model = ctt.train_network(model, train, 5, 24)
    preds, actuals = ctt.test_network(model, test)
    cm = metrics.confusion_matrix(actuals, preds)
    disp = metrics.ConfusionMatrixDisplay(cm)
    disp.plot()
    pp.show()


def confusion_matrix(preds, actuals):
    cm = metrics.confusion_matrix(actuals, preds)
    disp = metrics.ConfusionMatrixDisplay(cm)
    disp.plot()
    pp.show()


def classification_report(preds, actuals):
    r = metrics.classification_report(actuals, preds)
    print(r)

