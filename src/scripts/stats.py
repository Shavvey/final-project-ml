import network.cnn_train_test as ctt
import network.cnn as cnn
import sklearn.metrics as metrics
import matplotlib.pyplot as pp


def confusion_matrix_rgb_cnn():
    model = cnn.CNN()
    train_dataset, test_dataset = ctt.train_test_split([0.8, 0.2], ctt.IMAGE_TRANSFORM)
    model = ctt.train_network(model, train_dataset, 5, 24)
    preds, actuals = ctt.test_network(model, test_dataset)
    cm = metrics.confusion_matrix(actuals, preds)
    disp = metrics.ConfusionMatrixDisplay(cm)
    disp.plot()
    pp.show()
    
