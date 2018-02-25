import numpy as np
import keras
from keras.datasets import cifar10


class cifar_loader(object):
    def __init__(self, classes):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = len(classes)
        if num_classes != 0:
            selected_classes = classes
            x = [ex for ex, ey in zip(x_train, y_train)
                 if ey in selected_classes]
            y = [ey for ex, ey in zip(x_train, y_train)
                 if ey in selected_classes]
            x_train = np.stack(x)
            y_train = np.stack(y).reshape(-1, 1)

            x = [ex for ex, ey in zip(x_test, y_test)
                 if ey in selected_classes]
            y = [ey for ex, ey in zip(x_test, y_test)
                 if ey in selected_classes]
            x_test = np.stack(x)
            y_test = np.stack(y).reshape(-1, 1)
            print('train\n', x_train.shape, y_train.shape)
            print('test\n', x_test.shape, y_test.shape)
        else:
            print('train\n', x_train.shape, y_train.shape)
            print('test\n', x_test.shape, y_test.shape)

            # convert labels to "one hot" vector

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        x_train = x_train.astype('float32')
        x_test = x_train.astype('float32')
        x_train /= 255
        x_test /= 255

        self.Data = (x_train, y_train), (x_test, y_test)
