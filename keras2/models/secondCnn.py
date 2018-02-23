from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.regularizers import l2


class secondCnn(object):
    def __init__(self, input_shape, output_shape):
        model = Sequential()

        model.add(Conv2D(32, (2, 2), padding='same', input_shape=input_shape, kernel_regularizer=l2(0.01)))
        model.add(Conv2D(64, (2, 2), padding='same', kernel_regularizer=l2(0.01)))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(Dropout(0.4))

        model.add(Conv2D(64, (2, 2), padding='same', kernel_regularizer=l2(0.01)))
        model.add(Conv2D(92, (2, 2), padding='same', kernel_regularizer=l2(0.01)))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(Dense(output_shape))
        model.add(Activation('softmax'))

        opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model = model
