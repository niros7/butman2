from keras.models import Model
from keras.layers import Add, Input, PReLU, Dropout, Concatenate, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.optimizers as optimizers
from keras.regularizers import l2


# input_shape = [32, 32]
# output_shpae = [92, 92]


def create_conv2d(prev_layer, filters_size, kernel_size, dropout_rate):
    layer = Convolution2D(filters_size, kernel_size, activation=PReLU("he_normal"),
                          padding='same', use_bias=True, kernel_regularizer=l2(0.0001))(prev_layer)
    layer = Dropout(dropout_rate)(layer)
    return layer


def create_model(input_shape, output_shape):

    bicubic = Lambda(lambda img: bicubic(img, scale))(input_shape)

    dropout_rate = 0.8
    feature_ext_layers = []
    input_to_next_layer = input_shape

    for x in range(0, 5):
        input_to_next_layer = create_conv2d(input_to_next_layer, 96, (3, 3), dropout_rate)
        feature_ext_layers.append(input_to_next_layer)

    last_extraction_layer = create_conv2d(feature_ext_layers[-1], 32, (3, 3), dropout_rate)
    feature_ext_layers.append(last_extraction_layer)

    extractions_concat = Concatenate(feature_ext_layers)

    a1 = create_conv2d(extractions_concat, 64, (1, 1), dropout_rate)
    b1 = create_conv2d(extractions_concat, 32, (1, 1), dropout_rate)
    b2 = create_conv2d(b1, 32, (3, 3), dropout_rate)

    reconstruction_concat = Concatenate([a1, b1])

    l =


    # model = Model(init, decoded)
    # adam = optimizers.Adam(lr=1e-3)
    # model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
    # return model
