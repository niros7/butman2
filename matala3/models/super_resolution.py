import keras.optimizers as optimizers
from keras import backend as K
from keras.layers import add, Input, PReLU, Dropout, concatenate
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.regularizers import l2

def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def create_conv2d(prev_layer, filters_size, kernel_size, dropout_rate):
    layer = Convolution2D(filters_size, kernel_size,
                          padding='same', use_bias=True, kernel_regularizer=l2(0.0001))(prev_layer)
    layer = PReLU()(layer)
    if dropout_rate is not 0:
        layer = Dropout(dropout_rate)(layer)
    return layer


def create_model(input_shape, scale):
    channels = scale * scale

    h, w = input_shape[0], input_shape[1]
    y_img_input = Input((h, w, 1))
    multi_channel_bicubic_input = Input((h, w, channels))

    filters_size = 96
    dropout_rate = 0.8
    feature_ext_layers = []
    input_to_next_layer = y_img_input

    for x in range(0, 6):
        input_to_next_layer = create_conv2d(input_to_next_layer, filters_size, (3, 3), dropout_rate)
        feature_ext_layers.append(input_to_next_layer)
        filters_size -= 9

    last_extraction_layer = create_conv2d(feature_ext_layers[-1], 32, (3, 3), dropout_rate)
    feature_ext_layers.append(last_extraction_layer)

    extractions_concat = concatenate(feature_ext_layers)

    a1 = create_conv2d(extractions_concat, 430, (1, 1), dropout_rate)
    b1 = create_conv2d(extractions_concat, 430, (1, 1), dropout_rate)
    b2 = create_conv2d(b1, 32, (3, 3), dropout_rate)

    reconstruction_concat = concatenate([a1, b2])

    l = create_conv2d(reconstruction_concat, channels, (1, 1), 0)

    output = add([l, multi_channel_bicubic_input])

    model = Model([y_img_input, multi_channel_bicubic_input], output)
    adam = optimizers.Adam(beta_1=0.1, beta_2=0.1)
    model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss])
    return model
