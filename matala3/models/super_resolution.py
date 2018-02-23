from keras.models import Model
from keras.layers import Add, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.optimizers as optimizers


# input_shape = [32, 32]
# output_shpae = [92, 92]

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def create_model():
    init = Input((96, 96, 1))

    c1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(init)
    c1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(c1)

    x = MaxPooling2D((2, 2))(c1)

    c2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    c2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(c2)

    x = MaxPooling2D((2, 2))(c2)

    c3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D()(c3)

    c2_2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    c2_2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(c2_2)

    m1 = Add()([c2, c2_2])
    m1 = UpSampling2D()(m1)

    c1_2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(m1)
    c1_2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(c1_2)

    m2 = Add()([c1, c1_2])

    decoded = Convolution2D(1, (5, 5), activation='linear', padding='same')(m2)

    model = Model(init, decoded)
    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
    return model
