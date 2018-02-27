import os

import numpy as np
from keras.layers import Conv2D, UpSampling2D
from keras.layers import InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from keras.callbacks import ModelCheckpoint
import pickle

# change the path to your train pictures file
train_path = './Train/';

X = []
for filename in os.listdir(train_path):
    X.append(img_to_array(load_img(train_path + filename)))
X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.9 * len(X))
print(split)
Xtrain = X[:split]
Xtrain = 1.0 / 255 * Xtrain

model = Sequential()
model.add(InputLayer(input_shape=(96, 96, 1)))
model.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(24, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(48, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(96, (3, 3), activation='sigmoid', padding='same'))
model.add(Conv2D(96, (3, 3), activation='sigmoid', padding='same', strides=2))
model.add(Conv2D(192, (3, 3), activation='sigmoid', padding='same'))
model.add(Conv2D(96, (3, 3), activation='sigmoid', padding='same'))
model.add(Conv2D(48, (3, 3), activation='sigmoid', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(24, (3, 3), activation='sigmoid', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(12, (3, 3), activation='sigmoid', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Image transformer
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True)

# Generate training data


batch_size = 128


def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)


print(batch_size * split)

model_path_to_save = "models_checkpoint/color.hdf5"

checkpointer = ModelCheckpoint(filepath=model_path_to_save,
                               verbose=1)

history = model.fit_generator(image_a_b_gen(batch_size), epochs=15, steps_per_epoch=split, callbacks=[checkpointer])

with open('models_checkpoint/history/color', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)  # steps per epoc - whole dataset / batch_size

Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

color_me = []
test_path = './Test/'
for filename in os.listdir(test_path):
    color_me.append(img_to_array(load_img(test_path + filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((96, 96, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    imsave("result/img_" + str(i) + ".png", lab2rgb(cur))
