from models.super_resolution import create_model
from keras.layers import Input
from keras.preprocessing.image import img_to_array, load_img
import utils
import numpy as np
import os
from trainer import *
import datetime
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.io import imsave


def prepare_x_images(images):
    all_x_images_y = []
    all_x_images_bicubic = []

    for image in images:
        all_x_images_y.append(utils.convert_rgb_to_y(image))
        x = utils.resize_image_by_pil(image, 3)
        quad_image = np.zeros([32, 32, 9])
        utils.convert_to_multi_channel_image(quad_image, x, 3)
        all_x_images_bicubic.append(quad_image)

    return (all_x_images_y, all_x_images_bicubic)


def prepare_y_images(images):
    all_y_images_y = []

    for image in images:
        quad_image = np.zeros([32, 32, 9])
        utils.convert_to_multi_channel_image(quad_image, image, 3)
        all_y_images_y.append(quad_image)

    return all_y_images_y


#
# with open('models_checkpoint/history/sr_history', 'rb') as file_pi:
#     history = pickle.load(file_pi)
#
#
# print(history)

# def PSNRLoss(y_true, y_pred):
#     return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)
#
#
# model = load_model("models_checkpoint/sr", {"PSNRLoss": PSNRLoss})
# small_image = load_img("data/small/image_00001.jpg")
#
# y_image, quad_image = prepare_x_images([np.array(small_image)])
#
# x = [np.array(y_image[0]).reshape((1, 32, 32, 1)), np.array(quad_image[0]).reshape((1, 32, 32, 9))]
#
# y = model.predict(x)
# image = np.zeros(shape=[32 * 3, 32 * 3, 1])
# utils.convert_from_multi_channel_image(image, y[0], 3)
#
# utils.save_image("image.jpg", image)

def load_images(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        images.append(img_to_array(load_img(dir_path + filename)))
    return images


all_x_images = load_images("data/small/")
all_y_images = load_images("data/large/")
all_eval_x = load_images("data/small_eval/")
all_eval_y = load_images("data/large_eval/")

x_train, x_bicubic_train = prepare_x_images(all_x_images)
y_train = prepare_y_images(all_y_images)
x_eval, x_bicubic_eval = prepare_x_images(all_eval_x)
y_eval = prepare_y_images(all_eval_y)

x_eval = np.array(x_eval)
x_bicubic_eval = np.array(x_bicubic_eval)
y_eval = np.array(y_eval)

# create model
input_shape = [32, 32, 1]
scale = 3

sr_model = create_model(input_shape, scale)

batch_size = 5
itr = int(len(x_train) / batch_size)
print(itr)
history = []
for epoch in range(0, 10):
    print("epoch: %d" % epoch)
    for batch in range(0, itr):
        print("batch: %d" % batch)
        batch_start = batch * batch_size
        batch_end = batch_start + batch_size

        x = np.array(x_train[batch_start:batch_end])
        x_bicubic = np.array(x_bicubic_train[batch_start:batch_end])
        y = np.array(y_train[batch_start:batch_end])

        loss = sr_model.train_on_batch([x, x_bicubic], y)
        print(loss)
        history.append(loss)
    scores = sr_model.evaluate([x_eval, x_bicubic_eval], y_eval)
    print('==> Test loss:', scores)

sr_model.save("models_checkpoint/sr".format(datetime.datetime.now()))
with open('models_checkpoint/history/sr_history', 'wb') as file_pi:
    pickle.dump(history, file_pi)
