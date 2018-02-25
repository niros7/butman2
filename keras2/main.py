from cifar_loader import cifar_loader
from models.secondCnn import secondCnn
from models.fourthCnn import fourthCnn
from trainer import trainer
from models.thirdCnn import thirdCnn
from models.transferedLearningModel import transferedModel
import tensorflow as tf
from keras.models import load_model
from images_loader import load_images
import numpy as np


# tensorflow configuration for gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config=config)

input_shape = [32, 32, 3]
output_shape = 10

classes = [2, 3, 5, 6, 7]
batch_size = 128
epochs = 30


trainer = trainer()
cifar_data = cifar_loader(classes).Data


# model = load_model('models_checkpoint/fourth.hdf5')
# model = load_model('models_checkpoint/vanilla.hdf5')
# model = load_model('models_checkpoint/second.hdf5')
# model = load_model('models_checkpoint/third.hdf5')
# model = secondCnn(input_shape, output_shape).model
# model = thirdCnn(input_shape, output_shape).model
# model = fourthCnn(input_shape, output_shape).model
# model_name = "vanilla"
# model_name = "second"
# model_name = "third"
# model_name = "fourthx"

# x_train = cifar_data[0][0]
# y_train = cifar_data[0][1]
#
# x_eval = cifar_data[1][0]
# y_eval = cifar_data[1][1]

# model = trainer.trainModel(model, classes, batch_size, epochs, model_name, x_train, y_train)
# trainer.evalModel(model, classes, x_eval, y_eval)

model = transferedModel()
model_name = "transfered"

flowers_data = load_images("flowers images", [".jpg"], 11 - 1, 11)

# model = load_model('models_checkpoint/transfered.hdf5')

train_flowers_x = np.array(flowers_data[0][: int(len(flowers_data[0]) * .80)])
train_flowers_y = np.array(flowers_data[1][: int(len(flowers_data[1]) * .80)])
eval_flowers_x = np.array(flowers_data[0][int(len(flowers_data[0]) * .80) : int(len(flowers_data[0]) * 1)])
eval_flowers_y = np.array(flowers_data[1][int(len(flowers_data[1]) * .80) : int(len(flowers_data[1]) * 1)])


train_x = np.vstack((train_flowers_x, cifar_data[0][0]))
train_y = np.vstack((train_flowers_y, cifar_data[0][1]))
eval_x = np.vstack((eval_flowers_x, cifar_data[1][0]))
eval_y = np.vstack((eval_flowers_y, cifar_data[1][1]))


model = trainer.trainModel(model, 128, 10, model_name, train_x, train_y)
trainer.evalModel(model, eval_x, eval_y)



















