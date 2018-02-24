from models.secondCnn import secondCnn
from models.fourthCnn import fourthCnn
from cifarTrainer import cifarTrainer
from models.thirdCnn import thirdCnn
import tensorflow as tf
from keras.models import load_model

# tensorflow configuration for gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config=config)

trainer = cifarTrainer()

input_shape = [32, 32, 3]
output_shape = 10

classes = [2, 3, 5, 6, 7]
batch_size = 128
epochs = 30

model = load_model('models_checkpoint/fourth.hdf5')
# model = load_model('models_checkpoint/vanilla.hdf5')
# model = load_model('models_checkpoint/second.hdf5')
# model = load_model('models_checkpoint/third.hdf5')
# model = secondCnn(input_shape, output_shape).model
# model = thirdCnn(input_shape, output_shape).model
# model = fourthCnn(input_shape, output_shape).model
# model_name = "vanilla"
# model_name = "second"
# model_name = "third"
model_name = "fourth"

# model = trainer.trainModel(model, classes, batch_size, epochs, model_name)
trainer.evalModel(model, classes)
