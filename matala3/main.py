from models.super_resolution import create_model
from keras.layers import Input

# create model
input_shape = [32, 32, 1]
scale = 3

sr_model = create_model(input_shape, scale)

# pre process data
