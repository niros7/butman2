from models.super_resolution import create_model
from keras.layers import Input




sr_model = create_model(Input((32, 32, 3)), (96, 96, 3))
