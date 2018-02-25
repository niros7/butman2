from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import load_model, Model


def transferedModel():
    model = load_model('trainedModels/fourth.hdf5')
    x = model.output
    x = Dense(96, activation="relu", name="dense_11")(x)
    x = Dropout(0.5, name="drop")(x)
    x = Dense(96, activation="relu", name="dense_22")(x)

    predictions = Dense(11, activation="softmax", name="dense_33")(x)
    model_final = Model(input=model.input, output=predictions)

    # compile the model
    model_final.compile(loss='categorical_crossentropy',
                        optimizer="adam",
                        metrics=['accuracy'])

    return model_final
