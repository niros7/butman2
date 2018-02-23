from data import Data
from plotLoss import PlotLosses
from keras.callbacks import ModelCheckpoint

class cifarTrainer(object):
    def trainModel(self, model, classes, batch_size, epochs, model_name):
        data = Data(classes).Data
        x_train = data[0][0]
        y_train = data[0][1]
        x_train = x_train.astype('float32')
        x_train /= 255
        plot_losses = PlotLosses()

        model_path_to_save = "models_checkpoint/%s.hdf5" % model_name

        checkpointer = ModelCheckpoint(filepath=model_path_to_save,
                                       verbose=1,
                                       save_best_only=True)

        model.fit(x_train, y_train,  # this is our training examples & labels
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1,  # this parameter control the % of train data used for validation
                  shuffle=True,
                  callbacks=[plot_losses, checkpointer])  # thi
        return model


    def evalModel(self, model, classes):
        data = Data(classes).Data
        x_test = data[1][0]
        y_test = data[1][1]
        x_test = x_test.astype('float32')
        x_test /= 255
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('==> Test loss:', scores[0])
        print('==> Test accuracy:', scores[1])
