from keras.callbacks import ModelCheckpoint
import pickle
from datetime import datetime

class trainer(object):
    def trainModel(self, model, batch_size, epochs, model_name, x, y):
        model_path_to_save = "models_checkpoint/%s.hdf5" % model_name

        checkpointer = ModelCheckpoint(filepath=model_path_to_save,
                                       verbose=1,
                                       save_best_only=True)

        history = model.fit(x, y,  # this is our training examples & labels
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,  # this parameter control the % of train data used for validation
                            shuffle=True,
                            callbacks=[checkpointer])

        with open('models_checkpoint/history/%s-%s' % (model_name, datetime.now()) , 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        return model

    def evalModel(self, model, x, y):
        scores = model.evaluate(x, y, verbose=1)
        print('==> Test loss:', scores[0])
        print('==> Test accuracy:', scores[1])
