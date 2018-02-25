from PIL import Image
import os, os.path
import numpy as np
import keras


def load_images(dir_path, files_ext, label, number_of_classes):
    x = []
    valid_extensions = files_ext
    for f in os.listdir(dir_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_extensions:
            continue
        img = Image.open(os.path.join(dir_path, f))
        x.append(np.array(img))
        img.close()

    x = np.array(x)
    x = x.astype('float32')
    x /= 255
    y = np.empty(x.shape[0])
    y.fill(label)

    y = keras.utils.to_categorical(y, number_of_classes)

    return (x, y)
