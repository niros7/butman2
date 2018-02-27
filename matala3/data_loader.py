from PIL import Image
import os, os.path
import numpy as np
import keras


def load_images(dir_path, files_ext):
    images = []
    valid_extensions = files_ext
    for f in os.listdir(dir_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_extensions:
            continue
        img = Image.open(os.path.join(dir_path, f))
        images.append(np.array(img))
        img.close()

    return images

def load_train_data(x_dir_path, y_dir_path, files_ext):
    x_images = load_images(x_dir_path, files_ext)
    y_images = load_images(y_dir_path, files_ext)

    y_and_quad_images = []

    # for img in x_images:
