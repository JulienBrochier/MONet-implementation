import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize

class Data():

    def __init__(self,data_dir):
        files = []
        onlyfiles = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        for _file in onlyfiles:
            files.append(_file)
        print(files[0])
        
        image_width = 190
        image_height = 256

        channels = 3
        nb_classes = 1

        np_dataset = np.ndarray(shape=(len(files), image_width, image_height, channels),
                            dtype=np.float32)

        i=0
        for _file in files:
            img = load_img(data_dir + "/" + _file)
            img.thumbnail((image_width, image_height))
            # Convert to Numpy Array
            x = img_to_array(img)
            # Normalize
            x = (x - 128.0) / 128.0
            # Resize
            x = resize(x, (190, 256))
            np_dataset[i] = x
            i+=1

        self.dataset = tf.data.Dataset.from_tensor_slices(np_dataset)

""" list_ds = tf.data.Dataset.list_files(data_dir+"/**.png")
for f in list_ds.take(5):
    print(f.numpy())
## ... """