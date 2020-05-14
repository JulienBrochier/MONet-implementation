# https://www.tensorflow.org/guide/data_performance
# https://code.i-harness.com/fr/docs/tensorflow~guide/performance/datasets_performance

import tensorflow as tf
from matplotlib import pyplot as plt
import time

class Data():

    def __init__(self,data_dir,batch_size):
        self.list_ds = tf.data.Dataset.list_files("*.png")
        self.images_ds = self.list_ds.map(self.parse_image).cache()

    def parse_image(self,filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [128, 128])
        image = tf.reshape(image,[1,128,128,1])
        return image

    def show(self,image):
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()

