# https://www.tensorflow.org/guide/data_performance
# https://code.i-harness.com/fr/docs/tensorflow~guide/performance/datasets_performance

import tensorflow as tf
from matplotlib import pyplot as plt
import time

class Data():

    def __init__(self,data_dir,batch_size,image_width,input_channels):
        self.image_width = image_width
        self.input_channels = input_channels
        self.list_ds = tf.data.Dataset.list_files(data_dir+"*.png")
        self.images_ds = self.list_ds.map(self.parse_image).batch(batch_size).cache()

    def parse_image(self,filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.image_width, self.image_width])
        image = tf.reshape(image,[self.image_width,self.image_width,self.input_channels])
        return image

    def show(self,image):
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()

