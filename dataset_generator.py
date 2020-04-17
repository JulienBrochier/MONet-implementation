import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from matplotlib import pyplot as plt

class Data():

    def __init__(self,data_dir):
        list_ds = tf.data.Dataset.list_files("*.png")
        self.images_ds = list_ds.map(self.parse_image)

    def parse_image(self,filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        return image

    def show(self,image):
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()


