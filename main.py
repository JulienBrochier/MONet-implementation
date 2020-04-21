#
# https://www.tensorflow.org/guide/keras/functional
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
# https://www.tensorflow.org/tutorials/generative/cvae
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
# 

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_probability as tfp

import os
import time
from matplotlib import pyplot as plt
from IPython import display
from dataset_generator import Data
from monet import Monet

path = os.getcwd()
data_dir = os.path.join(path,"data")
data = Data(data_dir)
dataset = data.images_ds
l=[]
for element in dataset.as_numpy_iterator():
    l.append(element)

input_size = 128

#print(tf.shape(l[0]))
monet = Monet(input_size)
#print(monet.predict(l[0]))


#monet.summary()



