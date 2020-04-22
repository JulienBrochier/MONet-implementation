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

print(tf.shape(l[0]))
monet = Monet(input_size)
monet.call(l[0])

#print(monet.vae.trainable_variables)
#print(monet.unet.log_sk)
#print(monet.layers)
#print(monet.unet.layers)





#negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
#monet.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
#            loss=negative_log_likelihood)
#monet.fit(dataset, epochs=10)
#monet.summary()



