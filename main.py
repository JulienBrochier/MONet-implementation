#
# https://www.tensorflow.org/guide/keras/functional
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
# https://www.tensorflow.org/tutorials/generative/cvae
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
# 
import tensorflow as tf
import tensorflow_probability as tfp

import os
import time
from matplotlib import pyplot as plt
from IPython import display
from dataset_generator import Data
from monet import Monet

def plot_loss(L1,L2,L3):
    plt.figure()
    # L1
    plt.subplot(131)
    plt.plot(L1)
    plt.title("L1")
    # L2
    plt.subplot(132)
    plt.plot(L2)
    plt.title("L2")
    # L3
    plt.subplot(133)
    plt.plot(L3)
    plt.title("L3")
    plt.show()

path = os.getcwd()
data_dir = os.path.join(path,"data")
data = Data(data_dir)
dataset = data.images_ds

input_width = 128

monet = Monet(input_width, input_channels=1, nb_scopes=5)
L1,L2,L3 = monet.train(dataset)
plot_loss(L1,L2,L3)




