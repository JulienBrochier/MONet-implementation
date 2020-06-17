#
# https://www.tensorflow.org/guide/keras/functional
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
# https://www.tensorflow.org/tutorials/generative/cvae
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
# 

import tensorflow as tf
import tensorflow_probability as tfp
import os
from termcolor import colored
import datetime
import time
from matplotlib import pyplot as plt
from IPython import display
from dataset_generator import Data
from monet import Monet

os.system('color')

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

def show_evolution(save_path, dataset, input_channels, batch_size):
    """
     Show masks and images obtained before and after training
    """
    trained_model = Monet(image_width, input_channels=input_channels, nb_scopes=5, batch_size=batch_size)
    trained_model.load_weights(save_path)
    untrained_model = Monet(image_width, input_channels=1, nb_scopes=5, batch_size=1)
    test_batch = list(ds_test.take(1).as_numpy_iterator())[0]
    raw_img = tf.slice(test_batch,[0,0,0,0],[1,-1,-1,-1])
    untrained_output = untrained_model(raw_img)
    monet.vae.batch_size=1
    trained_ouput = trained_model(raw_img)
    monet.vae.batch_size=batch_size

    figure_titles = ["unet_mask","vae_mask","reconstructed_img"]
    column_titles = ["untrained_model","trained_model"]
    for k in range(3):
        fig = plt.figure(figsize=(trained_model.nb_scopes+1,2))
        fig.suptitle(figure_titles[k])
        for i,imgs in enumerate([untrained_output[k], trained_ouput[k]]):
            for j in range(trained_model.nb_scopes):
                img = tf.reshape(imgs[j],[128,128])
                plt.subplot(trained_model.nb_scopes,2,(2*j)+i+1)
                plt.axis('off')
                plt.title(column_titles[i]+" k="+str(j), fontsize=10)
                plt.imshow(img, cmap='gray')
        img = tf.reshape(raw_img,[128,128])
        plt.subplot(trained_model.nb_scopes,2,(2*trained_model.nb_scopes))
        plt.title("vanilla", fontsize=10)
        plt.imshow(img)
    plt.show()

# Image metadata
image_width = 64
input_channels = 1

# Create Dataset
batch_size = 20
path = os.getcwd()
data_dir = os.path.join(path,"Data")
t0 = time.time()
ds_train = Data(data_dir+"/train",batch_size,image_width,input_channels).images_ds
ds_test = Data(data_dir+"/test",batch_size,image_width,input_channels).images_ds
print("Datasets created in : {} sec".format(time.time()-t0))

# Path to save weights
save_path = './checkpoints/new_checkpoint'
# For Tensorboard
log_dir = 'logs/'

# Create the model
t0 = time.time()
monet = Monet(image_width, input_channels=1, nb_scopes=5, batch_size=batch_size)
print("Model created in : {} sec".format(time.time()-t0))
# Training
L1,L2,L3 = monet.fit(ds_train.prefetch(tf.data.experimental.AUTOTUNE), log_dir, save_path=save_path)
