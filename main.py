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

def show_evolution(save_path, dataset, input_channels, batch_size):
    trained_model = Monet(input_width, input_channels=input_channels, nb_scopes=5, batch_size=batch_size)
    trained_model.load_weights(save_path)
    untrained_model = Monet(input_width, input_channels=1, nb_scopes=5, batch_size=batch_size)

    "Compute the first image, before and after training"

    raw_img = list(dataset.take(1).as_numpy_iterator())[0]
    untrained_output = untrained_model(raw_img)
    trained_ouput = trained_model(raw_img)

    figure_titles = ["unet_mask","vae_mask","reconstructed_img"]
    column_titles = ["untrained_model","trained_model"]
    for k in range(3):
        fig = plt.figure(figsize=(trained_model.nb_scopes+1,2))
        fig.suptitle(figure_titles[k])
        for i,imgs in enumerate([untrained_output[k], trained_ouput[k]]):
            for j in range(trained_model.nb_scopes):
                #print(img)
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

    #print(untrained_model.compute_loss(raw_img))
    #print(trained_model.compute_loss(raw_img))


batch_size = 1
path = os.getcwd()
data_dir = os.path.join(path,"Data")
t0 = time.time()
ds = Data(data_dir,batch_size).images_ds
print("Dataset Creation : {} sec".format(time.time()-t0))

input_width = 128
input_channels = 1
save_path = './checkpoints/my_checkpoint'

t0 = time.time()
monet = Monet(input_width, input_channels=1, nb_scopes=5, batch_size=batch_size)
print("Model Creation : {} sec".format(time.time()-t0))
L1,L2,L3 = monet.fit(ds.prefetch(tf.data.experimental.AUTOTUNE), save_path=save_path)
#plot_loss(L1,L2,L3)

#show_evolution(save_path,dataset,input_channels,batch_size)


""" print("trainable variables :")
print("unet : {}".format(len(monet.unet.trainable_variables)))
print("vae : {}".format(len(monet.vae.trainable_variables)))
print("monet : {}".format(len(monet.trainable_variables))) """




