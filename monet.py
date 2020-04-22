#https://www.tensorflow.org/guide/keras/custom_layers_and_models

import tensorflow as tf
import tensorflow_probability as tfp
from unet import Unet
from vae import Vae


class Monet(tf.keras.Model):
  def __init__ (self,input_size):
    super(Monet, self).__init__()
    self.input_size = input_size
    self.encoded_size = 16
    self.unet = Unet(input_size)
    self.vae = Vae(input_size)
    self.beta = 0.5
    self.gamma = 0.5

  def call(self,image):
    model_input = image
    s0 = tf.ones([1,self.input_size,self.input_size,1], dtype=tf.dtypes.float32)
    log_sk=tf.math.log(s0)
    i=0
    """     while i<5 :   # will be replaced by something like "while sk>0"
      mask = self.unet.call(model_input)
      x = tf.keras.layers.concatenate([mask,image])
      [approx_posterior,output] = self.vae.call(x)

      model_input=
      i+=1 """

    mask = self.unet.call(model_input,log_sk)
    x = tf.keras.layers.concatenate([mask,image])
    [approx_posterior,output] = self.vae.call(x)
    return [mask, approx_posterior, output]

  def train(self,dataset):
    #for element in dataset.as_numpy_iterator():
    #    

    return

  def compute_loss(self,image):
    [mask, approx_posterior, output]=self.call(image)




 