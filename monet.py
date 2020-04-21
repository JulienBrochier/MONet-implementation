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

  def predict(self,image):
    model_input = image  
    mask = self.unet.predict(model_input)
    x = tf.keras.layers.concatenate([mask,image])
    output = self.vae.predict(x)

    return output


 