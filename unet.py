import tensorflow as tf
import tensorflow_probability as tfp

class Unet(tf.keras.Model):
  def __init__ (self,input_size):
    super(Unet, self).__init__()
    self.input_size = input_size
    self.encoded_size = 16

  def predict(self, inp):

    skips=[]
    # Downsampling
    x = self.get_conv(64, 3, apply_batchnorm=False)(inp)
    x = self.get_conv(64, 3)(x)
    skips.append(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    for i in range(2,5):
        x = self.get_conv(64*i,3)(x)
        x = self.get_conv(64*2**i,3)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # Non-skip connection
    x = self.get_conv(1024, 3)(x)
    x = self.get_conv(1024, 3)(x)

    # Upsampling
    for i in range(1,5):
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skips[5-i-1]])
        x = self.get_conv(64, 3, apply_batchnorm=False)(x)
        x = self.get_conv(64*2**(5-i), 3)(x)

    x = tf.keras.layers.Conv2D(1, 1, strides=1)(x)

    return x


  def get_conv(self,filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                kernel_initializer=initializer, use_bias=False)) 
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result
