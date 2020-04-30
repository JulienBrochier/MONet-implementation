import tensorflow as tf
import tensorflow_probability as tfp

class Unet(tf.keras.layers.Layer):
  def __init__ (self,input_width, input_channels):
    super(Unet, self).__init__()
    self.input_width = input_width
    self.input_channels = input_channels

  def call(self, inp, log_sk):
    skips=[]
    x = tf.keras.layers.Concatenate()([inp,log_sk])
    # Downsampling
    x = self.get_conv(64, 3, self.input_channels+1, apply_batchnorm=False)(x)
    x = self.get_conv(64, 3, 64)(x)
    skips.append(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    for i in range(2,5):
        x = self.get_conv(64*i, 3, 64*(i-1))(x)
        x = self.get_conv(64*2**i, 3, 64*i)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
    # Non-skip connection
    x = self.get_conv(1024, 3, 512)(x)
    x = self.get_conv(1024, 3, 1024)(x)
    # Upsampling
    for i in range(1,5):
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skips[5-i-1]])
        x = self.get_conv(64*2**(5-i), 3, 64*2**(5-i+1), apply_batchnorm=False)(x)
        x = self.get_conv(64*2**(5-i), 3, 64*2**(5-i))(x)
    x = tf.keras.layers.Conv2D(2, 1, strides=1)(x)
    # Reshape to apply Softmax on width and height
    ak = tf.keras.layers.Softmax(axis=-1)(x)
    ak = tf.split(ak,2,-1)[0]
    return self.compute_new_scope_and_mask(ak, log_sk)

  def get_conv(self,filters, size, nb_input_features, apply_batchnorm=True):
    stdev = tf.math.sqrt(2/(size**2*nb_input_features))
    initializer = tf.keras.initializers.TruncatedNormal(0., stdev)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                kernel_initializer=initializer, use_bias=False)) 
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

  def compute_new_scope_and_mask(self,ak,log_sk):
    # scope : part of the image that has already been masked
    log_mk = tf.math.log(ak)+log_sk
    log_sk = tf.math.log(1-ak)+log_sk
    return log_mk, log_sk
