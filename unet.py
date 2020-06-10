# U-Net: Convolutional Networks for BiomedicalImage Segmentation
# https://github.com/tensorflow/tensorflow/issues/29073
# https://www.geeksforgeeks.org/python-iterate-multiple-lists-simultaneously/

import tensorflow as tf
import tensorflow_probability as tfp
import time

class Unet(tf.keras.layers.Layer):
  def __init__ (self,input_width, input_channels,batch_size):
    super(Unet, self).__init__()
    self.input_width = input_width
    self.input_channels = input_channels
    self.batch_size = batch_size
    self.downsampling_layers = []
    self.downsampling_layers_output_channels = [[64,64],[128,128],[256,256],[512,512]]
    self.downsampling_layers_input_channels = [[self.input_channels+1,64],[64,128],[128,256],[256,512]]
    self.non_skip_layers = []
    self.non_skip_layers_output_channels = [[1024,1024]]
    self.non_skip_layers_input_channels = [[512,1024]]
    self.upsampling_layers = []
    self.upsampling_layers_output_channels = [[512,512],[256,256],[128,128],[64,64]]
    self.upsampling_layers_input_channels = [[1024,512],[512,256],[256,128],[128,64]]
    self.final_layers = []

    #first iteration : apply_batchnorm=false
    for output_channels, input_channels in zip(self.downsampling_layers_output_channels, self.downsampling_layers_input_channels) :
      self.downsampling_layers.append(self.get_conv(output_channels[0], 3, input_channels[0]))
      self.downsampling_layers.append(self.get_conv(output_channels[1], 3, input_channels[1]))
      self.downsampling_layers.append(tf.keras.layers.MaxPooling2D((2,2)))

    for output_channels, input_channels in zip(self.non_skip_layers_output_channels, self.non_skip_layers_input_channels) :
      self.non_skip_layers.append(self.get_conv(output_channels[0], 3, input_channels[0]))
      self.non_skip_layers.append(self.get_conv(output_channels[1], 3, input_channels[1]))

    for output_channels, input_channels in zip(self.upsampling_layers_output_channels, self.upsampling_layers_input_channels) :
      self.upsampling_layers.append(tf.keras.layers.UpSampling2D(size=(2, 2)))
      self.upsampling_layers.append(tf.keras.layers.Concatenate())
      self.upsampling_layers.append(self.get_conv(output_channels[0], 3, input_channels[0], apply_batchnorm=False))
      self.upsampling_layers.append(self.get_conv(output_channels[1], 3, input_channels[1], apply_batchnorm=False))

    self.final_layers.append(tf.keras.layers.Conv2D(2, 1, strides=1, activation="sigmoid"))
    self.final_layers.append(tf.keras.layers.Softmax(axis=-1))

  def call(self, inp, log_sk):
    t_unet = time.time()
    skips=[]
    x = tf.keras.layers.Concatenate()([inp,log_sk])
    # Downsampling
    for conv in self.downsampling_layers :
      if isinstance(conv, tf.keras.layers.MaxPool2D):
        skips.append(x)
        x = conv(x)
      else :
        for layer in conv :
          x = layer(x)
    skips.reverse()
    # Non-skip connection
    for conv in self.non_skip_layers :
      for layer in conv :
        x= layer(x)
    # Upsampling
    i=0
    for conv in self.upsampling_layers :
      if isinstance(conv, tf.keras.layers.UpSampling2D):
        x = conv(x)
      elif isinstance(conv, tf.keras.layers.Concatenate):
        x = conv([x,skips[i]])
        i+=1
      else:
        for layer in conv :
          x= layer(x)
    # Final Conv & Softmax
    for conv in self.final_layers :
      x = conv(x)
    ak = tf.split(x,2,-1)[0]
    output = self.compute_new_scope_and_mask(ak, log_sk)
    print("Forward pass Unet : {}sec".format(time.time()-t_unet))
    return output

  def get_conv(self,output_channels, filters_shape, nb_input_features, apply_batchnorm=True):
    stdev = tf.math.sqrt(2/(filters_shape**2*nb_input_features))
    initializer = tf.keras.initializers.TruncatedNormal(0., stdev)
    result = []
    result.append(
        tf.keras.layers.Conv2D(output_channels, filters_shape, strides=1, padding='same',
                                kernel_initializer=initializer, use_bias=False)) 
    if apply_batchnorm:
        result.append(tf.keras.layers.BatchNormalization())
    result.append(tf.keras.layers.LeakyReLU())
    return result

  def compute_new_scope_and_mask(self,ak,log_sk):
    # scope : part of the image that has already been masked
    log_mk = tf.math.log(ak)+log_sk
    log_sk = tf.math.log(1-ak)+log_sk
    return log_mk, log_sk

