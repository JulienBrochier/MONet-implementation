from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_probability as tfp

import os
import time
from matplotlib import pyplot as plt
from IPython import display

OUTPUT_CHANNELS = 3

input_size = 256


def get_conv(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                            kernel_initializer=initializer, use_bias=False)) 
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def get_uncompiled_Attention_Network(input_size):
  inputs = tf.keras.layers.Input(shape=[input_size,input_size,3])
  x = inputs

  # Downsampling through the model
  x = get_conv(64, 3, apply_batchnorm=False)(x)
  x = get_conv(64, 3)(x)
  s1 = x
  x = tf.keras.layers.MaxPooling2D((2,2))(x)

  x = get_conv(128, 3)(x)
  x = get_conv(128, 3)(x)
  s2 = x
  x = tf.keras.layers.MaxPooling2D((2,2))(x)

  x = get_conv(256, 3)(x)
  x = get_conv(256, 3)(x)
  s3 = x
  x = tf.keras.layers.MaxPooling2D((2,2))(x)

  x = get_conv(512, 3)(x)
  x = get_conv(512, 3)(x)
  s4 = x
  x = tf.keras.layers.MaxPooling2D((2,2))(x)

  # Non-skip connection

  x = get_conv(1024, 3)(x)
  x = get_conv(1024, 3)(x)

  # Upsampling and establishing the skip connections

  x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
  x = tf.keras.layers.Concatenate()([x, s4])

  x = get_conv(512, 3, apply_batchnorm=False)(x)
  x = get_conv(512, 3)(x)

  x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
  x = tf.keras.layers.Concatenate()([x, s3])

  x = get_conv(256, 3, apply_batchnorm=False)(x)
  x = get_conv(256, 3)(x)

  x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
  x = tf.keras.layers.Concatenate()([x, s2])

  x = get_conv(128, 3, apply_batchnorm=False)(x)
  x = get_conv(128, 3)(x)

  x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
  x = tf.keras.layers.Concatenate()([x, s1])

  x = get_conv(64, 3, apply_batchnorm=False)(x)
  x = get_conv(64, 3)(x)

  x = tf.keras.layers.Conv2D(1, 1, strides=1)(x)

  print(x.shape[0])
  print(x.shape[1])
  print(x.shape[2])
  print(x.shape[3]) 

  return tf.keras.Model(inputs=inputs, outputs=x)

#attention_mask_output = tf.maths.log(attention_mask_output)

### avoid using batch normalization when training VAEs, since the additional stochasticity due to using mini-batches may aggravate instability on top of the stochasticity from sampling.

def get_uncompiled_VAE_tf(input_size):
  
  image = tf.keras.layers.Input(shape=[input_size,input_size,3])
  mask = tf.keras.layers.Input(shape=[input_size,input_size,1])

  x = tf.keras.layers.concatenate([mask,image])

  # Encoder
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(x)
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(x)
  x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
  x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
  print(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(256, activation='relu')(x)
  x = tf.keras.layers.Dense(32)(x)
  mean, stddev = tf.split(x, num_or_size_splits=2, axis=1)

  # Variational layers
  latent = tf.random.normal(shape=(16,1))
  latent = tf.multiply(latent,stddev)
  latent = tf.reduce_max(latent, 1)
  x=tf.math.add(latent,mean)

  # Spatial Broadcast
  x=tf.reshape(x,[1,1,1,16])
  print(x)
  x = tf.tile(x, [1,input_size+8,input_size+8,1])
  print(x)
  line = tf.linspace(-1.0,1.0, input_size+8)
  x_channel, y_channel = tf.meshgrid(line, line)
  x_channel = tf.reshape(x_channel, [1,input_size+8, input_size+8, 1])
  y_channel = tf.reshape(y_channel, [1,input_size+8, input_size+8, 1])
  x = tf.keras.layers.Concatenate()([x, x_channel, y_channel])

  # Decoder
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)

  return tf.keras.Model(inputs=[mask,image], outputs=x)

def get_compiled_VAE(input_size):
  model = get_uncompiled_VAE_tf(input_size)
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
  return model


model = get_uncompiled_VAE_tf(input_size)

