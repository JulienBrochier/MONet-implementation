import tensorflow as tf
import tensorflow_probability as tfp
import time

class Vae(tf.keras.layers.Layer):
  def __init__ (self,input_width,input_channels,encoded_size,batch_size):
    super(Vae, self).__init__()
    self.input_width = input_width
    self.input_channels = input_channels
    self.encoded_size = encoded_size
    self.batch_size = batch_size
    self.inference_net = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[self.input_width,self.input_width,self.input_channels+1]),
        tf.keras.layers.Conv2D(32, 3, strides=2,
                    padding='valid', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(32, 3, strides=2,
                    padding='valid', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(64, 3, strides=2,
                    padding='valid', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(64, 3, strides=2,
                    padding='valid', activation=tf.nn.leaky_relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.encoded_size),
               activation=None),
        tfp.layers.MultivariateNormalTriL(
                self.encoded_size,
                activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior(), weight=1.0))
    ])
    self.broadcasting_net = self.spatial_broadcast()
    self.generative_net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=[self.input_width+8,self.input_width+8,self.encoded_size+2]),
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),  
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
      tf.keras.layers.Conv2D(self.input_channels+1, (1, 1), strides=1, activation='relu', padding='valid')
    ])

  def call(self, inp, scale):
    t_vae = time.time()
    approx_posterior = self.encoder(inp)
    tiled_output = self.broadcasting_net(approx_posterior.sample())
    reconstructed_image_distrib, reconstructed_mask_distrib = self.decoder(tiled_output, scale)
    print("Forward pass VAE : {}sec".format(time.time()-t_vae))
    return approx_posterior, reconstructed_image_distrib, reconstructed_mask_distrib

  def encoder(self,x):
    return self.inference_net(x)

  def spatial_broadcast(self):
    latent = tf.keras.layers.Input(shape=[self.encoded_size], name="latent_vector")
    latent_reshaped = tf.keras.layers.Reshape((1, 1, self.encoded_size))(latent)
    latent_broadcasted = tf.keras.layers.UpSampling2D((self.input_width+8,self.input_width+8))(latent_reshaped)
    x_tile, y_tile = self.create_tiles()
    output = tf.keras.layers.Concatenate()([latent_broadcasted, x_tile, y_tile])
    return tf.keras.Model(inputs=latent, outputs=output)

  def decoder(self, x, scale):
    x = self.generative_net(x)
    reconstructed_image_distrib = tfp.distributions.Normal(loc=x[...,:self.input_channels], scale=scale)
    reconstructed_mask_distrib = tfp.distributions.Bernoulli(logits=x[...,self.input_channels:])    
    return reconstructed_image_distrib, reconstructed_mask_distrib

  def prior(self):
    return tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.encoded_size), scale=1),
                            reinterpreted_batch_ndims=1)

  def create_tiles(self):
    line = tf.linspace(-1.0,1.0, self.input_width+8)
    ones = tf.ones(self.batch_size)
    x_tile = tf.meshgrid(line, ones, line)[0]
    y_tile = tf.meshgrid(line, ones, line)[2]
    x_tile = tf.reshape(x_tile, [self.batch_size,self.input_width+8, self.input_width+8, 1])
    y_tile = tf.reshape(y_tile, [self.batch_size,self.input_width+8, self.input_width+8, 1])
    return x_tile, y_tile