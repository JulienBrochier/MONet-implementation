import tensorflow as tf
import tensorflow_probability as tfp

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
    self.generative_net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=[self.input_width+8,self.input_width+8,self.encoded_size+2]),
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),  
      tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
      tf.keras.layers.Conv2D(self.input_channels+1, (1, 1), strides=1, activation='relu', padding='valid')
    ])

  def call(self, inp, scale):
    approx_posterior = self.encoder(inp)
    #print('unet trainable weights:', len(layers.trainable_weights))
    tiled_output = self.spatial_broadcast(approx_posterior.sample())
    likelihood, mask = self.decoder(tiled_output, scale)
    return approx_posterior, likelihood, mask

  def prior(self):
    return tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.encoded_size), scale=1),
                            reinterpreted_batch_ndims=1)

  def encoder(self,x):
    return self.inference_net(x)

  def spatial_broadcast(self,inp):
    x=tf.reshape(inp,[1,1,1,self.encoded_size])
    x = tf.tile(x, [1,self.input_width+8,self.input_width+8,1])
    line = tf.linspace(-1.0,1.0, self.input_width+8)
    x_channel, y_channel = tf.meshgrid(line, line)
    x_channel = tf.reshape(x_channel, [1,self.input_width+8, self.input_width+8, 1])
    y_channel = tf.reshape(y_channel, [1,self.input_width+8, self.input_width+8, 1])
    output = tf.keras.layers.Concatenate()([x, x_channel, y_channel])
    return output

  def decoder(self, x, scale):
    x = self.generative_net(x)
    likelihood = tfp.distributions.Normal(loc=x[...,:self.input_channels], scale=scale)
    mask = tfp.distributions.Bernoulli(logits=x[...,self.input_channels:])    
    return likelihood, mask