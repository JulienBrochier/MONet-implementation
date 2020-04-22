import tensorflow as tf
import tensorflow_probability as tfp

class Vae(tf.keras.Model):
  def __init__ (self,input_size):
    super(Vae, self).__init__()
    self.input_size = input_size
    self.encoded_size = 16

  def call(self, inp):
    vae1_out = self.encoder(inp)
    latent = vae1_out.sample()
    vae2_out = self.spatial_broadcast(latent)
    output = self.decoder()(vae2_out)
    return [latent,output]

  def prior(self):
    return tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.encoded_size), scale=1),
                            reinterpreted_batch_ndims=1)

  #@tf.function
  def encoder(self, x):
    ### avoid using batch normalization when training VAEs, since the additional stochasticity due to using mini-batches may aggravate instability on top of the stochasticity from sampling.

    vae_encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[self.input_size,self.input_size,2]),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        tf.keras.layers.Conv2D(32, 3, strides=1,
                    padding='valid', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(32, 3, strides=2,
                    padding='valid', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(64, 3, strides=1,
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
    vae_encoder = tf.keras.Model(inputs=vae_encoder.inputs,
                    outputs=vae_encoder.outputs[0])

    return vae_encoder(x)

  def spatial_broadcast(self,inp):
    x=tf.reshape(inp,[1,1,1,16])
    x = tf.tile(x, [1,self.input_size+8,self.input_size+8,1])
    line = tf.linspace(-1.0,1.0, self.input_size+8)
    x_channel, y_channel = tf.meshgrid(line, line)
    x_channel = tf.reshape(x_channel, [1,self.input_size+8, self.input_size+8, 1])
    y_channel = tf.reshape(y_channel, [1,self.input_size+8, self.input_size+8, 1])
    output = tf.keras.layers.Concatenate()([x, x_channel, y_channel])
    return output

  def decoder(self):
    vae_decoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(4, (1, 1), strides=1, activation='relu', padding='valid')
    ])
    return vae_decoder