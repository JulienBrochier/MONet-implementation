# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://www.tensorflow.org/tutorials/generative/pix2pix
# https://www.tensorflow.org/tutorials/keras/save_and_load
# https://www.tensorflow.org/guide/checkpoint

import tensorflow as tf
import tensorflow_probability as tfp
from unet import Unet
from vae import Vae
import time

class Monet(tf.keras.Model):
  def __init__ (self, input_width, input_channels, nb_scopes, batch_size):
    super(Monet, self).__init__()
    self.input_width = input_width
    self.input_channels = input_channels
    self.encoded_size = 32
    self.unet = Unet(input_width, input_channels, batch_size)
    self.vae = Vae(input_width, input_channels, self.encoded_size, batch_size)
    self.beta = 0.5
    self.gamma = 0.5
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.first_loss = []
    self.second_loss = []
    self.third_loss = []
    self.nb_scopes = nb_scopes
    self.batch_size = batch_size

  def call(self, image):
    """
    Forward Pass
    """
    # Initialize the first scope
    s0 = tf.ones([1,self.input_width,self.input_width,self.input_channels], dtype=tf.dtypes.float32)
    log_sk= tf.math.log(s0)
    scale = 0.09  #The "background" component scale, 0.09 at the first iteration, then 0.11 (MONet-$B.1-ComponentVAE)

    unet_masks = []
    vae_masks = []
    reconstructed_imgs = []

    ## Iterate through the scopes
    for i in range(self.nb_scopes):
      # Attention Network
      if(i==self.nb_scopes-1):
        log_mk = log_sk
      else:
        log_mk, log_sk = self.unet(image,log_sk)
      x = tf.keras.layers.concatenate([log_mk,image])
      # VAE
      [approx_posterior,reconstructed_image_distrib,reconstructed_mask_distrib] = self.vae(x, scale)
      scale = 0.11
      # Store outputs
      unet_masks.append(tf.exp(log_mk))
      vae_masks.append(reconstructed_mask_distrib.sample())
      reconstructed_imgs.append(reconstructed_image_distrib.sample())

    return unet_masks, vae_masks, reconstructed_imgs

  def compute_loss(self, image):
    """
    Forward Pass + Loss Computation
    """
    scale = 0.09  #The "background" component scale, 0.09 at the first iteration, then 0.11 (MONet-$B.1-ComponentVAE)
    # Initialize the first scope
    s0 = tf.ones([self.batch_size,self.input_width,self.input_width,1], dtype=tf.dtypes.float32)
    log_sk= tf.math.log(s0)
    # List to store Unet and VAE masks'
    l_log_mk = []
    l_mktilda = []
    # Initialize loss
    l1 = 0
    l2 = 0
    l3 = 0
    prior = self.make_mixture_prior()

    ## Iterate through the scopes
    i=0
    while i<self.nb_scopes and tf.math.exp(tf.math.reduce_mean(log_sk)) != 0.0 :
      # Attention Network
      if(i==self.nb_scopes-1):
        log_mk = log_sk
        i=self.nb_scopes-1
      else:
        log_mk, log_sk = self.unet(image,log_sk)
      x = tf.keras.layers.concatenate([log_mk,image])
      # VAE
      [approx_posterior,reconstructed_image_distrib,reconstructed_mask_distrib] = self.vae(x, scale)
      # l1 computation
      l1 += tf.math.reduce_mean(tf.math.exp(log_mk) * tf.cast(reconstructed_image_distrib.prob(image), tf.float32))
      # Store outputs for l3 computation
      l_mktilda.append(reconstructed_mask_distrib.prob(image))
      l_log_mk.append(log_mk)
      # Prepare next step
      i+=1
      scale = 0.11 #The "background" component scale, 0.09 at the first iteration, then 0.11

    l1 = -tf.math.log(l1)
    l3 = self.compute_third_loss(l_log_mk,l_mktilda)
    print("L1 = {}, L3 = {}".format(l1,l3))
    # Loss lists will be used to plot the model evolution
    self.first_loss.append(l1)
    self.second_loss.append(l2)
    self.third_loss.append(l3)

    return l1 + l3

  def compute_third_loss(self,l_log_mk,l_mktilda):
    p = tf.keras.backend.concatenate(l_mktilda,axis=-1)
    log_q = tf.keras.backend.concatenate(l_log_mk,axis=-1)
    # Normalise p(c|x) so sum_over_K(p(c=k|x) = 1)
    p_sum = tf.reduce_sum(p, axis=-1)
    p = p / tf.expand_dims(p_sum, -1)
    # Compute l3
    l3 = self.gamma * tf.reduce_sum( tf.math.exp(log_q)*(log_q-tf.math.log(p)), axis=-1)
    return tf.reduce_mean(l3)

  #@tf.function
  def compute_apply_gradient(self, batch):
    with tf.GradientTape() as tape:
      loss = self.compute_loss(batch)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

  def make_mixture_prior(self):
    """
    returns a probability distribution of a multivariate 
    normal law for the prior
    Returns p(x)
    """
    return tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros([self.encoded_size]),
        scale_identity_multiplier=1.0)

  def fit(self,dataset,save_path=None,summary_writer=None):
    for step, batch in enumerate(dataset):
      i=step+1
      t0 = time.time()
      self.compute_apply_gradient(batch)
      if save_path and i%5==0:
        self.save_weights(save_path+str(i//5))
        print("Training {} to {} : {}sec".format(i-5,i,time.time()-t0))
        #print("L1 = {}, L2 = {}, L3 = {}".format(self.first_loss[-1], self.second_loss[-1], self.third_loss[-1]))
        t0 = time.time()
        with summary_writer.as_default():
          tf.summary.scalar('l1', self.first_loss[-1], step=i)
          tf.summary.scalar('l2', self.second_loss[-1], step=i)
          tf.summary.scalar('l3', self.third_loss[-1], step=i)

    return self.first_loss, self.second_loss, self.third_loss



    