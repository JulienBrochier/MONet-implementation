#https://www.tensorflow.org/guide/keras/custom_layers_and_models
#https://www.tensorflow.org/tutorials/generative/pix2pix

import tensorflow as tf
import tensorflow_probability as tfp
from unet import Unet
from vae import Vae

class Monet(tf.keras.Model):
  def __init__ (self, input_width, input_channels, nb_scopes):
    super(Monet, self).__init__()
    self.input_width = input_width
    self.input_channels = input_channels
    self.encoded_size = 32
    self.unet = Unet(input_width, input_channels)
    self.vae = Vae(input_width, input_channels, self.encoded_size)
    self.beta = 0.5
    self.gamma = 0.5
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.first_loss = []
    self.second_loss = []
    self.third_loss = []
    self.nb_scopes = nb_scopes

  def call(self, image):
    """
    Forward Pass
    """
    # Initialize the first scope
    s0 = tf.ones([1,self.input_width,self.input_width,1], dtype=tf.dtypes.float32)
    log_sk= tf.math.log(s0)
    scale = 0.09  #The "background" component scale, 0.09 at the first iteration, then 0.11 (MONet-$B.1-ComponentVAE)

    unet_masks = []
    vae_masks = []
    reconstructed_imgs = []

    ## Iterate through the scopes
    for i in range(self.nb_scopes-1):
      # Attention Network
      log_mk, log_sk = self.unet(image,log_sk)
      x = tf.keras.layers.concatenate([log_mk,image])
      # VAE
      [approx_posterior,decoder_likelihood,vae_mask] = self.vae(x, scale)
      scale = 0.11
      # Store outputs
      unet_masks.append(tf.exp(log_mk))
      vae_masks.append(vae_mask)
      reconstructed_imgs.append(decoder_likelihood.sample())

    ## Last iteration, mk = sk-1
    # Attention Network
    log_mk = log_sk
    x = tf.keras.layers.concatenate([log_mk,image])
    # VAE
    [approx_posterior,decoder_likelihood,vae_mask] = self.vae(x, scale)
    # Store outputs
    unet_masks.append(tf.exp(log_mk))
    vae_masks.append(vae_mask)
    reconstructed_imgs.append(decoder_likelihood.sample())

    return unet_masks, vae_masks, reconstructed_imgs

  @tf.function
  def compute_loss(self, image):
    """
    Forward Pass + Loss Computation
    """
    scale = 0.09  #The "background" component scale, 0.09 at the first iteration, then 0.11 (MONet-$B.1-ComponentVAE)
    # Initialize the first scope
    s0 = tf.ones([1,self.input_width,self.input_width,1], dtype=tf.dtypes.float32)
    log_sk= tf.math.log(s0)
    # List to store Unet and VAE masks'
    l_log_mk = []
    l_vae_mask = []
    # Initialize loss
    first_loss_term = second_loss_term = 0
    prior = self.make_mixture_prior()

    ## Iterate through the scopes
    for i in range(self.nb_scopes-1):
      # Attention Network
      log_mk, log_sk = self.unet(image,log_sk)
      x = tf.keras.layers.concatenate([log_mk,image])
      # VAE
      [approx_posterior,decoder_likelihood,vae_mask] = self.vae(x, scale)
      # Fisrt and second loss term computation
      first_loss_term += tf.math.reduce_mean(tf.math.exp(log_mk)) * tf.math.reduce_mean(decoder_likelihood.mean())
      second_loss_term += tfp.distributions.kl_divergence(approx_posterior,prior)
      # Store variables for computation of the third loss term
      # They are converted from int32 to float32 to allow concatenation in self.compute_third_loss()
      log_mk = tf.keras.backend.cast(log_mk,"float32")
      vae_mask_sample = tf.keras.backend.cast(vae_mask.sample(),"float32")
      l_log_mk.append(log_mk)
      l_vae_mask.append(vae_mask_sample)
      scale = 0.11

    ## Last iteration, mk = sk-1
    # Attention Network
    log_mk = log_sk
    x = tf.keras.layers.concatenate([log_mk,image])
    # VAE
    [approx_posterior,decoder_likelihood,vae_mask] = self.vae(x, scale)
    # First and second loss term computation
    first_loss_term += tf.math.reduce_mean(tf.math.exp(log_mk)) * tf.math.reduce_mean(decoder_likelihood.mean())
    second_loss_term += tfp.distributions.kl_divergence(approx_posterior,prior)
    # Third loss term
    log_mk = tf.keras.backend.cast(log_mk,"float32")
    vae_mask_sample = tf.keras.backend.cast(vae_mask.sample(),"float32")
    l_log_mk.append(log_mk)
    l_vae_mask.append(vae_mask_sample)
    third_loss_term = self.compute_third_loss(l_log_mk, l_vae_mask)
    # Loss lists will be returned by self.train()
    self.first_loss.append(first_loss_term)
    self.second_loss.append(second_loss_term)
    self.third_loss.append(third_loss_term)

    return first_loss_term + third_loss_term   #+ second_loss_term

  @tf.function
  def compute_third_loss(self,l_log_mk, l_vae_mask):
    """
    gamma*Dkl(q(c|x) || p(c|z))
    q(c=k|x) : probability for a pixel c to belong to the scope k (the kth mask returned by Unet)
    p(c=k|z) : probability for a pixel c to belong to the kth reconstruction mask (the kth mask returned by the VAE)
    """
    vae_global_mask = tf.keras.backend.concatenate(l_vae_mask,axis=-1)
    unet_global_mask = tf.math.exp(tf.keras.backend.concatenate(l_log_mk,axis=-1))
    # Normalize, so sum for each k of q(c=k|x) = 1
    vae_global_mask = tf.nn.softmax(vae_global_mask,axis=-1)
    unet_global_mask = tf.nn.softmax(unet_global_mask,axis=-1)

    # Create the distributions q(c|x) and p(c|z)
    vae_distrib = tfp.distributions.Categorical(probs=vae_global_mask)
    unet_distrib = tfp.distributions.Categorical(probs=unet_global_mask)
    # Compute KL
    third_loss_term = tfp.distributions.kl_divergence(unet_distrib,vae_distrib) * self.gamma
    third_loss_term = tf.math.reduce_mean(third_loss_term)
    return third_loss_term

  @tf.function
  def compute_apply_gradient(self, batch):
    with tf.GradientTape() as tape:
      loss = self.compute_loss(batch)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

  def make_mixture_prior(self):
    """c
    returns a probability distribution of a multivariate 
    normal law for the prior
    Returns p(x)
    """
    return tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros([self.encoded_size]),
        scale_identity_multiplier=1.0)

  def fit(self,dataset,save_path=None):
    i=1
    for batch in dataset.as_numpy_iterator():
      print("Training {}".format(i))
      self.compute_apply_gradient(batch)
      i = i+1
      if save_path and i%50==0:
        self.save_weights(save_path+str(i//50))
    return self.first_loss, self.second_loss, self.third_loss



    