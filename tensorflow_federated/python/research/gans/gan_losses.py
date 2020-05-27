# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library of classes which define GAN loss functions to use during training.

This code is intended to only use vanilla TensorFlow (no TFF dependency); it is
used as part of a federated computation in gan_training_tf_fns.py.
"""

import abc

import tensorflow as tf

# This controls the behavior of BatchNormalization Layers.
# For some reason I don't fully understand, even though we are
# not training these models where this flag is used,
# it must be set to True or training does not converge.
TRAINING_KWARG_FOR_SECOND_MODEL = True


class AbstractGanLossFns(abc.ABC):
  """An abstract class for functions defining GAN Losses."""

  @abc.abstractmethod
  def generator_loss(self, generator: tf.keras.Model,
                     discriminator: tf.keras.Model, gen_inputs):
    """Does the forward pass and computes losses for the generator."""
    # N.B. The complete pass must be inside generator_loss() for gradient
    # tracing.
    raise NotImplementedError

  @abc.abstractmethod
  def discriminator_loss(self, generator: tf.keras.Model,
                         discriminator: tf.keras.Model, gen_inputs, real_data):
    """Does the forward pass and computes losses for the discriminator."""
    # N.B. The complete pass must be inside discriminator_loss() for gradient
    # tracing.
    raise NotImplementedError


class WassersteinGanLossFns(AbstractGanLossFns):
  """Class with functions implementing Wasserstein GAN loss.

  This class can be used for both the basic Wasserstein GAN ('WGAN') loss (as
  presented in the original Wasserstein GAN paper,
  https://arxiv.org/abs/1701.07875) as well as the improved Wasserstein GAN
  ('WGAN-GP') loss (as presented in 'Improved Training of Wasserstein GANs',
  https://arxiv.org/abs/1704.00028).

  The key difference b/w WGAN and WGAN-GP is that in WGAN-GP the discriminator
  loss adds a gradient penalty ('GP') term to enforce the Lipschitz constraint
  on the discriminator (aka the 'critic' in the context of Wasserstein GANs),
  whereas with original Wasserstein GAN this Lipschitz constraint is supposed to
  be enforced via weight clipping.

  To enable the gradient penalty term, set the `grad_penalty_lambda` argument in
  the constructor to a value greater than 0.0.
  """

  def __init__(self, grad_penalty_lambda=0.0):
    self._grad_penalty_lambda = grad_penalty_lambda

  def generator_loss(self, generator: tf.keras.Model,
                     discriminator: tf.keras.Model, gen_inputs):
    """Does the forward pass and computes losses for the generator."""
    gen_images = generator(gen_inputs, training=True)
    return _wass_gen_loss_fn(gen_images, discriminator, generator)

  def discriminator_loss(self, generator: tf.keras.Model,
                         discriminator: tf.keras.Model, gen_inputs, real_data):
    """Does the forward pass and computes losses for the discriminator."""
    gen_images = generator(gen_inputs, training=TRAINING_KWARG_FOR_SECOND_MODEL)
    return _wass_disc_loss_fn(
        real_data,
        gen_images,
        discriminator,
        grad_penalty_lambda=self._grad_penalty_lambda)


class _ImprovedWassersteinGanLossFns(WassersteinGanLossFns):
  """Class allows for getting an easy WGAN-GP instance (w/ default lambda)."""

  def __init__(self):
    super().__init__(grad_penalty_lambda=10.0)


GAN_LOSS_FNS_DICT = {
    'wasserstein': WassersteinGanLossFns,
    'improved_wasserstein': _ImprovedWassersteinGanLossFns
}


def get_gan_loss_fns(gan_loss_fns_name):
  """Utility providing easy way to get instance of AbstractGanLossFns."""
  if gan_loss_fns_name in GAN_LOSS_FNS_DICT.keys():
    gan_loss_fns_fn = GAN_LOSS_FNS_DICT[gan_loss_fns_name]
    return gan_loss_fns_fn()
  raise KeyError(
      '\'%s\' is invalid flavor of GAN loss function. Choices are %s' %
      (gan_loss_fns_name, str([name for name in GAN_LOSS_FNS_DICT.keys()])))


def _wass_grad_penalty_term(real_images,
                            gen_images,
                            discriminator: tf.keras.Model,
                            grad_penalty_lambda,
                            epsilon=1e-10):
  """Calculate the gradient penalty term used in improved Wasserstein."""

  def _get_interpolates(real_images, gen_images):
    # Forms batch of interpolated images, for use in grad penalty calculation.
    # This is step 6 of Algorithm 1 in WGAN-GP paper. See discussion in
    # ' Sampling distribution' part of Section 4 of the paper.
    differences = gen_images - real_images
    batch_size = tf.shape(differences)[0]
    alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
    alpha = tf.random_uniform(shape=alpha_shape)
    interpolates = real_images + (alpha * differences)
    return interpolates

  interpolates = _get_interpolates(real_images, gen_images)

  # The gradient of the discriminator ('critic') w.r.t. the interpolated images.
  with tf.GradientTape() as wass_grad_tape:
    wass_grad_tape.watch(interpolates)
    disc_interpolates = discriminator(interpolates, training=True)
  gradients = wass_grad_tape.gradient(disc_interpolates, interpolates)

  def _calculate_penalty(gradients, epsilon=1e-10):
    # See Section 4 of the WGAN-GP paper (https://arxiv.org/abs/1704.00028),
    # describing the additional gradient penalty term.
    gradient_squares = tf.reduce_sum(
        tf.square(gradients), axis=list(range(1, gradients.shape.ndims)))
    gradient_norm = tf.sqrt(gradient_squares + epsilon)
    penalties_squared = tf.square(gradient_norm - 1.0)
    return tf.reduce_mean(penalties_squared)

  penalty = _calculate_penalty(gradients, epsilon)

  return penalty * grad_penalty_lambda


def _wass_disc_loss_fn(real_images,
                       gen_images,
                       discriminator: tf.keras.Model,
                       grad_penalty_lambda=0.0):
  """Calculate the Wasserstein (discriminator) loss."""
  if grad_penalty_lambda < 0.0:
    raise ValueError('grad_penalty_lambda must be greater than or equal to 0.0')

  # For calculating the discriminator loss, it's desirable to have equal-sized
  # contributions from both the real and fake data. Also, it's necessary if
  # computing the Wasserstein gradient penalty (where a difference is taken b/w
  # the real and fake data). So we assert batch_size equality here.
  with tf.control_dependencies(
      [tf.assert_equal(tf.shape(real_images)[0],
                       tf.shape(gen_images)[0])]):

    disc_gen_output = discriminator(gen_images, training=True)
    score_on_generated = tf.reduce_mean(disc_gen_output)

    disc_real_output = discriminator(real_images, training=True)
    score_on_real = tf.reduce_mean(disc_real_output)

    disc_loss = score_on_generated - score_on_real
    # Add gradient penalty, if indicated.
    if grad_penalty_lambda > 0.0:
      disc_loss += _wass_grad_penalty_term(real_images, gen_images,
                                           discriminator, grad_penalty_lambda)

    # Now add discriminator model regularization losses in.
    if discriminator.losses:
      disc_loss += tf.add_n(discriminator.losses)
    return disc_loss


def _wass_gen_loss_fn(gen_images, discriminator: tf.keras.Model,
                      generator: tf.keras.Model):
  """Calculate the Wasserstein (generator) loss."""
  disc_gen_output = discriminator(
      gen_images, training=TRAINING_KWARG_FOR_SECOND_MODEL)

  gen_loss = tf.reduce_mean(-disc_gen_output)
  # Now add generator model regularization losses in.
  if generator.losses:
    gen_loss += tf.add_n(generator.losses)
  return gen_loss
