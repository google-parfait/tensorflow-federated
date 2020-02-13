# Lint as: python3
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
"""TensorFlow training code for Federated GANs.

This code is intended to only use vanilla TensorFlow (no TFF dependency); it is
wired together into a federated computation in gan_training_tff_fns.py. The one
exception is some handling for conversion from AnonymousTuple, which should go
away when b/130724878 is fixed.
"""

import attr
import tensorflow as tf

from tensorflow_federated.python.research.gans import gan_losses
from tensorflow_federated.python.tensorflow_libs import tensor_utils


def assert_no_anon_tuples(x):
  """Checks that a nested structure has no AnonymousTuples at the leaves."""

  def check_anon(t):
    if 'AnonymousTuple' in str(type(t)):
      raise ValueError('Found AnonymousTuple:\n', t)
    return None

  tf.nest.map_structure(check_anon, x)
  return x


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=True)
class FromServer(object):
  """Container for data that is broadcast from the server to clients.

  Attributes:
    generator_weights: Weights for the generator model, in the order of
      `tf.keras.Model.weights`.
    discriminator_weights: Weights for the discriminator model, in the order of
      `tf.keras.Model.weights`.
  """
  generator_weights = attr.ib()
  discriminator_weights = attr.ib()

  @classmethod
  def from_tff_result(cls, anon_tuple):
    # TODO(b/123092620): These conversions should not be needed.
    return assert_no_anon_tuples(
        cls(generator_weights=list(anon_tuple.generator_weights),
            discriminator_weights=list(anon_tuple.discriminator_weights)))


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=False)
class ServerState(object):
  """Container for all server state that must pass from round to round.

  Attributes:
    generator_weights: Weights for the generator model, in the order of
      `tf.keras.Model.weights`.
    discriminator_weights: Weights for the discriminator model, in the order of
      `tf.keras.Model.weights`.
    counters: Aggregated training counters.
    dp_averaging_state: State (possibly empty) of the dp_averaging_fn.
  """
  generator_weights = attr.ib()
  discriminator_weights = attr.ib()
  counters = attr.ib()
  dp_averaging_state = attr.ib(default=None)

  @classmethod
  def from_tff_result(cls, anon_tuple):
    # TODO(b/123092620): These conversions should not be needed.
    return assert_no_anon_tuples(
        cls(
            generator_weights=list(anon_tuple.generator_weights),
            discriminator_weights=list(anon_tuple.discriminator_weights),
            counters=anon_tuple.counters._asdict(),
            # TODO(b/123092620): Using _asdict(recursive=True) is a work-around
            # which at least gets rid of AnonymousTuples to allow the use of
            # tf.nest. However, really these should be the appropriate
            # namedtuple types expected by the TF Privacy code. This
            # means that in some cases ServerState.dp_averaging_state
            # needs dict-style access, and sometimes attribute-style.
            # However, since this is really opaque state, this only comes up
            # in the test.
            dp_averaging_state=anon_tuple.dp_averaging_state._asdict(
                recursive=True)))


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Container for data that is sent from clients back to the server..

  Attributes:
    discriminator_weights_delta: Update for the discriminator model, in the
      order of tf.keras.Model.weights`.
    update_weight: Weight to be associated with the update.
    counters: Metrics that are summed across clients.
  """
  discriminator_weights_delta = attr.ib()
  update_weight = attr.ib()
  counters = attr.ib()

  @classmethod
  def from_tff_result(cls, anon_tuple):
    # TODO(b/123092620): These conversions should not be needed.
    return assert_no_anon_tuples(
        cls(discriminator_weights_delta=list(
            anon_tuple.discriminator_weights_delta),
            update_weight=anon_tuple.update_weight,
            counters=anon_tuple.counters._asdict()))


def _weights(model):
  """Returns tensors of model weights, in the order of the variables."""
  return [v.read_value() for v in model.weights]


@tf.function
def client_computation(
    # Tensor/Dataset arguments that will be supplied by TFF:
    gen_inputs_ds: tf.data.Dataset,
    real_data_ds: tf.data.Dataset,
    from_server: FromServer,
    # Python arguments bound to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    train_discriminator_fn) -> ClientOutput:
  """The computation to run on the client, training the discriminator.

  Args:
    gen_inputs_ds: A `tf.data.Dataset` of generator_inputs.
    real_data_ds: A `tf.data.Dataset` of data from the real distribution.
    from_server: A `FromServer` object, including the current model weights.
    generator:  The generator.
    discriminator: The discriminator.
    train_discriminator_fn: A function which takes the two networks, generator
      input, and real data and trains the discriminator.

  Returns:
    A `ClientOutput` object.
  """
  tf.nest.map_structure(lambda a, b: a.assign(b), generator.weights,
                        from_server.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), discriminator.weights,
                        from_server.discriminator_weights)

  num_examples = tf.constant(0)
  gen_inputs_and_real_data = tf.data.Dataset.zip((gen_inputs_ds, real_data_ds))
  for gen_inputs, real_data in gen_inputs_and_real_data:
    # It's possible that real_data and gen_inputs have different batch sizes.
    # For calculating the discriminator loss, it's desirable to have equal-sized
    # contributions from both the real and fake data. Also, it's necessary if
    # using the Wasserstein gradient penalty (where a difference is taken b/w
    # the real and fake data). So here we reduce to the min batch size. This
    # also ensures num_examples properly reflects the amount of data trained on.
    min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(gen_inputs)[0])
    real_data = real_data[0:min_batch_size]
    gen_inputs = gen_inputs[0:min_batch_size]
    num_examples += train_discriminator_fn(generator, discriminator, gen_inputs,
                                           real_data)

  # TODO(b/142341957): This control_dependency should not be needed, but is
  # currently necessary to work around a TF bug with how tf.function handles
  # tf.data.Datasets.
  with tf.control_dependencies([num_examples]):
    weights_delta = tf.nest.map_structure(tf.subtract, discriminator.weights,
                                          from_server.discriminator_weights)
  weights_delta, has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  update_weight = tf.cast(num_examples, tf.float32)
  # Zero out the weight if there are any non-finite values.
  # TODO(b/122071074): federated_mean might not do the right thing if
  # all clients have zero weight.
  update_weight = tf.cond(
      tf.equal(has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))
  return ClientOutput(
      discriminator_weights_delta=weights_delta,
      update_weight=update_weight,
      counters={'num_discriminator_train_examples': num_examples})


def server_initial_state(generator, discriminator, dp_averaging_state=None):
  """Returns the initial state of the server."""
  return ServerState(
      generator_weights=_weights(generator),
      discriminator_weights=_weights(discriminator),
      counters={
          'num_rounds': tf.constant(0),
          'num_generator_train_examples': tf.constant(0),
          'num_discriminator_train_examples': tf.constant(0)
      },
      dp_averaging_state=dp_averaging_state)


@tf.function
def server_computation(
    # Tensor/Dataset arguments that will be supplied by TFF:
    server_state: ServerState,
    gen_inputs_ds: tf.data.Dataset,
    client_output: ClientOutput,
    # Python arguments to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    server_disc_update_optimizer: tf.keras.optimizers.Optimizer,
    train_generator_fn,
    # Not an argument bound at TFF computation construction time, but placed
    # last so that it can be defaulted to empty tuple (for non-DP use cases).
    new_dp_averaging_state=()
) -> ServerState:
  """The computation to run on the server, training the generator.

  Args:
    server_state: The initial `ServerState` for the round.
    gen_inputs_ds: An infinite `tf.data.Dataset` of inputs to the `generator`.
    client_output: The (possibly aggregated) `ClientOutput`.
    generator:  The generator.
    discriminator: The discriminator.
    server_disc_update_optimizer: Optimizer used to `apply_gradients` based on
      the client_output delta.
    train_generator_fn: A function which takes the two networks and generator
      input and trains the generator.
    new_dp_averaging_state: The updated state of the DP averaging aggregator.

  Returns:
    An updated `ServerState` object.
  """
  # A tf.function can't modify the structure of its input arguments,
  # so we make a semi-shallow copy:
  server_state = attr.evolve(server_state, counters=dict(server_state.counters))

  tf.nest.map_structure(lambda a, b: a.assign(b), generator.weights,
                        server_state.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), discriminator.weights,
                        server_state.discriminator_weights)

  # Update the server discriminator.
  delta = client_output.discriminator_weights_delta
  tf.nest.assert_same_structure(delta, discriminator.weights)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (-1.0 * x, v), delta,
                                         discriminator.weights)
  server_disc_update_optimizer.apply_gradients(
      grads_and_vars, name='server_update')

  for k, v in client_output.counters.items():
    server_state.counters[k] += v

  # Update the state of the DP averaging aggregator.
  server_state.dp_averaging_state = new_dp_averaging_state

  gen_examples_this_round = tf.constant(0)

  for gen_inputs in gen_inputs_ds:  # Compiled by autograph.
    gen_examples_this_round += train_generator_fn(generator, discriminator,
                                                  gen_inputs)

  # TODO(b/142341957): This control_dependency should not be needed, but is
  # currently necessary to work around a TF bug with how tf.function handles
  # tf.data.Datasets.
  with tf.control_dependencies([gen_examples_this_round]):
    server_state.counters[
        'num_generator_train_examples'] += gen_examples_this_round
    server_state.generator_weights = _weights(generator)
    server_state.discriminator_weights = _weights(discriminator)
  server_state.counters['num_rounds'] += 1
  return server_state


def create_train_generator_fn(gan_loss_fns: gan_losses.AbstractGanLossFns,
                              gen_optimizer: tf.keras.optimizers.Optimizer):
  """Create a function that trains generator, binding loss and optimizer.

  Args:
    gan_loss_fns: Instance of gan_losses.AbstractGanLossFns interface,
      specifying the generator/discriminator training losses.
    gen_optimizer: Optimizer for training the generator.

  Returns:
    Function that executes one step of generator training.
  """
  # We check that the optimizer has not been used previously, which ensures
  # that when it is bound the train fn isn't holding onto a different copy of
  # the optimizer variables then the copy that is being exchanged b/w server and
  # clients.
  if gen_optimizer.variables():
    raise ValueError(
        'Expected gen_optimizer to not have been used previously, but '
        'variables were already initialized.')

  @tf.function
  def train_generator_fn(generator: tf.keras.Model,
                         discriminator: tf.keras.Model, generator_inputs):
    """Trains the generator on a single batch.

    Args:
      generator:  The generator.
      discriminator: The discriminator.
      generator_inputs: A batch of inputs (usually noise) for the generator.

    Returns:
      The number of examples trained on.
    """

    def gen_loss():
      """Does the forward pass and computes losses for the generator."""
      # N.B. The complete pass must be inside loss() for gradient tracing.
      return gan_loss_fns.generator_loss(generator, discriminator,
                                         generator_inputs)

    gen_optimizer.minimize(gen_loss, var_list=generator.trainable_variables)
    return tf.shape(generator_inputs)[0]

  return train_generator_fn


def create_train_discriminator_fn(
    gan_loss_fns: gan_losses.AbstractGanLossFns,
    disc_optimizer: tf.keras.optimizers.Optimizer):
  """Create a function that trains discriminator, binding loss and optimizer.

  Args:
    gan_loss_fns: Instance of gan_losses.AbstractGanLossFns interface,
      specifying the generator/discriminator training losses.
    disc_optimizer: Optimizer for training the discriminator.

  Returns:
    Function that executes one step of discriminator training.
  """
  # We assert that the optimizer has not been used previously, which ensures
  # that when it is bound the train fn isn't holding onto a different copy of
  # the optimizer variables then the copy that is being exchanged b/w server and
  # clients.
  if disc_optimizer.variables():
    raise ValueError(
        'Expected disc_optimizer to not have been used previously, but '
        'variables were already initialized.')

  @tf.function
  def train_discriminator_fn(generator: tf.keras.Model,
                             discriminator: tf.keras.Model, generator_inputs,
                             real_data):
    """Trains the discriminator on a single batch.

    Args:
      generator:  The generator.
      discriminator: The discriminator.
      generator_inputs: A batch of inputs (usually noise) for the generator.
      real_data: A batch of real data for the discriminator.

    Returns:
      The size of the batch.
    """

    def disc_loss():
      """Does the forward pass and computes losses for the discriminator."""
      # N.B. The complete pass must be inside loss() for gradient tracing.
      return gan_loss_fns.discriminator_loss(generator, discriminator,
                                             generator_inputs, real_data)

    disc_optimizer.minimize(
        disc_loss, var_list=discriminator.trainable_variables)
    return tf.shape(real_data)[0]

  return train_discriminator_fn
