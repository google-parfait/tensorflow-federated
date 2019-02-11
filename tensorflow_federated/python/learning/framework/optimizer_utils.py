# Copyright 2018, The TensorFlow Federated Authors.
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
"""Common building blocks for federated optimization algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six
from six.moves import zip
import tensorflow as tf

# TODO(b/123578208): Remove deep keras imports after updating TF version.
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.learning import model_utils

nest = tf.contrib.framework.nest

# TODO(b/117226648): Make this a proper class for better documentation.
ClientOutput = collections.namedtuple(
    'ClientOutput',
    [
        # A dictionary of updates to the model's trainable variables.
        'weights_delta',
        # Weight to use in a weighted mean when aggregating weights_delta.
        'weights_delta_weight',
        # A structure matching model.report_local_outputs,
        # reflecting the results of training on the input dataset.
        'model_output',
        # Additional metrics or other outputs defined by the optimizer.
        'optimizer_output'
    ])


@six.add_metaclass(abc.ABCMeta)
class ClientDeltaFn(object):
  """Represents a client computation that produces an update to a model."""

  @abc.abstractproperty
  def variables(self):
    """Returns all the variables of this object.

    Note this only includes variables that are part of the state of this object,
    and not the model variables themselves.

    Returns:
      An iterable of `tf.Variable` objects.
    """
    pass

  @abc.abstractmethod
  def __call__(self, dataset, initial_weights):
    """Defines the complete client computation.

    Typically implementations should be decorated with `tf.function`.

    Args:
      dataset: A `tf.data.Dataset` producing batches than can be fed to
        `model.forward_pass`.
      initial_weights: A dictionary of initial values for all trainable and
        non-trainable model variables, keyed by name. This will be supplied by
        the server in Federated Averaging.

    Returns:
      An `optimizer_utils.ClientOutput` namedtuple.
    """
    pass


def _create_optimizer_and_server_state(model, optimizer):
  """A helper for server computations that constructs the model and optimizer.

  This code is needed both in server_init (to introduce variables so
  we can read their initial values) and in server_update_model.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A tuple of (apply_delta_fn, server_state), where:
      *  apply_delta_fn is a TensorFlow function that takes a model delta and
         updates the trainable model weights as well as possibly optimizer_state
         variables introduced by the optimizer.
      *  server_state is a `tff.learning.framework.ServerState` tuple holding
         those variables.
  """

  @tf.contrib.eager.defun()
  def apply_delta(delta):
    """Applies `delta` to `model.weights`."""
    nest.assert_same_structure(delta, model.weights.trainable)
    grads_and_vars = nest.map_structure(lambda x, v: (-1.0 * x, v),
                                        nest.flatten(delta),
                                        nest.flatten(model.weights.trainable))
    # N.B. This may create variables.
    optimizer.apply_gradients(grads_and_vars, name='server_update')
    return tf.constant(1)  # We have to return something.

  # Create a dummy input and trace apply_delta so that
  # we can determine the optimizer's variables.
  weights_delta = nest.map_structure(tf.zeros_like, model.weights.trainable)

  # TODO(b/109733734): We would like to call get_concrete_function,
  # but that does not currently work with structured inputs.
  # For now, we just call the function on dummy input, which
  # still ensures the function is traced (so variables are created).
  apply_delta(delta=weights_delta)

  # N.B. Using to_var_dict doesn't work here, because we
  # may get non-unique names, so we just use a flat list.
  optimizer_vars = optimizer.variables()

  return apply_delta, ServerState(
      model=model.weights, optimizer_state=optimizer_vars)


# Represents the state of the server carried between rounds.
ServerState = collections.namedtuple(
    'ServerState',
    [
        # A ModelWeights structure, containing Tensors or Variables.
        'model',
        # A list of Tensors or Variables, in the order
        # returned by optimizer.variables()
        'optimizer_state'
    ])


def server_init(model_fn, optimizer_fn):
  """Returns initial `tff.learning.framework.ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`.

  Returns:
    A `tff.learning.framework.ServerState` namedtuple.
  """
  model = model_utils.enhance(model_fn())
  optimizer = optimizer_fn()
  _, server_state = _create_optimizer_and_server_state(model, optimizer)
  return server_state


def server_update_model(current_server_state, weights_delta, model_fn,
                        optimizer_fn):
  """Updates `server_state` based on `weights_delta`.

  Args:
    current_server_state: A `tff.learning.framework.ServerState` namedtuple.
    weights_delta: An update to the trainable variables of the model.
    model_fn: A no-arg function that returns a `tff.learning.Model`. Passing in
      a function ensures any variables are created when server_update_model is
      called, so they can be captured in a specific graph or other context.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`. As with
      model_fn, we pass in a function to control when variables are created.

  Returns:
    An updated `tff.learning.framework.ServerState`.
  """
  model = model_utils.enhance(model_fn())
  optimizer = optimizer_fn()
  apply_delta_fn, new_server_state = _create_optimizer_and_server_state(
      model, optimizer)

  # TODO(b/123092620): Mixing AnonymousTuple with other nested types is not
  # pretty, fold this into anonymous_tuple module or get working with
  # tf.contrib.framework.nest.
  if isinstance(weights_delta, anonymous_tuple.AnonymousTuple):
    flat_delta = anonymous_tuple.flatten(weights_delta)
    weights_delta = nest.pack_sequence_as(model.weights.trainable, flat_delta)

  # TODO(b/109733734): Does this really need to be wrapped, since its
  # immediately called below?
  @tf.contrib.eager.function()
  def update_model_inner():
    # TODO(b/123092620): Mixing AnonymousTuple with other nested types is a bit
    # cumbersome. Make this easier with better support.
    for x, y in zip(
        nest.flatten(new_server_state),
        anonymous_tuple.flatten(current_server_state)):
      tf.assign(x, y)
    apply_delta_fn(weights_delta)
    return new_server_state

  return update_model_inner()


def build_model_delta_optimizer_process(model_fn,
                                        model_to_client_delta_fn,
                                        server_optimizer_fn=None):
  """Constructs `tff.utils.IterativeProcess` for Federated Averaging or SGD.

  This provides the TFF orchestration logic connecting the common server logic
  which applies aggregated model deltas to the server model with a ClientDeltaFn
  that specifies how weight_deltas are computed on device.

  Note: We pass in functions rather than constructed objects so we can ensure
  any variables or ops created in constructors are placed in the correct graph.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    model_to_client_delta_fn: A function from a model_fn to a `ClientDeltaFn`.
    server_optimizer_fn: A no-arg function that returns a `tf.Optimizer`. The
      apply_gradients method of this optimizer is used to apply client updates
      to the server model. The default returns a `tf.train.GradientDescent` with
      a learning_rate of 1.0, which simply adds the average client delta to the
      server's model.

  Returns:
    A `tff.utils.IterativeProcess`.
  """
  if server_optimizer_fn is None:
    server_optimizer_fn = lambda: gradient_descent.SGD(learning_rate=1.0)

  # TODO(b/109733734): would be nice not to have the construct a throwaway model
  # here just to get the types.
  # Wrap in a new Graph to prevent pollution.
  with tf.Graph().as_default():
    model = model_utils.enhance(model_fn())

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    no_arg_server_init_fn = lambda: server_init(model_fn, server_optimizer_fn)
    server_init_tf = tff.tf_computation(no_arg_server_init_fn)
    return tff.federated_value(server_init_tf(), tff.SERVER)

  federated_server_state_type = server_init_tff.type_signature.result
  server_state_type = federated_server_state_type.member

  tf_dataset_type = tff.SequenceType(model.input_spec)
  federated_dataset_type = tff.FederatedType(
      tf_dataset_type, tff.CLIENTS, all_equal=False)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round_tff(server_state, federated_dataset):
    """Orchestration logic for one round of optimization.

    Args:
      server_state: a `tff.learning.framework.ServerState` named tuple.
      federated_dataset: a federated `tf.Dataset` with placement tff.CLIENTS.

    Returns:
      An updated `tff.learning.framework.ServerState`.
    """
    model_weights_type = federated_server_state_type.member.model

    @tff.tf_computation(tf_dataset_type, model_weights_type)
    def client_delta_tf(tf_dataset, initial_model_weights):
      """Performs client local model optimization.

      Args:
        tf_dataset: a `tf.data.Dataset` that provides training examples.
        initial_model_weights: a `model_utils.ModelWeights` containing the
          starting weights.

      Returns:
        A `ClientOutput` structure.
      """
      client_delta_fn = model_to_client_delta_fn(model_fn)

      # TODO(b/123092620): this can be removed once AnonymousTuple works with
      # tf.contrib.framework.nest, or the following behavior is moved to
      # anonymous_tuple module.
      if isinstance(initial_model_weights, anonymous_tuple.AnonymousTuple):
        initial_model_weights = anonymous_tuple.flatten(initial_model_weights)
        initial_model_weights = nest.pack_sequence_as(model.weights,
                                                      initial_model_weights)

      client_output = client_delta_fn(tf_dataset, initial_model_weights)
      return client_output

    client_outputs = tff.federated_map(
        client_delta_tf,
        (federated_dataset, tff.federated_broadcast(server_state.model)))

    @tff.tf_computation(server_state_type, model_weights_type.trainable)
    def server_update_model_tf(server_state, model_delta):
      server_update_model(
          server_state,
          model_delta,
          model_fn=model_fn,
          optimizer_fn=server_optimizer_fn)
      return server_state

    round_model_delta = tff.federated_average(
        client_outputs.weights_delta,
        weight=client_outputs.weights_delta_weight)
    # TODO(b/123408447): remove tff.federated_apply and call
    # server_update_model_tf directly once T <-> T@SERVER isomorphism is
    # supported.
    server_state = tff.federated_apply(server_update_model_tf,
                                       (server_state, round_model_delta))

    aggregated_outputs = model.federated_output_computation(
        client_outputs.model_output)
    # Promote the FederatedType outside the NamedTupleType, or return the
    # singluar federated value.
    num_outputs = len(aggregated_outputs)
    if num_outputs == 1:
      aggregated_outputs = aggregated_outputs[0]
    elif num_outputs >= 2:
      aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  return tff.utils.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round_tff)
