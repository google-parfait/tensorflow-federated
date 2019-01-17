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

# Dependency imports

import six
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.learning import model_utils

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
      *  server_state is a `ServerState` tuple holding those variables.
  """

  @tf.contrib.eager.defun(autograph=False)
  def apply_delta(delta):
    """Applies delta to model.weights."""
    tf.contrib.framework.nest.assert_same_structure(delta,
                                                    model.weights.trainable)
    grads_and_vars = tf.contrib.framework.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.contrib.framework.nest.flatten(delta),
        tf.contrib.framework.nest.flatten(model.weights.trainable))
    # N.B. This may create variables.
    # TODO(b/109733734): Perhaps use Keras optimizers or OptimizerV2?
    optimizer.apply_gradients(grads_and_vars, name='server_update')
    return tf.constant(1)  # We have to return something.

  # Create a dummy input and trace apply_delta so that
  # we can determine the optimizer's variables.
  weights_delta = tf.contrib.framework.nest.map_structure(
      tf.zeros_like, model.weights.trainable)

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
  """Returns initial `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`.

  Returns:
    A `ServerState` namedtuple.
  """
  model = model_utils.enhance(model_fn())  # Constructs variables
  optimizer = optimizer_fn()  # Might create variables?
  _, server_state = _create_optimizer_and_server_state(model, optimizer)
  return server_state


def server_update_model(server_state, weights_delta, model_fn, optimizer_fn):
  """Updates `server_state` based on `weights_delta`.

  Args:
    server_state: A `ServerState` namedtuple.
    weights_delta: An update to the trainable variables of the model.
    model_fn: A no-arg function that returns a `tff.learning.Model`. Passing in
      a function ensures any variables are created when server_update_model is
      called, so they can be captured in a specific graph or other context.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`. As with
      model_fn, we pass in a function to control when variables are created.

  Returns:
    An updated `ServerState` namedtuple.
  """
  model = model_utils.enhance(model_fn())  # Constructs variables
  optimizer = optimizer_fn()  # Might create variables?
  apply_delta_fn, server_vars = _create_optimizer_and_server_state(
      model, optimizer)

  @tf.contrib.eager.function(autograph=False)
  def update_model_inner():
    tf.contrib.framework.nest.map_structure(tf.assign, server_vars,
                                            server_state)
    apply_delta_fn(weights_delta)
    return server_vars

  return update_model_inner()


def build_model_delta_optimizer_tff(model_fn,
                                    model_to_client_delta_fn,
                                    server_optimizer_fn=None):
  """Constructs complete TFF computations for Federated Averaging or SGD.

  This provides the TFF orchestration logic connecting the common server logic
  which applies aggregated model deltas to the server model with a ClientDeltaFn
  that specifies how weight_deltas are computed on device.

  Note: We pass in functions rather than constructed objects so we can ensure
  any variables or ops created in constructors are placed in the correct graph.
  TODO(b/122081673): This can be simplified once we move fully to TF 2.0.

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
    # TODO(b/121287923): Using an optimizer without variables is blocked
    # on supporing NamedTuples with no elements. This should be vanilla
    # tf.train.GradientDescent.
    server_optimizer_fn = lambda: tf.train.MomentumOptimizer(  # pylint: disable=g-long-lambda
        learning_rate=1.0, momentum=0.0)

  @tff.tf_computation
  def server_init_tff():
    return server_init(model_fn, server_optimizer_fn)

  server_state_type = server_init_tff.type_signature.result
  model_delta_type = server_init_tff.type_signature.result.model

  # TODO(b/109733734): Complete FedAvg orchestration. Currently blocked
  # by the federated_computation and federated types not being implemented.
  # server_state_type = tff.FederatedType(server_state_type, tff.SERVER, True)
  # model_delta_type = tff.FederatedType(model_delta_type, tff.SERVER, True)

  # @tff.federated_computation((server_state_type, model_delta_type))
  @tff.tf_computation((server_state_type, model_delta_type))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of optimization."""
    del federated_dataset  # Currently unused.

    @tff.tf_computation
    def client_tf_tff(dataset, initial_weights):  # pylint:disable=unused-variable
      client_delta_fn = model_to_client_delta_fn(model_fn)
      return client_delta_fn(dataset, initial_weights)

    @tff.tf_computation
    def server_update_model_tff(server_state, model_delta):  # pylint:disable=unused-variable
      server_update_model(
          server_state,
          model_delta,
          model_fn=model_fn,
          optimizer_fn=server_optimizer_fn)

    # TODO(b/109733734): Complete FedAvg orchestration. Currently blocked
    # by errors on calling:
    # client_output = tff.federated_map(
    #    (federated_dataset, tff.federated_broadcast(server_state.model)),
    #    client_tf_tff)

    # TODO(b/109733734): Finish implementing this function:
    # * Pass Compute federated_averages and other aggregates of client_output.
    # * TODO(b/120147094): Compute aggregate metrics based on
    #   model.federated_output_computation().
    # * Pass the model_delta into server_update_model
    # * Return the updated state.

    return server_state

  return tff.utils.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round)
