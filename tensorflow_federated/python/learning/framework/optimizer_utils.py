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
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils

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

  @tf.contrib.eager.defun(autograph=False)
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


def state_with_new_model_weights(server_state, trainable_weights,
                                 non_trainable_weights):
  """Returns a `ServerState` with updated model weights.

  Args:
    server_state: A server state object returned by an iterative training
      process like `tff.learning.build_federated_averaging_process`.
    trainable_weights: A list of numpy arrays in the order of the original
      model's `trainable_variables`.
    non_trainable_weights: A list of numpy arrays in the order of the original
      model's `non_trainable_variables`.

  Returns:
    A new server state object which can be passed to the `next` method of
    the iterative process.
  """
  # TODO(b/123092620): Simplify this.
  py_typecheck.check_type(server_state, anonymous_tuple.AnonymousTuple)

  def pack_values(old, new_values, name):
    """Packs new_values in an OrderedDict matching old."""
    if len(old) != len(new_values):
      raise ValueError('Lengths differ for {} weights: {} vs {}'.format(
          name, len(old), len(new_values)))
    tuples = []
    for (key, old_value), new_value in zip(
        anonymous_tuple.to_elements(old), new_values):
      if (old_value.dtype != new_value.dtype or
          old_value.shape != new_value.shape):
        raise ValueError('The shapes or dtypes do not match for {} weight {}:\n'
                         'current weights: shape {} dtype {}\n'
                         '    new weights: shape {} dtype {}'.format(
                             name, key, old_value.shape, old_value.dtype,
                             new_value.shape, new_value.dtype))

      tuples.append((key, new_value))
    return collections.OrderedDict(tuples)

  renamed_new_weights = model_utils.ModelWeights(
      trainable=pack_values(server_state.model.trainable, trainable_weights,
                            'trainable'),
      non_trainable=pack_values(server_state.model.non_trainable,
                                non_trainable_weights, 'non_trainable'))
  return ServerState(
      model=renamed_new_weights, optimizer_state=server_state.optimizer_state)


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
  py_typecheck.check_type(current_server_state, ServerState)
  py_typecheck.check_type(weights_delta, collections.OrderedDict)
  model = model_utils.enhance(model_fn())
  optimizer = optimizer_fn()
  apply_delta_fn, server_state_vars = _create_optimizer_and_server_state(
      model, optimizer)

  # We might have a NaN value e.g. if all of the clients processed
  # had no data, so the denominator in the federated_average is zero.
  # If we see any NaNs, zero out the whole update.
  no_nan_weights_delta, _ = tensor_utils.zero_all_if_any_non_finite(
      weights_delta)
  # TODO(b/124538167): We should increment a server counter to
  # track the fact a non-finite weiths_delta was encountered.

  @tf.contrib.eager.function(autograph=False)
  def update_model_inner():
    """Applies the update."""
    nest.map_structure(tf.assign, server_state_vars, current_server_state)
    apply_delta_fn(no_nan_weights_delta)
    return server_state_vars

  return update_model_inner()


#
# N. B. All functions above this should be standard TensorFlow, in
# the remainder of this file we use TFF specific concepts to bind
# the TensorFlow building blocks into a federated computation.
#


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

  # TODO(b/122081673): would be nice not to have the construct a throwaway model
  # here just to get the types. After fully moving to TF2.0 and eager-mode, we
  # should re-evaluate what happens here and where `g` is used below.
  with tf.Graph().as_default() as g:
    dummy_model_for_metadata = model_utils.enhance(model_fn())

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    no_arg_server_init_fn = lambda: server_init(model_fn, server_optimizer_fn)
    server_init_tf = tff.tf_computation(no_arg_server_init_fn)
    return tff.federated_value(server_init_tf(), tff.SERVER)

  federated_server_state_type = server_init_tff.type_signature.result
  server_state_type = federated_server_state_type.member

  tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)
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
      A tuple of updated `tff.learning.framework.ServerState` and the result of
    `tff.learning.Model.federated_output_computation`.
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
        initial_model_weights = model_utils.ModelWeights.from_tff_value(
            initial_model_weights)

      client_output = client_delta_fn(tf_dataset, initial_model_weights)
      return client_output

    client_outputs = tff.federated_map(
        client_delta_tf,
        (federated_dataset, tff.federated_broadcast(server_state.model)))

    @tff.tf_computation(server_state_type, model_weights_type.trainable)
    def server_update_model_tf(server_state, model_delta):
      """Converts args to correct python types and calls server_update_model."""
      # We need to convert TFF types to the types server_update_model expects.
      # TODO(b/123092620): Mixing AnonymousTuple with other nested types is not
      # pretty, fold this into anonymous_tuple module or get working with
      # tf.contrib.framework.nest.
      py_typecheck.check_type(model_delta, anonymous_tuple.AnonymousTuple)
      model_delta = anonymous_tuple.to_odict(model_delta)
      py_typecheck.check_type(server_state, anonymous_tuple.AnonymousTuple)
      server_state = ServerState(
          model=model_utils.ModelWeights.from_tff_value(server_state.model),
          optimizer_state=list(server_state.optimizer_state))

      return server_update_model(
          server_state,
          model_delta,
          model_fn=model_fn,
          optimizer_fn=server_optimizer_fn)

    # TODO(b/124070381): We hope to remove this explicit cast once we have a
    # full solution for type analysis in multiplications and divisions
    # inside TFF
    fed_weight_type = client_outputs.weights_delta_weight.type_signature.member
    py_typecheck.check_type(fed_weight_type, tff.TensorType)
    if fed_weight_type.dtype.is_integer:

      @tff.tf_computation(fed_weight_type)
      def _cast_to_float(x):
        return tf.cast(x, tf.float32)

      weight_denom = tff.federated_map(_cast_to_float,
                                       client_outputs.weights_delta_weight)
    else:
      weight_denom = client_outputs.weights_delta_weight
    round_model_delta = tff.federated_average(
        client_outputs.weights_delta, weight=weight_denom)

    # TODO(b/123408447): remove tff.federated_apply and call
    # server_update_model_tf directly once T <-> T@SERVER isomorphism is
    # supported.
    server_state = tff.federated_apply(server_update_model_tf,
                                       (server_state, round_model_delta))

    # Re-use graph used to construct `model`, since it has the variables, which
    # need to be read in federated_output_computation to get the correct shapes
    # and types for the federated aggregation.
    with g.as_default():
      aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
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
