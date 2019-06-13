# Lint as: python3
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

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientOutput(
    collections.namedtuple('ClientOutput', [
        'weights_delta', 'weights_delta_weight', 'model_output',
        'optimizer_output'
    ])):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: a dictionary of updates to the model's trainable
      variables.
  -   `weights_delta_weight`: weight to use in a weighted mean when aggregating
      `weights_delta`.
  -   `model_output`: a structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: additional metrics or other outputs defined by the
      optimizer.
  """
  __slots__ = ()


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
        `tff.learning.Model.forward_pass`.
      initial_weights: A dictionary of initial values for all trainable and
        non-trainable model variables, keyed by name. This will be supplied by
        the server in Federated Averaging.

    Returns:
      An `optimizer_utils.ClientOutput` namedtuple.
    """
    pass


def _build_server_optimizer(model, optimizer):
  """A helper for server computations that constructs  the optimizer.

  This code is needed both in server_init (to introduce variables so
  we can read their initial values) and in server_update_model.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A tuple of (apply_delta_fn, optimizer_vars), where:
      *  apply_delta_fn is a TensorFlow function that takes a model delta and
         updates the trainable weights of `model` as well as possibly
         optimizer_state variables introduced by the optimizer.
      *  optimizer_vars is a list of optimizer variables.
  """

  @tf.function
  def apply_delta(delta):
    """Applies `delta` to `model.weights`."""
    tf.nest.assert_same_structure(delta, model.weights.trainable)
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
        tf.nest.flatten(model.weights.trainable))
    # N.B. This may create variables.
    optimizer.apply_gradients(grads_and_vars, name='server_update')
    return tf.constant(1)  # We have to return something.

  # Create a dummy input and trace apply_delta so that
  # we can determine the optimizer's variables.
  weights_delta = tf.nest.map_structure(tf.zeros_like, model.weights.trainable)

  # TODO(b/109733734): We would like to call get_concrete_function,
  # but that does not currently work with structured inputs.
  # For now, we just call the function on dummy input, which
  # still ensures the function is traced (so variables are created).
  apply_delta(delta=weights_delta)

  # N.B. Using to_var_dict doesn't work here, because we
  # may get non-unique names, so we just use a flat list.
  optimizer_vars = optimizer.variables()

  return (apply_delta, optimizer_vars)


# Represents the state of the server carried between rounds.
ServerState = collections.namedtuple(
    'ServerState',
    [
        # A ModelWeights structure, containing Tensors or Variables.
        'model',
        # A list of Tensors or Variables, in the order
        # returned by optimizer.variables()
        'optimizer_state',
        # State (possibly empty) of the delta_aggregate_fn.
        'delta_aggregate_state',
        # State (possibly empty) of the model_broadcast_fn.
        'model_broadcast_state'
    ])


def state_with_new_model_weights(server_state, trainable_weights,
                                 non_trainable_weights):
  """Returns a `ServerState` with updated model weights.

  Args:
    server_state: A server state object returned by an iterative training
      process like `tff.learning.build_federated_averaging_process`.
    trainable_weights: A list of `numpy` arrays in the order of the original
      model's `trainable_variables`.
    non_trainable_weights: A list of `numpy` arrays in the order of the original
      model's `non_trainable_variables`.

  Returns:
    A new server `ServerState` object which can be passed to the `next` method
    of the iterative process.
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
  # TODO(b/123092620): We can't use tff.utils.update_state because this
  # is an AnonymousTuple, not a ServerState. We should do something
  # that doesn't mention every entry in the state.
  return ServerState(
      model=renamed_new_weights,
      optimizer_state=server_state.optimizer_state,
      delta_aggregate_state=server_state.delta_aggregate_state,
      model_broadcast_state=server_state.model_broadcast_state)


def server_init(model_fn, optimizer_fn, delta_aggregate_state,
                model_broadcast_state):
  """Returns initial `tff.learning.framework.ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`.
    delta_aggregate_state: The initial state of the delta_aggregator.
    model_broadcast_state: The initial state of the model_broadcaster.

  Returns:
    A `tff.learning.framework.ServerState` namedtuple.
  """
  model = model_utils.enhance(model_fn())
  optimizer = optimizer_fn()
  _, optimizer_vars = _build_server_optimizer(model, optimizer)
  return ServerState(
      model=model.weights,
      optimizer_state=optimizer_vars,
      delta_aggregate_state=delta_aggregate_state,
      model_broadcast_state=model_broadcast_state)


def server_update_model(server_state, weights_delta, model_fn, optimizer_fn):
  """Updates `server_state` based on `weights_delta`.

  Args:
    server_state: A `tff.learning.framework.ServerState` namedtuple, the state
      to be updated.
    weights_delta: An update to the trainable variables of the model.
    model_fn: A no-arg function that returns a `tff.learning.Model`. Passing in
      a function ensures any variables are created when server_update_model is
      called, so they can be captured in a specific graph or other context.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`. As with
      model_fn, we pass in a function to control when variables are created.

  Returns:
    An updated `tff.learning.framework.ServerState`.
  """
  py_typecheck.check_type(server_state, ServerState)
  py_typecheck.check_type(weights_delta, collections.OrderedDict)
  model = model_utils.enhance(model_fn())
  optimizer = optimizer_fn()
  apply_delta_fn, optimizer_vars = _build_server_optimizer(model, optimizer)

  # We might have a NaN value e.g. if all of the clients processed
  # had no data, so the denominator in the federated_mean is zero.
  # If we see any NaNs, zero out the whole update.
  no_nan_weights_delta, _ = tensor_utils.zero_all_if_any_non_finite(
      weights_delta)
  # TODO(b/124538167): We should increment a server counter to
  # track the fact a non-finite weights_delta was encountered.

  @tf.function
  def update_model_inner():
    """Applies the update."""
    tf.nest.map_structure(lambda a, b: a.assign(b),
                          (model.weights, optimizer_vars),
                          (server_state.model, server_state.optimizer_state))
    apply_delta_fn(no_nan_weights_delta)
    return model.weights, optimizer_vars

  model_weights, optimizer_vars = update_model_inner()
  # TODO(b/123092620): We must do this outside of the above tf.function, because
  # there could be an AnonymousTuple hiding in server_state,
  # and tf.function's can't return AnonymousTuples.
  return tff.utils.update_state(
      server_state, model=model_weights, optimizer_state=optimizer_vars)


#
# N. B. All functions above this should be standard TensorFlow, in
# the remainder of this file we use TFF specific concepts to bind
# the TensorFlow building blocks into a federated computation.
#


def build_stateless_mean():
  """Just tff.federated_mean with empty state, to use as a default."""
  return tff.utils.StatefulAggregateFn(
      initialize_fn=lambda: (),
      next_fn=lambda state, value, weight=None: (  # pylint: disable=g-long-lambda
          state, tff.federated_mean(value, weight=weight)))


def build_stateless_broadcaster():
  """Just tff.federated_broadcast with empty state, to use as a default."""
  return tff.utils.StatefulBroadcastFn(
      initialize_fn=lambda: (),
      next_fn=lambda state, value: (  # pylint: disable=g-long-lambda
          state, tff.federated_broadcast(value)))


def build_model_delta_optimizer_process(
    model_fn,
    model_to_client_delta_fn,
    server_optimizer_fn,
    stateful_delta_aggregate_fn=build_stateless_mean(),
    stateful_model_broadcast_fn=build_stateless_broadcaster()):
  """Constructs `tff.utils.IterativeProcess` for Federated Averaging or SGD.

  This provides the TFF orchestration logic connecting the common server logic
  which applies aggregated model deltas to the server model with a
  `ClientDeltaFn` that specifies how `weight_deltas` are computed on device.

  Note: We pass in functions rather than constructed objects so we can ensure
  any variables or ops created in constructors are placed in the correct graph.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    model_to_client_delta_fn: A function from a `model_fn` to a `ClientDeltaFn`.
    server_optimizer_fn: A no-arg function that returns a `tf.Optimizer`. The
      `apply_gradients` method of this optimizer is used to apply client updates
      to the server model.
    stateful_delta_aggregate_fn: A `tff.utils.StatefulAggregateFn` where the
      `next_fn` performs a federated aggregation and upates state. That is, it
      has TFF type `(state@SERVER, value@CLIENTS, weights@CLIENTS) ->
      (state@SERVER, aggregate@SERVER)`, where the `value` type is
      `tff.learning.framework.ModelWeights.trainable` corresponding to the
      object returned by `model_fn`.
    stateful_model_broadcast_fn: A `tff.utils.StatefulBroadcastFn` where the
      `next_fn` performs a federated broadcast and upates state. That is, it has
      TFF type `(state@SERVER, value@SERVER) -> (state@SERVER, value@CLIENTS)`,
      where the `value` type is `tff.learning.framework.ModelWeights`
      corresponding to the object returned by `model_fn`.

  Returns:
    A `tff.utils.IterativeProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(model_to_client_delta_fn)
  py_typecheck.check_callable(server_optimizer_fn)
  py_typecheck.check_type(stateful_delta_aggregate_fn,
                          tff.utils.StatefulAggregateFn)
  py_typecheck.check_type(stateful_model_broadcast_fn,
                          tff.utils.StatefulBroadcastFn)

  # TODO(b/122081673): would be nice not to have the construct a throwaway model
  # here just to get the types. After fully moving to TF2.0 and eager-mode, we
  # should re-evaluate what happens here.
  with tf.Graph().as_default():
    dummy_model_for_metadata = model_utils.enhance(model_fn())

  # ===========================================================================
  # TensorFlow Computations

  @tff.tf_computation
  def tf_init_fn():
    return server_init(model_fn, server_optimizer_fn,
                       stateful_delta_aggregate_fn.initialize(),
                       stateful_model_broadcast_fn.initialize())

  tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)
  server_state_type = tf_init_fn.type_signature.result

  @tff.tf_computation(tf_dataset_type, server_state_type.model)
  def tf_client_delta(tf_dataset, initial_model_weights):
    """Performs client local model optimization.

    Args:
      tf_dataset: a `tf.data.Dataset` that provides training examples.
      initial_model_weights: a `model_utils.ModelWeights` containing the
        starting weights.

    Returns:
      A `ClientOutput` structure.
    """
    client_delta_fn = model_to_client_delta_fn(model_fn)
    client_output = client_delta_fn(tf_dataset, initial_model_weights)
    return client_output

  @tff.tf_computation(server_state_type, server_state_type.model.trainable,
                      server_state_type.delta_aggregate_state,
                      server_state_type.model_broadcast_state)
  def tf_server_update(server_state, model_delta, new_delta_aggregate_state,
                       new_broadcaster_state):
    """Converts args to correct python types and calls server_update_model."""
    py_typecheck.check_type(server_state, ServerState)
    server_state = ServerState(
        model=server_state.model,
        optimizer_state=list(server_state.optimizer_state),
        delta_aggregate_state=new_delta_aggregate_state,
        model_broadcast_state=new_broadcaster_state)

    return server_update_model(
        server_state,
        model_delta,
        model_fn=model_fn,
        optimizer_fn=server_optimizer_fn)

  weight_type = tf_client_delta.type_signature.result.weights_delta_weight

  @tff.tf_computation(weight_type)
  def _cast_weight_to_float(x):
    return tf.cast(x, tf.float32)

  # ===========================================================================
  # Federated Computations

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(tf_init_fn(), tff.SERVER)

  federated_server_state_type = server_init_tff.type_signature.result
  federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

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
    new_broadcaster_state, client_model = stateful_model_broadcast_fn(
        server_state.model_broadcast_state, server_state.model)

    client_outputs = tff.federated_map(tf_client_delta,
                                       (federated_dataset, client_model))

    # TODO(b/124070381): We hope to remove this explicit cast once we have a
    # full solution for type analysis in multiplications and divisions
    # inside TFF
    weight_denom = tff.federated_map(_cast_weight_to_float,
                                     client_outputs.weights_delta_weight)
    new_delta_aggregate_state, round_model_delta = stateful_delta_aggregate_fn(
        server_state.delta_aggregate_state,
        client_outputs.weights_delta,
        weight=weight_denom)

    # TODO(b/123408447): remove tff.federated_apply and call
    # tf_server_update directly once T <-> T@SERVER isomorphism is
    # supported.
    server_state = tff.federated_apply(
        tf_server_update, (server_state, round_model_delta,
                           new_delta_aggregate_state, new_broadcaster_state))

    aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
        client_outputs.model_output)

    # Promote the FederatedType outside the NamedTupleType
    aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  return tff.utils.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round_tff)
