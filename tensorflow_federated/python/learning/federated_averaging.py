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
"""An implementation of the Federated Averaging algorithm.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientFedAvg(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging."""

  def __init__(self, model):
    self._model = model_utils.enhance(model)
    py_typecheck.check_type(self._model, model_utils.EnhancedTrainableModel)

  @property
  def variables(self):
    return []

  @tf.contrib.eager.function(autograph=False)
  def __call__(self, dataset, initial_model):
    # N.B. When not in eager mode, this code must be wrapped as a defun
    # as it uses program-order semantics to avoid adding many explicit
    # control dependencies.
    model = self._model
    py_typecheck.check_type(dataset, tf.data.Dataset)

    # TODO(b/120801384): We should initialize model.local_variables here.
    # Or, we may just need a convention that TFF initializes all variables
    # before invoking the TF function.

    # Assign the model variables the init_model:
    nest.map_structure(tf.assign, model.vars, initial_model)

    @tf.contrib.eager.function(autograph=False)
    def reduce_fn(dummy_state, batch):
      """Runs train_on_batch on batch."""
      # Question: Do we want to compute num_examples and num_batches as
      # counters here, so clients don't have to do this in their Models
      # themselves? But then is there potentially more "surprise"
      # when those are available? If we do compute them here, do we add
      # them into the user's `aggregated_outputs` or return them separately?
      model.train_on_batch(batch)
      return dummy_state

    # TODO(b/121400757): Remove dummy_output when bug fixed.
    dummy_output = dataset.reduce(
        initial_state=tf.constant(0.0), reduce_func=reduce_fn)

    model_delta = nest.map_structure(tf.subtract, model.vars.trainable,
                                     initial_model.trainable)

    # TODO(b/109733734): Add a check that all of the model deltas are finite;
    # if not, then send an all-zero update, and increment an error counter.

    return optimizer_utils.ClientOutput(
        model_delta,
        model.aggregated_outputs(),
        tensor_utils.to_odict({'workaround for b/121400757': dummy_output}))


#
# Server TF computations
#


def _create_optimizer_and_server_state(model, optimizer):
  """A helper that constructs the model and optimizer.

  This code is needed both in server_init (to introduce variables so
  we can read off there initial values) and in server_update_model.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A tuple of (apply_delta_fn, server_state), where:
      *  apply_delta_fn is a tensorflow function that takes a model delta and
         updates the model variables as well as possibly optimizer_state
         variables introduced by the optimizer.
      *  server_state is a `ServerState` tuple holding those variables.
  """

  @tf.contrib.eager.defun(autograph=False)
  def apply_delta(delta):
    """Applies delta to model.vars."""
    nest.assert_same_structure(delta, model.vars.trainable)
    grads_and_vars = nest.map_structure(
        lambda x, v: (-1.0 * x, v), nest.flatten(delta),
        nest.flatten(model.vars.trainable))
    # N.B. This may create variables.
    # TODO(b/109733734): In TF 2, you shouldn't create variables
    # inside a defun. Perhaps use Keras optimizers or OptimizerV2?
    optimizer.apply_gradients(grads_and_vars, name='server_update')
    return tf.constant(1)  # We have to return something.

  # Create a dummy input and trace apply_delta so that
  # we can determine the optimizers variables.
  model_delta = nest.map_structure(tf.zeros_like, model.vars.trainable)

  # TODO(b/109733734): We would like to call get_concrete_function,
  # but that does not currently work with structured inputs.
  # For now, we just call the function on dummy input, which
  # still ensures the function is traced (so variables are created).
  apply_delta(delta=model_delta)

  # N.B. Using to_var_dict doesn't work here, because we
  # may get different names.
  optimizer_vars = optimizer.variables()

  return apply_delta, ServerState(model=model.vars,
                                  optimizer_state=optimizer_vars)


# Represents the state of the server carried between rounds.
ServerState = collections.namedtuple(
    'ServerState',
    [
        # A ModelVars structure, containing Tensors or Variables.
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
      Returns A `ServerState` namedtuple.
  """
  model = model_utils.enhance(model_fn())  # Constructs variables
  optimizer = optimizer_fn()  # Might create variables?
  _, server_state = _create_optimizer_and_server_state(model, optimizer)
  return server_state


def server_update_model(server_state, model_delta, model_fn, optimizer_fn):
  """Updates `server_state` based on `model_delta`.

  Args:
    server_state: A `ServerState` namedtuple.
    model_delta: An update to the trainable variables of the model.
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
    nest.map_structure(tf.assign, server_vars, server_state)
    apply_delta_fn(model_delta)
    return server_vars

  return update_model_inner()
