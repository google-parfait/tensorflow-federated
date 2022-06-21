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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""An implementation of the client logic in the Federated Averaging algorithm.

The original Federated Averaging algorithm is proposed by the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import Callable, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientFedAvg(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging."""

  def __init__(
      self,
      model: model_lib.Model,
      optimizer: Union[optimizer_base.Optimizer,
                       Callable[[], tf.keras.optimizers.Optimizer]],
      client_weighting: client_weight_lib.ClientWeightType = client_weight_lib
      .ClientWeighting.NUM_EXAMPLES,
      use_experimental_simulation_loop: bool = False):
    """Creates the client computation for Federated Averaging.

    Note: All variable creation required for the client computation (e.g. model
    variable creation) must occur in during construction, and not during
    `__call__`.

    Args:
      model: A `tff.learning.Model` instance.
      optimizer: A `optimizer_base.Optimizer` instance, or a no-arg callable
        that returns a `tf.keras.Optimizer` instance..
      client_weighting: A value of `tff.learning.ClientWeighting` that specifies
        a built-in weighting method, or a callable that takes the model output
        and returns a tensor that provides the weight in the federated average
        of model deltas.
      use_experimental_simulation_loop: Controls the reduce loop function for
        input dataset. An experimental reduce loop is used for simulation.
    """
    py_typecheck.check_type(model, model_lib.Model)
    self._model = model
    self._optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        optimizer,
        model_utils.ModelWeights.from_model(self._model).trainable,
        disjoint_init_and_next=False)
    client_weight_lib.check_is_client_weighting_or_callable(client_weighting)
    self._client_weighting = client_weighting
    self._dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset, initial_weights):
    model = self._model
    optimizer = self._optimizer
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                          initial_weights)

    def reduce_fn(state, batch):
      """Train `tff.learning.Model` on local client batch."""
      num_examples_sum, optimizer_state = state

      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      optimizer_state, updated_weights = optimizer.next(optimizer_state,
                                                        model_weights.trainable,
                                                        gradients)
      if not isinstance(optimizer, keras_optimizer.KerasOptimizer):
        # Keras optimizer mutates model variables within the `next` step.
        tf.nest.map_structure(lambda a, b: a.assign(b), model_weights.trainable,
                              updated_weights)

      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum, optimizer_state

    def initial_state_for_reduce_fn():
      trainable_tensor_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights.trainable)
      return tf.zeros(
          shape=[],
          dtype=tf.int64), optimizer.initialize(trainable_tensor_specs)

    num_examples_sum, _ = self._dataset_reduce_fn(
        reduce_fn, dataset, initial_state_fn=initial_state_for_reduce_fn)

    weights_delta = tf.nest.map_structure(tf.subtract, model_weights.trainable,
                                          initial_weights.trainable)
    model_output = model.report_local_unfinalized_metrics()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      # TODO(b/176171842): Zeroing has no effect with unweighted aggregation.
      weights_delta_weight = tf.constant(0.0)
    elif self._client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
    elif self._client_weighting is client_weight_lib.ClientWeighting.UNIFORM:
      weights_delta_weight = tf.constant(1.0)
    else:
      weights_delta_weight = self._client_weighting(model_output)
    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, model_output, optimizer_output=())
