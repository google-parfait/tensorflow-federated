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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A simple implementation of federated evaluation."""

import collections
from typing import Callable, Optional

from absl import logging
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning.metrics import aggregator

# Convenience aliases.
SequenceType = computation_types.SequenceType


def build_local_evaluation(
    model_fn: Callable[[], model_lib.Model],
    model_weights_type: computation_types.StructType,
    batch_type: computation_types.Type,
    use_experimental_simulation_loop: bool = False
) -> computation_base.Computation:
  """Builds the local TFF computation for evaluation of the given model.

  This produces an unplaced function that evaluates a `tff.learning.Model`
  on a `tf.data.Dataset`. This function can be mapped to placed data, i.e.
  is mapped to client placed data in `build_federated_evaluation`.

  The TFF type notation for the returned computation is:

  ```
  (<M, D*> → <local_outputs=N, num_examples=tf.int64>)
  ```

  Where `M` is the model weights type structure, `D` is the type structure of a
  single data point, and `N` is the type structure of the local metrics.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    model_weights_type: The `tff.Type` of the model parameters that will be used
      to initialize the model during evaluation.
    batch_type: The type of one entry in the dataset.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and sequential data, and returns the evaluation metrics.
  """

  @computations.tf_computation(model_weights_type, SequenceType(batch_type))
  @tf.function
  def client_eval(incoming_model_weights, dataset):
    """Returns local outputs after evaluting `model_weights` on `dataset`."""
    with tf.init_scope():
      model = model_fn()
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          incoming_model_weights)

    def reduce_fn(num_examples, batch):
      model_output = model.forward_pass(batch, training=False)
      if model_output.num_examples is None:
        # Compute shape from the size of the predictions if model didn't use the
        # batch size.
        return num_examples + tf.shape(
            model_output.predictions, out_type=tf.int64)[0]
      else:
        return num_examples + tf.cast(model_output.num_examples, tf.int64)

    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    num_examples = dataset_reduce_fn(reduce_fn, dataset,
                                     lambda: tf.zeros([], dtype=tf.int64))
    # TODO(b/202027089): Remove this try/except logic once all models do not
    # implement `report_local_outputs` and `federated_output_computation`.
    try:
      model_output = model.report_local_outputs()
      logging.warning(
          'DeprecationWarning: `report_local_outputs` and '
          '`federated_output_computation` are deprecated and will be removed '
          'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
          '`metric_finalizers` instead. The cross-client metrics aggregation '
          'should be specified as the `metrics_aggregator` argument when you '
          'build a training process or evaluation computation using this model.'
      )
    except NotImplementedError:
      model_output = model.report_local_unfinalized_metrics()
    return collections.OrderedDict(
        local_outputs=model_output, num_examples=num_examples)

  return client_eval


# TODO(b/202027089): Remove the note on `metrics_aggregator` once all models do
# not implement `report_local_outputs` and `federated_output_computation`.
def build_federated_evaluation(
    model_fn: Callable[[], model_lib.Model],
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    metrics_aggregator: Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation] = aggregator.sum_then_finalize,
    use_experimental_simulation_loop: bool = False,
) -> computation_base.Computation:
  """Builds the TFF computation for federated evaluation of the given model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENTS)` and have empty state. If
      set to default None, the server model is broadcast to the clients using
      the default tff.federated_broadcast.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a federated TFF computation of the following type signature
      `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`. Default
      is `tff.learning.metrics.sum_then_finalize`, which returns a federated TFF
      computation that sums the unfinalized metrics from `CLIENTS`, and then
      applies the corresponding metric finalizers at `SERVER`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the evaluation metrics.
    Note that if `federated_output_computation` and `report_local_outputs` are
    implemented in `model_fn()` (these two methods are deprecated and will be
    removed in 2022Q1), then the `metrics_aggregator` argument will be ignored,
    and the aggregated evaluation metrics are the result of applying
    `federated_output_computation` on the clients' local outputs. Otherwise,
    if calling `federated_output_computation` and `report_local_outputs` on the
    `model_fn()` throw `NotImplementedError`, then the `metrics_aggregator`
    argument will be used to generate the metrics aggregation computation, which
    is then applied to the clients' local unfinalized metrics.
  """
  if broadcast_process is not None:
    if not isinstance(broadcast_process, measured_process.MeasuredProcess):
      raise ValueError('`broadcast_process` must be a `MeasuredProcess`, got '
                       f'{type(broadcast_process)}.')
    if optimizer_utils.is_stateful_process(broadcast_process):
      raise ValueError(
          'Cannot create a federated evaluation with a stateful '
          'broadcast process, must be stateless, has state: '
          f'{broadcast_process.initialize.type_signature.result!r}')
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO(b/124477628): Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_utils.weights_type_from_model(model)
    batch_type = computation_types.to_type(model.input_spec)
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        model.report_local_unfinalized_metrics())
    # TODO(b/202027089): Remove this try/except logic once all models do not
    # implement `report_local_outputs` and `federated_output_computation`.
    try:
      metrics_aggregation_computation = model.federated_output_computation
      logging.warning(
          'DeprecationWarning: `report_local_outputs` and '
          '`federated_output_computation` are deprecated and will be removed '
          'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
          '`metric_finalizers` instead. The cross-client metrics aggregation '
          'should be specified as the `metrics_aggregator` argument when you '
          'build a training process or evaluation computation using this model.'
      )
    except NotImplementedError:
      metrics_aggregation_computation = metrics_aggregator(
          model.metric_finalizers(), unfinalized_metrics_type)

  @computations.federated_computation(
      computation_types.at_server(model_weights_type),
      computation_types.at_clients(SequenceType(batch_type)))
  def server_eval(server_model_weights, federated_dataset):
    client_eval = build_local_evaluation(model_fn, model_weights_type,
                                         batch_type,
                                         use_experimental_simulation_loop)
    if broadcast_process is not None:
      # TODO(b/179091838): Zip the measurements from the broadcast_process with
      # the result of `model.federated_output_computation` below to avoid
      # dropping these metrics.
      broadcast_output = broadcast_process.next(broadcast_process.initialize(),
                                                server_model_weights)
      client_outputs = intrinsics.federated_map(
          client_eval, (broadcast_output.result, federated_dataset))
    else:
      client_outputs = intrinsics.federated_map(client_eval, [
          intrinsics.federated_broadcast(server_model_weights),
          federated_dataset
      ])
    model_metrics = metrics_aggregation_computation(
        client_outputs.local_outputs)
    statistics = collections.OrderedDict(
        num_examples=intrinsics.federated_sum(client_outputs.num_examples))
    return intrinsics.federated_zip(
        collections.OrderedDict(eval=model_metrics, stat=statistics))

  return server_eval
