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
"""An implementation of Federated Averaging with client optimizer scheduling."""

import collections
from collections.abc import Callable
from typing import Optional, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning.algorithms import fed_avg
from tensorflow_federated.python.learning.metrics import aggregator as metric_aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work

TFFOrKerasOptimizer = Union[
    optimizer_base.Optimizer, tf.keras.optimizers.Optimizer
]


def build_scheduled_client_work(
    model_fn: Callable[[], variable.VariableModel],
    learning_rate_fn: Callable[[int], float],
    optimizer_fn: Callable[[float], TFFOrKerasOptimizer],
    metrics_aggregator: Callable[
        [
            types.MetricFinalizersType,
            computation_types.StructWithPythonType,
        ],
        computation_base.Computation,
    ],
    use_experimental_simulation_loop: bool = False,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for federated averaging.

  This `ClientWorkProcess` creates a state containing the current round number,
  which is incremented at each call to `ClientWorkProcess.next`. This integer
  round number is used to call `optimizer_fn(round_num)`, in order to construct
  the proper optimizer.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    learning_rate_fn: A callable accepting an integer round number and returning
      a float to be used as a learning rate for the optimizer. That is, the
      client work will call `optimizer_fn(learning_rate_fn(round_num))` where
      `round_num` is the integer round number.
    optimizer_fn: A callable accepting a float learning rate, and returning a
      `tff.learning.optimizers.Optimizer` or a `tf.keras.Optimizer`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `ClientWorkProcess`.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    whimsy_model = model_fn()
    whimsy_optimizer = optimizer_fn(1.0)
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        whimsy_model.report_local_unfinalized_metrics()
    )
    metrics_aggregation_fn = metrics_aggregator(
        whimsy_model.metric_finalizers(), unfinalized_metrics_type
    )
  data_type = computation_types.SequenceType(whimsy_model.input_spec)
  weights_type = model_weights.weights_type_from_model(whimsy_model)

  if isinstance(whimsy_optimizer, optimizer_base.Optimizer):
    build_client_update_fn = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer
    )
  else:
    build_client_update_fn = (
        model_delta_client_work.build_model_delta_update_with_keras_optimizer
    )

  @tensorflow_computation.tf_computation(weights_type, data_type, tf.int32)
  def client_update_computation(initial_model_weights, dataset, round_num):
    learning_rate = learning_rate_fn(round_num)
    optimizer = optimizer_fn(learning_rate)
    client_update = build_client_update_fn(
        model_fn=model_fn,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=use_experimental_simulation_loop,
    )
    return client_update(optimizer, initial_model_weights, dataset)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value(0, placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32)
  @tf.function
  def add_one(x):
    return x + 1

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type),
  )
  def next_fn(state, weights, client_data):
    round_num_at_clients = intrinsics.federated_broadcast(state)
    client_result, model_outputs = intrinsics.federated_map(
        client_update_computation, (weights, client_data, round_num_at_clients)
    )
    updated_state = intrinsics.federated_map(add_one, state)
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        updated_state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def build_weighted_fed_avg_with_optimizer_schedule(
    model_fn: Callable[[], variable.VariableModel],
    client_learning_rate_fn: Callable[[int], float],
    client_optimizer_fn: Callable[[float], TFFOrKerasOptimizer],
    server_optimizer_fn: Union[
        optimizer_base.Optimizer, Callable[[], tf.keras.optimizers.Optimizer]
    ] = fed_avg.DEFAULT_SERVER_OPTIMIZER_FN,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    metrics_aggregator: Optional[
        Callable[
            [
                types.MetricFinalizersType,
                computation_types.StructWithPythonType,
            ],
            computation_base.Computation,
        ]
    ] = None,
    use_experimental_simulation_loop: bool = False,
) -> learning_process.LearningProcess:
  """Builds a learning process for FedAvg with client optimizer scheduling.

  This function creates a `LearningProcess` that performs federated averaging on
  client models. The iterative process has the following methods inherited from
  `LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `LearningAlgorithmState` representing the
      initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as aggregated
      metrics at the server, including client training metrics and any other
      metrics from distribution and aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next`, and `M` represents the type
      of the model weights used during training.
  *   `set_model_weights`: A `tff.Computation` with type signature
      `(<S, M> -> S)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `M` represents the type of the model weights
      used during training.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. For each client, local training is
  performed using `client_optimizer_fn`. Each client computes the difference
  between the client model after training and the initial broadcast model.
  These model deltas are then aggregated at the server using a weighted
  aggregation function. Clients weighted by the number of examples they see
  thoughout local training. The aggregate model delta is applied at the server
  using a server optimizer.

  The primary purpose of this implementation of FedAvg is that it allows for the
  client optimizer to be scheduled across rounds. The process keeps track of how
  many iterations of `.next` have occurred (starting at `0`), and for each such
  `round_num`, the clients will use `client_optimizer_fn(round_num)` to perform
  local optimization. This allows learning rate scheduling (eg. starting with
  a large learning rate and decaying it over time) as well as a small learning
  rate (eg. switching optimizers as learning progresses).

  Note: the default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 1.0, which corresponds to adding the model delta to
  the current server model. This recovers the original FedAvg algorithm in
  [McMahan et al., 2017](https://arxiv.org/abs/1602.05629). More
  sophisticated federated averaging procedures may use different learning rates
  or server optimizers.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    client_learning_rate_fn: A callable accepting an integer round number and
      returning a float to be used as a learning rate for the optimizer. The
      client work will call `optimizer_fn(learning_rate_fn(round_num))` where
      `round_num` is the integer round number. Note that the round numbers
      supplied will start at `0` and increment by one each time `.next` is
      called on the resulting process. Also note that this function must be
      serializable by TFF.
    client_optimizer_fn: A callable accepting a float learning rate, and
      returning a `tff.learning.optimizers.Optimizer` or a `tf.keras.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. By default, this uses
      `tf.keras.optimizers.SGD` with a learning rate of 1.0.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics. If
      `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `LearningProcess`.
  """
  py_typecheck.check_callable(model_fn)

  @tensorflow_computation.tf_computation()
  def initial_model_weights_fn():
    return model_weights.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)

  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)
  aggregator = model_aggregator.create(
      model_weights_type.trainable, computation_types.TensorType(tf.float32)
  )
  process_signature = aggregator.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError(
        '`model_update_aggregation_factory` does not produce a '
        'compatible `AggregationProcess`. The processes must '
        'retain the type structure of the inputs on the '
        f'server, but got {input_client_value_type.member} != '
        f'{result_server_value_type.member}.'
    )

  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize
  client_work = build_scheduled_client_work(
      model_fn,
      client_learning_rate_fn,
      client_optimizer_fn,
      metrics_aggregator,
      use_experimental_simulation_loop,
  )
  finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type
  )
  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      aggregator,
      finalizer,
  )
