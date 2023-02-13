# Copyright 2022, The TensorFlow Federated Authors.
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
"""An implementation of stateful federated evaluation."""

import collections
from collections.abc import Callable
from typing import Optional, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import federated_evaluation
from tensorflow_federated.python.learning.metrics import aggregation_factory
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process


_AggregationProcess = aggregation_process.AggregationProcess


def _build_fed_eval_client_work(
    model_fn: Callable[[], variable.VariableModel],
    metrics_aggregation_process: Optional[_AggregationProcess],
    model_weights_type: computation_types.StructType,
    use_experimental_simulation_loop: bool = False,
) -> client_works.ClientWorkProcess:
  """Builds a `ClientWorkProcess` that performs model evaluation at clients."""

  with tf.Graph().as_default():
    model = model_fn()
    batch_type = computation_types.to_type(model.input_spec)
    if metrics_aggregation_process is None:
      metrics_finalizers = model.metric_finalizers()
      local_unfinalized_metrics_type = type_conversions.type_from_tensors(
          model.report_local_unfinalized_metrics()
      )
      metrics_aggregation_process = aggregation_factory.SumThenFinalizeFactory(
          metrics_finalizers
      ).create(local_unfinalized_metrics_type)
    else:
      py_typecheck.check_type(
          metrics_aggregation_process,
          _AggregationProcess,
          'metrics_aggregation_process',
      )

  @federated_computation.federated_computation
  def init_fn():
    return metrics_aggregation_process.initialize()

  client_update_computation = federated_evaluation.build_local_evaluation(
      model_fn, model_weights_type, batch_type, use_experimental_simulation_loop
  )

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_clients(model_weights_type),
      computation_types.at_clients(computation_types.SequenceType(batch_type)),
  )
  def next_fn(state, model_weights, client_data):
    model_outputs = intrinsics.federated_map(
        client_update_computation, (model_weights, client_data)
    )
    metrics_output = metrics_aggregation_process.next(
        state, model_outputs.local_outputs
    )
    current_round_metrics, total_rounds_metrics = metrics_output.result
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=current_round_metrics,
                total_rounds_metrics=total_rounds_metrics,
            )
        )
    )
    # Return empty result as no model update will be performed for evaluation.
    empty_client_result = intrinsics.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        placements.CLIENTS,
    )
    return measured_process.MeasuredProcessOutput(
        metrics_output.state, empty_client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_functional_fed_eval_client_work(
    model: functional.FunctionalModel,
    metrics_aggregation_process: Optional[_AggregationProcess],
    model_weights_type: computation_types.StructType,
) -> client_works.ClientWorkProcess:
  """Builds a `ClientWorkProcess` that performs model evaluation at clients."""

  def ndarray_to_tensorspec(ndarray):
    return tf.TensorSpec(
        shape=ndarray.shape, dtype=tf.dtypes.as_dtype(ndarray.dtype)
    )

  # Wrap in a `ModelWeights` structure that is required by the `finalizer.`
  weights_type = model_weights_lib.ModelWeights(
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[0]),
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[1]),
  )
  tuple_weights_type = (weights_type.trainable, weights_type.non_trainable)
  batch_type = computation_types.to_type(model.input_spec)
  local_eval = federated_evaluation.build_functional_local_evaluation(
      model, tuple_weights_type, batch_type
  )

  if metrics_aggregation_process is None:
    unfinalized_metrics_type = local_eval.type_signature.result
    metrics_aggregation_process = aggregation_factory.SumThenFinalizeFactory(
        model.finalize_metrics
    ).create(unfinalized_metrics_type)

  @federated_computation.federated_computation
  def init_fn():
    return metrics_aggregation_process.initialize()

  @tensorflow_computation.tf_computation(
      model_weights_type, computation_types.SequenceType(batch_type)
  )
  def client_update_computation(model_weights, client_data):
    # Switch to the tuple expected by FunctionalModel.
    tuple_weights = (model_weights.trainable, model_weights.non_trainable)
    return local_eval(tuple_weights, client_data)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_clients(model_weights_type),
      computation_types.at_clients(computation_types.SequenceType(batch_type)),
  )
  def next_fn(state, model_weights, client_data):
    unfinalized_metrics = intrinsics.federated_map(
        client_update_computation, (model_weights, client_data)
    )
    metrics_output = metrics_aggregation_process.next(
        state, unfinalized_metrics
    )
    current_round_metrics, total_rounds_metrics = metrics_output.result
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=current_round_metrics,
                total_rounds_metrics=total_rounds_metrics,
            )
        )
    )
    # Return empty result as no model update will be performed for evaluation.
    empty_client_result = intrinsics.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        placements.CLIENTS,
    )
    return measured_process.MeasuredProcessOutput(
        metrics_output.state, empty_client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_identity_finalizer(
    model_weights_type: computation_types.StructType,
    update_type: computation_types.StructType,
) -> finalizers.FinalizerProcess:
  """Builds a `FinalizerProcess` that performs no update on model weights."""

  @federated_computation.federated_computation()
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  # The type signature of `next` function is defined so that the created
  # `tff.learning.templates.FinalizerProcess` can be used in
  # `tff.learning.templates.compose_learning_process`.
  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_server(model_weights_type),
      computation_types.at_server(update_type),
  )
  def next_fn(state, weights, update):
    del update
    empty_measurements = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(
        state, weights, empty_measurements
    )

  return finalizers.FinalizerProcess(init_fn, next_fn)


def build_fed_eval(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    model_distributor: Optional[distributors.DistributionProcess] = None,
    metrics_aggregation_process: Optional[
        aggregation_process.AggregationProcess
    ] = None,
    use_experimental_simulation_loop: bool = False,
) -> learning_process.LearningProcess:
  """Builds a learning process that performs federated evaluation.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  federated evaluation on clients. The learning process has the following
  methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with type signature `( -> S@SERVER)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `LearningAlgorithmState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. The output `L` contains the updated
      server state, as well as aggregated metrics at the server, including
      client evaluation metrics and any other metrics from distribution and
      aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next`, and `M` represents the type
      of the model weights used during evaluation.
  *   `set_model_weights`: A `tff.Computation` with type signature
      `(<S, M> -> S)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `M` represents the type of the model weights
      used during evaluation.

  Each time `next` is called, the server model is broadcast to each client using
  a distributor. Each client evaluates the model and reports local unfinalized
  metrics. The local unfinalized metrics are then aggregated and finalized at
  server using the metrics aggregator. Both current round and total rounds
  metrics will be produced. There are no update of the server model during the
  evaluation process.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`, or an
      instance of a `tff.learning.models.FunctionalModel`. When passing a
      callable, the callable must *not* capture TensorFlow tensors or variables
      and use them.  The model must be constructed entirely from scratch on each
      invocation, returning the same pre-constructed model each call will result
      in an error.
    model_distributor: An optional `tff.learning.templates.DistributionProcess`
      that broadcasts the model weights on the server to the clients. It must
      support the signature `(input_values@SERVER -> output_values@CLIENTS)` and
      have empty state. If None, the server model is broadcast to the clients
      using the default `tff.federated_broadcast`.
    metrics_aggregation_process: An optional `tff.templates.AggregationProcess`
      which aggregates the local unfinalized metrics at clients to server and
      finalizes the metrics at server. The `tff.templates.AggregationProcess`
      accumulates unfinalized metrics across round in the state, and produces a
      tuple of current round metrics and total rounds metrics in the result. If
      None, the `tff.templates.AggregationProcess` created by the
      `SumThenFinalizeFactory` with metric finalizers defined in the model is
      used.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tff.learning.templates.LearningProcess` performs federated evaluation on
    clients, and returns updated state and metrics.

  Raises:
    TypeError: If any argument type mismatches.
  """
  if not callable(model_fn):
    if not isinstance(model_fn, functional.FunctionalModel):
      raise TypeError(
          'If `model_fn` is not a callable, it must be an instance '
          f'tff.learning.models.FunctionalModel. Got {type(model_fn)}'
      )

    @tensorflow_computation.tf_computation()
    def initial_model_weights_fn():
      trainable_weights, non_trainable_weights = model_fn.initial_weights
      return model_weights_lib.ModelWeights(
          tuple(tf.convert_to_tensor(w) for w in trainable_weights),
          tuple(tf.convert_to_tensor(w) for w in non_trainable_weights),
      )

  else:

    @tensorflow_computation.tf_computation()
    def initial_model_weights_fn():
      return model_weights_lib.ModelWeights.from_model(
          model_fn()  # pytype: disable=not-callable
      )

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)
  else:
    py_typecheck.check_type(
        model_distributor, distributors.DistributionProcess, 'model_distributor'
    )

  if not callable(model_fn):
    client_work = _build_functional_fed_eval_client_work(
        model_fn, metrics_aggregation_process, model_weights_type
    )
  else:
    client_work = _build_fed_eval_client_work(
        model_fn,
        metrics_aggregation_process,
        model_weights_type,
        use_experimental_simulation_loop,
    )

  client_work_result_type = computation_types.at_clients(
      client_works.ClientResult(update=(), update_weight=())
  )
  model_update_type = client_work_result_type.member.update
  model_update_weight_type = client_work_result_type.member.update_weight
  model_aggregator_factory = mean.MeanFactory()
  model_aggregator = model_aggregator_factory.create(
      model_update_type, model_update_weight_type
  )

  # The finalizer performs no update on model weights.
  finalizer = _build_identity_finalizer(
      model_weights_type,
      model_aggregator.next.type_signature.result.result.member,
  )

  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      model_aggregator,
      finalizer,
  )
