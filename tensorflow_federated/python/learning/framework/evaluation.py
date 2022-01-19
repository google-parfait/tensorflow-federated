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
"""An implementation of federated evaluation using re-usable layers."""

import collections
from typing import Any, Callable, OrderedDict

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.metrics import aggregator


def build_eval_work(
    model_fn: Callable[[], model_lib.Model],
    model_weights_type: computation_types.Type,
    data_type: computation_types.Type,
    use_experimental_simulation_loop: bool = False
) -> computation_impl.ConcreteComputation:
  """Builds a `tff.Computation` for evaluating a model on a dataset.

  This function accepts model weights matching `model_weights_type` and data
  whose batch type matches `data_type`, and returns metrics computed at
  the corresponding model over the data.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    model_weights_type: A `tff.Type` representing the type of the model weights.
    data_type: A `tff.Type` representing the batch type of the data. This must
      be compatible with the batch type expected by the forward pass of the
      model returned by `model_fn`.
    use_experimental_simulation_loop: A boolean controlling the reduce loop used
      to iterate over client datasets. If set to `True`, an experimental reduce
      loop is used.

  Returns:
    A `tff.Computation`.
  """

  @computations.tf_computation(model_weights_type,
                               computation_types.SequenceType(data_type))
  @tf.function
  def client_eval(incoming_model_weights, dataset):
    with tf.init_scope():
      model = model_fn()
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          incoming_model_weights)

    def reduce_fn(num_examples, batch):
      model_output = model.forward_pass(batch, training=False)
      return num_examples + tf.cast(model_output.num_examples, tf.int64)

    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    num_examples = dataset_reduce_fn(
        reduce_fn=reduce_fn,
        dataset=dataset,
        initial_state_fn=lambda: tf.zeros([], dtype=tf.int64))
    return collections.OrderedDict(
        local_outputs=model.report_local_unfinalized_metrics(),
        num_examples=num_examples)

  return client_eval


def build_model_metrics_aggregator(
    model: model_lib.Model,
    metrics_type: computation_types.Type,
    metrics_aggregator: Callable[[
        OrderedDict[str, Callable[[Any],
                                  Any]], computation_types.StructWithPythonType
    ], computation_base.Computation] = aggregator.sum_then_finalize
) -> computation_impl.ConcreteComputation:
  """Creates a stateless aggregator for client metrics."""
  unfinalized_metrics_type = type_conversions.type_from_tensors(
      model.report_local_unfinalized_metrics())
  metrics_aggregation_computation = metrics_aggregator(
      model.metric_finalizers(), unfinalized_metrics_type)

  @computations.federated_computation(
      computation_types.at_clients(metrics_type))
  def aggregate_metrics(client_metrics):
    model_metrics = metrics_aggregation_computation(
        client_metrics.local_outputs)
    statistics = collections.OrderedDict(
        num_examples=intrinsics.federated_sum(client_metrics.num_examples))
    return intrinsics.federated_zip(
        collections.OrderedDict(eval=model_metrics, stat=statistics))

  return aggregate_metrics


class FederatedEvalTypeError(TypeError):
  """Raises evaluation components do not have expected type signatures."""
  pass


class FederatedEvalInputOutputError(TypeError):
  """Raises when evaluation components have mismatched input/outputs."""


def check_federated_type_with_correct_placement(value_type, placement):
  """Checks that a `tff.Type` has the desired federated placement."""
  if value_type is None:
    return False
  elif value_type.is_federated() and value_type.placement == placement:
    return True
  return False


def _validate_eval_types(
    stateless_distributor: computation_impl.ConcreteComputation,
    client_eval_work: computation_impl.ConcreteComputation,
    stateless_aggregator: computation_impl.ConcreteComputation):
  """Checks `compose_eval_computation` arguments meet documented constraints."""
  py_typecheck.check_type(stateless_distributor,
                          computation_impl.ConcreteComputation)
  py_typecheck.check_type(client_eval_work,
                          computation_impl.ConcreteComputation)
  py_typecheck.check_type(stateless_aggregator,
                          computation_impl.ConcreteComputation)

  distributor_type = stateless_distributor.type_signature
  distributor_parameter = distributor_type.parameter
  if not check_federated_type_with_correct_placement(distributor_parameter,
                                                     placements.SERVER):
    raise FederatedEvalTypeError(
        f'The distributor must have a single input placed at `SERVER`, found '
        f'input type signature {distributor_parameter}.')
  distributor_result = distributor_type.result
  if not check_federated_type_with_correct_placement(distributor_result,
                                                     placements.CLIENTS):
    raise FederatedEvalTypeError(
        f'The distributor must have a single output placed at `CLIENTS`, found '
        f'output type signature {distributor_result}.')

  client_work_type = client_eval_work.type_signature
  if client_work_type.parameter.is_federated(
  ) or client_work_type.result.is_federated():
    raise FederatedEvalTypeError(
        f'The `client_eval_work` must be not be a federated computation, but '
        f'found type signature {client_work_type}.')

  aggregator_type = stateless_aggregator.type_signature
  aggregator_parameter = aggregator_type.parameter
  if not check_federated_type_with_correct_placement(aggregator_parameter,
                                                     placements.CLIENTS):
    raise FederatedEvalTypeError(
        f'The aggregator must have a single input placed at `CLIENTS`, found '
        f'type signature {aggregator_parameter}.')
  aggregator_result = aggregator_type.result
  if not check_federated_type_with_correct_placement(aggregator_result,
                                                     placements.SERVER):
    raise FederatedEvalTypeError(
        f'The aggregator must have a single output placed at `SERVER`, found '
        f'type signature {aggregator_result}.')

  if not client_work_type.parameter[0].is_assignable_from(
      distributor_result.member):
    raise FederatedEvalInputOutputError(
        f'The output of the distributor must be assignable to the first input '
        f'of the client work. Found distributor result of type '
        f'{distributor_result.member}, but client work argument of type'
        f'{client_work_type.parameter[0]}.')

  if not aggregator_parameter.member.is_assignable_from(
      client_work_type.result):
    raise FederatedEvalInputOutputError(
        f'The output of the client work must be assignable to the input of the '
        f'aggregator. Found client work output of type '
        f'{client_work_type.result}, but aggregator parameter of type '
        f'{aggregator_parameter.member}.')


def compose_eval_computation(
    stateless_distributor: computation_impl.ConcreteComputation,
    client_eval_work: computation_impl.ConcreteComputation,
    stateless_metrics_aggregator: computation_impl.ConcreteComputation):
  """Builds a TFF computation performing stateless evaluation across clients.

  The resulting computation has type signature
  `(S@SERVER, T@CLIENTS -> A@SERVER)`, where `S` represents some value of the
  server, `T` represents data held by the clients, and `A` are the aggregate
  metrics across all clients.

  Args:
    stateless_distributor: A `tff.Computation` that broadcasts a value placed at
      `tff.SERVER` to the clients. It must have type signature `(S@SERVER ->
      R@CLIENTS)`, where the member of `R` matches the first type expected by
      `client_eval_work`.
    client_eval_work: A `tff.Computation` used to compute client metrics. Must
      have type signature (<R, T> -> M) of unplaced types, where `R` is a type
      representing a reference value broadcast from the server (such as model
      weights), `T` represents values held by the client, and `M` is the type of
      the metrics computed by the client.
    stateless_metrics_aggregator: A `tff.Computation` that aggregates metrics
      from the client. It must have type signature `(M@CLIENTS -> A@SERVER)`
      where `M` has member matching the output of `client_eval_work`, and `A`
      represents the aggregated metrics.

  Returns:
    A `tff.Computation`.
  """
  _validate_eval_types(stateless_distributor, client_eval_work,
                       stateless_metrics_aggregator)

  distributor_input_type = stateless_distributor.type_signature.parameter
  client_data_type = client_eval_work.type_signature.parameter[1]

  @computations.federated_computation(
      distributor_input_type, computation_types.at_clients(client_data_type))
  def evaluate(server_value, client_data):
    distributor_output = stateless_distributor(server_value)
    client_metrics = intrinsics.federated_map(client_eval_work,
                                              (distributor_output, client_data))
    return stateless_metrics_aggregator(client_metrics)

  return evaluate
