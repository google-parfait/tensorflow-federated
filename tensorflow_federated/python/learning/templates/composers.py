# Copyright 2021, The TensorFlow Federated Authors.
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
"""Methods for composing learning components into a LearningProcess."""

import collections
from typing import Callable

import attr

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work


# TODO(b/190334722): Add SLO guarantees / backwards compatibility guarantees.
@attr.s(eq=False, frozen=True)
class LearningAlgorithmState:
  """A structure representing the state of a learning process.

  Attributes:
    global_model_weights: A structure representing the model weights of the
      global model being trained.
    distributor: State of the distributor component.
    client_work: State of the client work component.
    aggregator: State of the aggregator component.
    finalizer: State of the finalizer component.
  """
  global_model_weights = attr.ib()
  distributor = attr.ib()
  client_work = attr.ib()
  aggregator = attr.ib()
  finalizer = attr.ib()


# pyformat: disable
# TODO(b/190334722): Add a visualization of how the 4 components fit together,
# and/or a pointer to a more detailed public tutiorial.
def compose_learning_process(
    initial_model_weights_fn: computation_base.Computation,
    model_weights_distributor: distributors.DistributionProcess,
    client_work: client_works.ClientWorkProcess,
    model_update_aggregator: aggregation_process.AggregationProcess,
    model_finalizer: finalizers.FinalizerProcess
) -> learning_process.LearningProcess:
  """Composes specialized measured processes into a learning process.

  Given the 4 specialized measured processes that make a learning process as
  documented in [TODO(b/190334722)], and a computation that returns initial
  model weights to be used for training, this method validates that the
  processes fit together, and returns a `LearningProcess`.

  The main purpose of the 4 measured processes are:
    * `model_weights_distributor`: Make global model weights at server available
      as the starting point for learning work to be done at clients.
    * `client_work`: Produce an update to the model received at clients.
    * `model_update_aggregator`: Aggregates the model updates from clients to
      the server.
    * `model_finalizer`: Updates the global model weights using the aggregated
      model update at server.

  The `next` computation of the created learning process is composed from the
  `next` computations of the 4 measured processes, in order as visualized below.
  The type signatures of the processes must be such that this chaining is
  possible. Each process also reports its own metrics.

  ```
  ┌─────────────────────────┐
  │model_weights_distributor│
  └△─┬─┬────────────────────┘
   │ │┌▽──────────┐
   │ ││client_work│
   │ │└┬─────┬────┘
   │┌▽─▽────┐│
   ││metrics││
   │└△─△────┘│
   │ │┌┴─────▽────────────────┐
   │ ││model_update_aggregator│
   │ │└┬──────────────────────┘
  ┌┴─┴─▽──────────┐
  │model_finalizer│
  └┬──────────────┘
  ┌▽─────┐
  │result│
  └──────┘
  ```

  Args:
    initial_model_weights_fn: A `tff.Computation` that returns (unplaced)
      initial model weights.
    model_weights_distributor: A `DistributionProcess`.
    client_work: A `ClientWorkProcess`.
    model_update_aggregator: A `tff.templates.AggregationProcess`.
    model_finalizer: A `FinalizerProcess`.

  Returns:
    A `LearningProcess`.
  """
  # pyformat: enable
  _validate_args(initial_model_weights_fn, model_weights_distributor,
                 client_work, model_update_aggregator, model_finalizer)
  client_data_type = client_work.next.type_signature.parameter[2]

  @computations.federated_computation()
  def init_fn():
    initial_model_weights = intrinsics.federated_eval(initial_model_weights_fn,
                                                      placements.SERVER)
    return intrinsics.federated_zip(
        LearningAlgorithmState(initial_model_weights,
                               model_weights_distributor.initialize(),
                               client_work.initialize(),
                               model_update_aggregator.initialize(),
                               model_finalizer.initialize()))

  @computations.federated_computation(init_fn.type_signature.result,
                                      client_data_type)
  def next_fn(state, client_data):
    # Compose processes.
    distributor_output = model_weights_distributor.next(
        state.distributor, state.global_model_weights)
    client_work_output = client_work.next(state.client_work,
                                          distributor_output.result,
                                          client_data)
    aggregator_output = model_update_aggregator.next(
        state.aggregator, client_work_output.result.update,
        client_work_output.result.update_weight)
    finalizer_output = model_finalizer.next(state.finalizer,
                                            state.global_model_weights,
                                            aggregator_output.result)

    # Form the learning process output.
    new_global_model_weights = finalizer_output.result
    new_state = intrinsics.federated_zip(
        LearningAlgorithmState(new_global_model_weights,
                               distributor_output.state,
                               client_work_output.state,
                               aggregator_output.state, finalizer_output.state))
    metrics = intrinsics.federated_zip(
        collections.OrderedDict(
            distributor=distributor_output.measurements,
            client_work=client_work_output.measurements,
            aggregator=aggregator_output.measurements,
            finalizer=finalizer_output.measurements))

    return learning_process.LearningProcessOutput(new_state, metrics)

  @computations.tf_computation(next_fn.type_signature.result.state.member)
  def model_weights_fn(state):
    return state.global_model_weights

  return learning_process.LearningProcess(init_fn, next_fn, model_weights_fn)


def _validate_args(initial_model_weights_fn, model_weights_distributor,
                   client_work, model_update_aggregator, model_finalizer):
  """Checks `compose_learning_process` args meet the documented constraints."""
  py_typecheck.check_type(initial_model_weights_fn,
                          computation_base.Computation)
  if initial_model_weights_fn.type_signature.parameter is not None:
    raise TypeError(
        f'Provided initial_model_weights_fn must be a no-arg tff.Computation.\n'
        f'Found input parameter: '
        f'{initial_model_weights_fn.type_signature.parameter}')
  global_model_weights_type = initial_model_weights_fn.type_signature.result
  if global_model_weights_type.is_federated():
    raise TypeError(
        f'Provided initial_model_weights_fn must be a tff.Computation with '
        f'unplaced return type.\n'
        f'Return type found: {global_model_weights_type}')
  global_model_weights_type = computation_types.at_server(
      global_model_weights_type)
  py_typecheck.check_type(model_weights_distributor,
                          distributors.DistributionProcess)
  py_typecheck.check_type(client_work, client_works.ClientWorkProcess)
  py_typecheck.check_type(model_update_aggregator,
                          aggregation_process.AggregationProcess)
  if not model_update_aggregator.is_weighted:
    raise TypeError('Provided model_update_aggregator must be weighted.')
  py_typecheck.check_type(model_finalizer, finalizers.FinalizerProcess)

  # TODO(b/190334722): Consider adding custom error messages.
  distributor_param = model_weights_distributor.next.type_signature.parameter
  distributor_result = model_weights_distributor.next.type_signature.result
  client_work_param = client_work.next.type_signature.parameter
  client_work_result = client_work.next.type_signature.result
  aggregator_param = model_update_aggregator.next.type_signature.parameter
  aggregator_result = model_update_aggregator.next.type_signature.result
  finalizer_param = model_finalizer.next.type_signature.parameter
  finalizer_result = model_finalizer.next.type_signature.result

  distributor_param[1].check_assignable_from(global_model_weights_type)
  client_work_param[1].check_assignable_from(distributor_result.result)
  aggregator_param[1].member.check_assignable_from(
      client_work_result.result.member.update)
  aggregator_param[2].member.check_assignable_from(
      client_work_result.result.member.update_weight)
  finalizer_param[1].check_assignable_from(global_model_weights_type)
  finalizer_param[2].check_assignable_from(aggregator_result.result)
  global_model_weights_type.check_assignable_from(finalizer_result.result)


def build_basic_fedavg_process(model_fn: Callable[[], model_lib.Model],
                               client_learning_rate: float):
  """Builds vanilla Federated Averaging process.

  The created process is the basic form of the Federated Averaging algorithm as
  proposed by http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf in
  Algorithm 1, for training the model created by `model_fn`. The following is
  the algorithm in pseudo-code:

  ```
  # Inputs: m: Initial model weights; eta: Client learning rate
  for i in num_rounds:
    for c in available_clients_indices:
      delta_m_c, w_c = client_update(m, eta)
    aggregate_model_delta = sum_c(model_delta_c * w_c) / sum_c(w_c)
    m = m - aggregate_model_delta
  return m  # Final trained model.

  def client_udpate(m, eta):
    initial_m = m
    for batch in client_dataset:
      m = m - eta * grad(m, b)
    delta_m = initial_m - m
    return delta_m, size(dataset)
  ```

  The other algorithm hyper parameters (batch size, number of local epochs) are
  controlled by the data provided to the built process.

  An example usage of the returned `LearningProcess` in simulation:

  ```
  fedavg = build_basic_fedavg_process(model_fn, 0.1)

  # Create a `LearningAlgorithmState` containing the initial model weights for
  # the model returned from `model_fn`.
  state = fedavg.initialize()
  for _ in range(num_rounds):
    client_data = ...  # Preprocessed client datasets
    output = fedavg.next(state, client_data)
    write_round_metrics(outpus.metrics)
    # The new state contains the updated model weights after this round.
    state = output.state
  ```

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_learning_rate: A float. Learning rate for the SGD at clients.

  Returns:
    A `LearningProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_type(client_learning_rate, float)

  @computations.tf_computation()
  def initial_model_weights_fn():
    return model_utils.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  distributor = distributors.build_broadcast_process(model_weights_type)
  client_work = model_delta_client_work.build_model_delta_client_work(
      model_fn,
      sgdm.build_sgdm(client_learning_rate),
      client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
  aggregator = mean.MeanFactory().create(
      client_work.next.type_signature.result.result.member.update,
      client_work.next.type_signature.result.result.member.update_weight)
  finalizer = finalizers.build_apply_optimizer_finalizer(
      sgdm.build_sgdm(1.0), model_weights_type)

  return compose_learning_process(initial_model_weights_fn, distributor,
                                  client_work, aggregator, finalizer)
