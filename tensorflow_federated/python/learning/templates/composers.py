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
"""Methods for composing learning components into a LearningProcess."""

import collections
from collections.abc import Callable
from typing import Any, NamedTuple

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work
from tensorflow_federated.python.learning.templates import type_checks


# TODO: b/190334722 - Add SLO guarantees / backwards compatibility guarantees.
class LearningAlgorithmState(NamedTuple):
  """A structure representing the state of a learning process.

  Attributes:
    global_model_weights: A structure representing the model weights of the
      global model being trained.
    distributor: State of the distributor component.
    client_work: State of the client work component.
    aggregator: State of the aggregator component.
    finalizer: State of the finalizer component.
  """

  global_model_weights: Any
  distributor: Any
  client_work: Any
  aggregator: Any
  finalizer: Any


# pyformat: disable
def compose_learning_process(
    initial_model_weights_fn: computation_base.Computation,
    model_weights_distributor: distributors.DistributionProcess,
    client_work: client_works.ClientWorkProcess,
    model_update_aggregator: aggregation_process.AggregationProcess,
    model_finalizer: finalizers.FinalizerProcess
) -> learning_process.LearningProcess:
  """Composes specialized measured processes into a learning process.

  Given 4 specialized measured processes (described below) that make a learning
  process, and a computation that returns initial model weights to be used for
  training, this method validates that the processes fit together, and returns a
  `LearningProcess`. Please see the tutorial at
  https://www.tensorflow.org/federated/tutorials/composing_learning_algorithms
  for more details on composing learning processes.

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

  The `get_hparams` computation of the created learning process produces a
  nested ordered dictionary containing the result of `client_work.get_hparams`
  and `finalizer.get_hparams`. The `set_hparams` computation operates similarly,
  by delegating to `client_work.set_hparams` and `finalizer.set_hparams` to set
  the hyperparameters in their associated states.

  Args:
    initial_model_weights_fn: A `tff.Computation` that returns (unplaced)
      initial model weights.
    model_weights_distributor: A `tff.learning.templates.DistributionProcess`.
    client_work: A `tff.learning.templates.ClientWorkProcess`.
    model_update_aggregator: A `tff.templates.AggregationProcess`.
    model_finalizer: A `tff.learning.templates.FinalizerProcess`.

  Returns:
    A `tff.learning.templates.LearningProcess`.

  Raises:
    ClientSequenceTypeError: If the first arg of the `next` method of the
    resulting `LearningProcess` is not a structure of sequences placed at
    `tff.CLIENTS`.
  """
  # pyformat: enable
  _validate_args(initial_model_weights_fn, model_weights_distributor,
                 client_work, model_update_aggregator, model_finalizer)
  client_data_type = client_work.next.type_signature.parameter[2]  # pytype: disable=unsupported-operands

  @federated_computation.federated_computation()
  def init_fn():
    initial_model_weights = intrinsics.federated_eval(initial_model_weights_fn,
                                                      placements.SERVER)
    return intrinsics.federated_zip(
        LearningAlgorithmState(initial_model_weights,
                               model_weights_distributor.initialize(),
                               client_work.initialize(),
                               model_update_aggregator.initialize(),
                               model_finalizer.initialize()))

  @federated_computation.federated_computation(init_fn.type_signature.result,
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

  state_parameter_type = next_fn.type_signature.parameter[0].member

  @tensorflow_computation.tf_computation(state_parameter_type)
  def get_model_weights_fn(state):
    return state.global_model_weights

  @tensorflow_computation.tf_computation(
      state_parameter_type, state_parameter_type.global_model_weights)
  def set_model_weights_fn(state, model_weights):
    return LearningAlgorithmState(
        global_model_weights=model_weights,
        distributor=state.distributor,
        client_work=state.client_work,
        aggregator=state.aggregator,
        finalizer=state.finalizer)

  @tensorflow_computation.tf_computation(state_parameter_type)
  def get_hparams_fn(state):
    client_work_hparams = client_work.get_hparams(state.client_work)
    finalizer_hparams = model_finalizer.get_hparams(state.finalizer)
    return collections.OrderedDict(
        client_work=client_work_hparams, finalizer=finalizer_hparams)

  hparams_type = get_hparams_fn.type_signature.result

  @tensorflow_computation.tf_computation(state_parameter_type, hparams_type)
  def set_hparams_fn(state, hparams):
    updated_client_work_state = client_work.set_hparams(state.client_work,
                                                        hparams['client_work'])
    updated_finalizer_state = model_finalizer.set_hparams(
        state.finalizer, hparams['finalizer'])
    return LearningAlgorithmState(
        global_model_weights=state.global_model_weights,
        distributor=state.distributor,
        client_work=updated_client_work_state,
        aggregator=state.aggregator,
        finalizer=updated_finalizer_state)

  composed_learning_process = learning_process.LearningProcess(
      init_fn,
      next_fn,
      get_model_weights_fn,
      set_model_weights_fn,
      get_hparams_fn=get_hparams_fn,
      set_hparams_fn=set_hparams_fn)

  # LearningProcess.__init__ does some basic type checking. Here we do more
  # specific type checking to validate that the second arg of `next` is a
  # CLIENTS-placed structure of sequences.
  next_fn = composed_learning_process.next
  next_fn_param = next_fn.type_signature.parameter
  try:
    type_checks.check_is_client_placed_structure_of_sequences(
        next_fn_param[1],  # pytype: disable=unsupported-operands
    )
  except type_checks.ClientSequenceTypeError as type_error:
    raise TypeError(
        'The learning process composition produced a `next` function with type '
        f'signature {next_fn.type_signature}. However, the second arg of `next`'
        ' must be a CLIENTS-placed structure of sequences.') from type_error
  return composed_learning_process


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
  if isinstance(global_model_weights_type, computation_types.FederatedType):
    raise TypeError(
        f'Provided initial_model_weights_fn must be a tff.Computation with '
        f'unplaced return type.\n'
        f'Return type found: {global_model_weights_type}')
  global_model_weights_type = computation_types.FederatedType(
      global_model_weights_type, placements.SERVER)
  py_typecheck.check_type(model_weights_distributor,
                          distributors.DistributionProcess)
  py_typecheck.check_type(client_work, client_works.ClientWorkProcess)
  py_typecheck.check_type(model_update_aggregator,
                          aggregation_process.AggregationProcess)
  if not model_update_aggregator.is_weighted:
    raise TypeError('Provided model_update_aggregator must be weighted.')
  py_typecheck.check_type(model_finalizer, finalizers.FinalizerProcess)

  # TODO: b/190334722 - Consider adding custom error messages.
  distributor_param = model_weights_distributor.next.type_signature.parameter
  distributor_result = model_weights_distributor.next.type_signature.result
  client_work_param = client_work.next.type_signature.parameter
  client_work_result = client_work.next.type_signature.result
  aggregator_param = model_update_aggregator.next.type_signature.parameter
  aggregator_result = model_update_aggregator.next.type_signature.result
  finalizer_param = model_finalizer.next.type_signature.parameter
  finalizer_result = model_finalizer.next.type_signature.result

  distributor_param[1].check_assignable_from(global_model_weights_type)  # pytype: disable=unsupported-operands
  client_work_param[1].check_assignable_from(
      distributor_result.result,  # pytype: disable=attribute-error
  )  # pytype: disable=unsupported-operands
  aggregator_param[1].member.check_assignable_from(
      client_work_result.result.member.update,  # pytype: disable=attribute-error
  )  # pytype: disable=unsupported-operands
  aggregator_param[2].member.check_assignable_from(
      client_work_result.result.member.update_weight,  # pytype: disable=attribute-error
  )  # pytype: disable=unsupported-operands
  finalizer_param[1].check_assignable_from(global_model_weights_type)  # pytype: disable=unsupported-operands
  finalizer_param[2].check_assignable_from(
      aggregator_result.result,  # pytype: disable=attribute-error
  )  # pytype: disable=unsupported-operands
  global_model_weights_type.check_assignable_from(
      finalizer_result.result,   # pytype: disable=attribute-error
  )


def build_basic_fedavg_process(model_fn: Callable[[], variable.VariableModel],
                               client_learning_rate: float,
                               server_learning_rate: float = 1.0):
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
    model_fn: A no-arg function that returns a `tff.learning.models.VariableModel`.
    client_learning_rate: A float. Learning rate for the SGD at clients.
    server_learning_rate: A float representing the learning rate for the SGD
      step occuring at the server. Defaults to 1.0.

  Returns:
    A `LearningProcess`.
  """
  py_typecheck.check_type(client_learning_rate, float)

  @tensorflow_computation.tf_computation()
  def initial_model_weights_fn():
    return model_weights_lib.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  distributor = distributors.build_broadcast_process(model_weights_type)
  client_work = model_delta_client_work.build_model_delta_client_work(
      model_fn,
      sgdm.build_sgdm(client_learning_rate),
      client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
  aggregator = mean.MeanFactory().create(
      client_work.next.type_signature.result.result.member.update,  # pytype: disable=attribute-error
      client_work.next.type_signature.result.result.member.update_weight,  # pytype: disable=attribute-error
  )
  finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
      sgdm.build_sgdm(server_learning_rate), model_weights_type)

  return compose_learning_process(initial_model_weights_fn, distributor,
                                  client_work, aggregator, finalizer)
