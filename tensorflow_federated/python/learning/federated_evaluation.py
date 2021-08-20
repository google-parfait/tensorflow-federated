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
"""A simple implementation of federated evaluation."""

from typing import Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import evaluation
from tensorflow_federated.python.learning.framework import optimizer_utils


# Convenience aliases.
SequenceType = computation_types.SequenceType


def build_federated_evaluation(
    model_fn: Callable[[], model_lib.Model],
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
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
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the evaluation metrics
    as aggregated by `tff.learning.Model.federated_output_computation`.
  """
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO(b/124477628): Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_utils.weights_type_from_model(model)
    batch_type = computation_types.to_type(model.input_spec)

  if broadcast_process is not None:
    if not isinstance(broadcast_process, measured_process.MeasuredProcess):
      raise ValueError('`broadcast_process` must be a `MeasuredProcess`, got '
                       f'{type(broadcast_process)}.')
    if optimizer_utils.is_stateful_process(broadcast_process):
      raise ValueError(
          'Cannot create a federated evaluation with a stateful '
          'broadcast process, must be stateless, has state: '
          f'{broadcast_process.initialize.type_signature.result!r}')

    @computations.federated_computation(
        computation_types.at_server(model_weights_type))
    def stateless_distributor(server_model_weights):
      return broadcast_process.next(broadcast_process.initialize(),
                                    server_model_weights).result

  else:

    @computations.federated_computation(
        computation_types.at_server(model_weights_type))
    def stateless_distributor(server_model_weights):
      return intrinsics.federated_broadcast(server_model_weights)

  client_eval_work = evaluation.build_eval_work(
      model_fn,
      model_weights_type,
      batch_type,
      use_experimental_simulation_loop=use_experimental_simulation_loop)

  metrics_type = client_eval_work.type_signature.result
  stateless_aggregator = evaluation.build_model_metrics_aggregator(
      model, metrics_type)
  return evaluation.compose_eval_computation(stateless_distributor,
                                             client_eval_work,
                                             stateless_aggregator)
