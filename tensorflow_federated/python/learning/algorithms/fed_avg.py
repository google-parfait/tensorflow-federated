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
"""An implementation of the Federated Averaging algorithm.

The original Federated Averaging algorithm is proposed by the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

This file implements a generalized version of the Federated Averaging algorithm:

Adaptive Federated Optimization
    Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
    Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan. ICLR 2021.
    https://arxiv.org/abs/2003.00295
"""

from typing import Callable, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.algorithms import aggregation
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)


def build_weighted_fed_avg(
    model_fn: Callable[[], model_lib.Model],
    client_optimizer_fn: Union[optimizer_base.Optimizer,
                               Callable[[], tf.keras.optimizers.Optimizer]],
    server_optimizer_fn: Union[optimizer_base.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    client_weighting: Optional[
        client_weight_lib
        .ClientWeighting] = client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    use_experimental_simulation_loop: bool = False
) -> learning_process.LearningProcess:
  """Builds a learning process that performs federated averaging.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  federated averaging on client models. The iterative process has the following
  methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` representing the initial
      state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as metrics that
      are the result of `tff.learning.Model.federated_output_computation` during
      client training, and any other metrics from broadcast and aggregation
      processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matchs the output of `initialize` and `next` and `M` represents the type
      of the model weights used during training.

  Each time the `next` method is called, the server model is communicated to
  each client using the provided `model_distributor`. For each client, local
  training is performed using `client_optimizer_fn`. Each client computes the
  difference between the client model after training and its initial model.
  These model deltas are then aggregated at the server using a weighted
  aggregation function, according to `client_weighting`. The aggregate model
  delta is applied at the server using a server optimizer.

  Note: the default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 1.0, which corresponds to adding the model delta to
  the current server model. This recovers the original FedAvg algorithm in
  [McMahan et al., 2017](https://arxiv.org/abs/1602.05629). More
  sophisticated federated averaging procedures may use different learning rates
  or server optimizers (this generalized FedAvg algorithm is described in
  [Reddi et al., 2021](https://arxiv.org/abs/2003.00295)).

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. By default, this uses
      `tf.keras.optimizers.SGD` with a learning rate of 1.0.
    client_weighting: A member `tff.learning.ClientWeighting` that specifies a
      built-in weighting method. By default, weighting by number of examples is
      used.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)

  @computations.tf_computation()
  def initial_model_weights_fn():
    return model_utils.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)

  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)
  aggregator = model_aggregator.create(
      model_weights_type.trainable, computation_types.TensorType(tf.float32))
  process_signature = aggregator.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError('`model_aggregator` does not produce a compatible '
                    '`AggregationProcess`. The processes must retain the type '
                    'structure of the inputs on the server, but got '
                    f'{input_client_value_type.member} != '
                    f'{result_server_value_type.member}.')

  client_work = model_delta_client_work.build_model_delta_client_work(
      model_fn=model_fn,
      optimizer=client_optimizer_fn,
      client_weighting=client_weighting,
      use_experimental_simulation_loop=use_experimental_simulation_loop)
  finalizer = finalizers.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type)
  return composers.compose_learning_process(initial_model_weights_fn,
                                            model_distributor, client_work,
                                            aggregator, finalizer)


def build_unweighted_fed_avg(
    model_fn: Callable[[], model_lib.Model],
    client_optimizer_fn: Union[optimizer_base.Optimizer,
                               Callable[[], tf.keras.optimizers.Optimizer]],
    server_optimizer_fn: Union[optimizer_base.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.UnweightedAggregationFactory] = None,
    use_experimental_simulation_loop: bool = False
) -> learning_process.LearningProcess:
  """Builds a learning process that performs federated averaging.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  federated averaging on client models. The iterative process has the following
  methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` representing the initial
      state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as metrics that
      are the result of `tff.learning.Model.federated_output_computation` during
      client training, and any other metrics from broadcast and aggregation
      processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matchs the output of `initialize` and `next` and `M` represents the type
      of the model weights used during training.

  Each time the `next` method is called, the server model is communicated to
  each client using the provided `model_distributor`. For each client, local
  training is performed using `client_optimizer_fn`. Each client computes the
  difference between the client model after training and its initial model.
  These model deltas are then aggregated at the server using an unweighted
  aggregation function. The aggregate model delta is applied at the server using
  a server optimizer.

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
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. By default, this uses
      `tf.keras.optimizers.SGD` with a learning rate of 1.0.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.UnweightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.UnweightedMeanFactory`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  if model_aggregator is None:
    model_aggregator = mean.UnweightedMeanFactory()
  py_typecheck.check_type(model_aggregator,
                          factory.UnweightedAggregationFactory)

  return build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weight_lib.ClientWeighting.UNIFORM,
      model_distributor=model_distributor,
      model_aggregator=aggregation.as_weighted_aggregator(model_aggregator),
      use_experimental_simulation_loop=use_experimental_simulation_loop)
