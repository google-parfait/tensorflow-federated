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

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning.metrics import aggregator as metric_aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work

DEFAULT_SERVER_OPTIMIZER_FN = lambda: sgdm.build_sgdm(learning_rate=1.0)


def build_weighted_fed_avg(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    client_optimizer_fn: optimizer_base.Optimizer,
    server_optimizer_fn: Optional[optimizer_base.Optimizer] = None,
    *,
    client_weighting: Optional[
        client_weight_lib.ClientWeighting
    ] = client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
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

  Each time the `next` method is called, the server model is communicated to
  each client using the provided `model_distributor`. For each client, local
  training is performed using `client_optimizer_fn`. Each client computes the
  difference between the client model after training and its initial model.
  These model deltas are then aggregated at the server using a weighted
  aggregation function, according to `client_weighting`. The aggregate model
  delta is applied at the server using a server optimizer.

  Note: the default server optimizer function is
  `tff.learning.optimizers.build_sgdm` with a learning rate of 1.0, which
  corresponds to adding the model delta to the current server model. This
  recovers the original FedAvg algorithm in [McMahan et al.,
  2017](https://arxiv.org/abs/1602.05629). More sophisticated federated
  averaging procedures may use different learning rates or server optimizers
  (this generalized FedAvg algorithm is described in [Reddi et al.,
  2021](https://arxiv.org/abs/2003.00295)).

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`. By default, this
      uses `tff.leanring.optimizers.build_sgdm` with a learning rate of 1.0.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method. By default, weighting by number of examples
      is used.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via
      `tff.learning.templates.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
      If `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tff.learning.templates.LearningProcess`.

  Raises:
    TypeError: If arguments are not the documented types.
  """
  py_typecheck.check_type(
      client_weighting,
      client_weight_lib.ClientWeighting,
      label='client_weighting',
  )
  if server_optimizer_fn is None:
    server_optimizer_fn = DEFAULT_SERVER_OPTIMIZER_FN()

  if not isinstance(model_fn, Callable):
    if not isinstance(model_fn, functional.FunctionalModel):
      raise TypeError(
          'If `model_fn` is not a callable, it must be an instance of '
          f'`tff.learning.models.FunctionalModel`. Got {type(model_fn)}'
      )
    if not isinstance(client_optimizer_fn, optimizer_base.Optimizer):
      raise TypeError(
          'If `model_fn` is a `tff.learning.models.FunctionalModel`, '
          '`client_optimizer_fn` must be an instance of '
          '`tff.learning.optimizers.Optimizer`. '
          f'Got {type(client_optimizer_fn)}'
      )
    if not isinstance(server_optimizer_fn, optimizer_base.Optimizer):
      raise TypeError(
          'If `model_fn` is a `tff.learning.models.FunctionalModel`, '
          '`server_optimizer_fn` must be an instance of '
          '`tff.learning.optimizers.Optimizer`. '
          f'Got {type(server_optimizer_fn)}'
      )

    @tensorflow_computation.tf_computation()
    def initial_model_weights_fn():
      trainable_weights, non_trainable_weights = model_fn.initial_weights
      return model_weights.ModelWeights(
          tuple(tf.convert_to_tensor(w) for w in trainable_weights),
          tuple(tf.convert_to_tensor(w) for w in non_trainable_weights),
      )

  else:

    @tensorflow_computation.tf_computation()
    def initial_model_weights_fn():
      model = model_fn()  # pytype: disable=not-callable
      if not isinstance(model, variable.VariableModel):
        raise TypeError(
            'When `model_fn` is a callable, it return instances of'
            ' tff.learning.models.VariableModel. Instead callable returned'
            f' type: {type(model)}'
        )
      return model_weights.ModelWeights.from_model(model)

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)
  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)

  if not callable(model_fn):
    trainable_weights_type, _ = model_weights_type
    model_update_type = trainable_weights_type
  else:
    model_update_type = model_weights_type.trainable
  aggregator = model_aggregator.create(
      model_update_type, computation_types.TensorType(np.float32)
  )

  process_signature = aggregator.next.type_signature
  input_client_value_type = process_signature.parameter[1]  # pytype: disable=unsupported-operands
  result_server_value_type = process_signature.result[1]  # pytype: disable=unsupported-operands
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError(
        '`model_aggregator` does not produce a compatible '
        '`AggregationProcess`. The processes must retain the type '
        'structure of the inputs on the server, but got '
        f'{input_client_value_type.member} != '
        f'{result_server_value_type.member}.'
    )

  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize

  if not callable(model_fn):
    client_work = (
        model_delta_client_work.build_functional_model_delta_client_work(
            model=model_fn,
            optimizer=client_optimizer_fn,
            client_weighting=client_weighting,
            metrics_aggregator=metrics_aggregator,
            loop_implementation=loop_implementation,
        )
    )
  else:
    client_work = model_delta_client_work.build_model_delta_client_work(
        model_fn=model_fn,
        optimizer=client_optimizer_fn,
        client_weighting=client_weighting,
        metrics_aggregator=metrics_aggregator,
        loop_implementation=loop_implementation,
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


def build_unweighted_fed_avg(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    client_optimizer_fn: optimizer_base.Optimizer,
    server_optimizer_fn: Optional[optimizer_base.Optimizer] = None,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.UnweightedAggregationFactory] = None,
    metrics_aggregator: types.MetricsAggregatorType = metric_aggregator.sum_then_finalize,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
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

  Each time the `next` method is called, the server model is communicated to
  each client using the provided `model_distributor`. For each client, local
  training is performed using `client_optimizer_fn`. Each client computes the
  difference between the client model after training and its initial model.
  These model deltas are then aggregated at the server using an unweighted
  aggregation function. The aggregate model delta is applied at the server using
  a server optimizer.

  Note: the default server optimizer function is
  `tff.learning.optimizers.build_sgdm` with a learning rate of 1.0, which
  corresponds to adding the model delta to the current server model. This
  recovers the original FedAvg algorithm in [McMahan et al.,
  2017](https://arxiv.org/abs/1602.05629). More sophisticated federated
  averaging procedures may use different learning rates or server optimizers.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer`.
    server_optimizer_fn: An optional `tff.learning.optimizers.Optimizer`. By
      default, uses `tff.learning.optimizers.build_sgdm(learning_rate=1.0)`.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.UnweightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.UnweightedMeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
      If `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  if model_aggregator is None:
    model_aggregator = mean.UnweightedMeanFactory()
  py_typecheck.check_type(
      model_aggregator, factory.UnweightedAggregationFactory
  )
  if server_optimizer_fn is None:
    server_optimizer_fn = DEFAULT_SERVER_OPTIMIZER_FN()

  return build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weight_lib.ClientWeighting.UNIFORM,
      model_distributor=model_distributor,
      model_aggregator=factory_utils.as_weighted_aggregator(model_aggregator),
      metrics_aggregator=metrics_aggregator,
      loop_implementation=loop_implementation,
  )
