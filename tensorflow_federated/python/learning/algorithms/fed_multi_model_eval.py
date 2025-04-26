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
from collections.abc import Callable, Mapping, Sequence
import string
from typing import Any, Optional, Union

import federated_language
import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning.metrics import sum_aggregation_factory
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process


_AggregationProcess = aggregation_process.AggregationProcess
_VariableModelFnType = Callable[[], variable.VariableModel]
_ModelFnType = Union[_VariableModelFnType, functional.FunctionalModel]


def _build_local_evaluation(
    model_fn: Callable[[], variable.VariableModel],
    model_weights_type: federated_language.StructType,
    batch_type: federated_language.Type,
    loop_implementation: loop_builder.LoopImplementation,
) -> federated_language.framework.Computation:
  """Builds the local TFF computation for evaluation of the given model.

  This produces an unplaced function that evaluates a
  `tff.learning.models.VariableModel`
  on a `tf.data.Dataset`. This function can be mapped to placed data, i.e.
  is mapped to client placed data in `build_federated_evaluation`.

  The TFF type notation for the returned computation is:

  ```
  (<M, D*> → <local_outputs=N, num_examples=tf.int64>)
  ```

  Where `M` is the model weights type structure, `D` is the type structure of a
  single data point, and `N` is the type structure of the local metrics.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`.
    model_weights_type: The `federated_language.Type` of the model parameters
      that will be used to initialize the model during evaluation.
    batch_type: The type of one entry in the dataset.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A federated computation (an instance of `federated_language.Computation`)
    that accepts
    model parameters and sequential data, and returns the evaluation metrics.
  """

  @tensorflow_computation.tf_computation(
      model_weights_type, federated_language.SequenceType(batch_type)
  )
  @tf.function
  def client_eval(incoming_model_weights, dataset):
    """Returns local outputs after evaluating `model_weights` on `dataset`."""
    with tf.init_scope():
      model = model_fn()
    model_weights = model_weights_lib.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda v, t: v.assign(t), model_weights, incoming_model_weights
    )

    def reduce_fn(num_examples, batch):
      model_output = model.forward_pass(batch, training=False)
      if model_output.num_examples is None:
        # Compute shape from the size of the predictions if model didn't use the
        # batch size.
        return (
            num_examples
            + tf.shape(model_output.predictions, out_type=tf.int64)[0]
        )
      else:
        return num_examples + tf.cast(model_output.num_examples, tf.int64)

    dataset_reduce_fn = loop_builder.build_training_loop(
        loop_implementation=loop_implementation
    )
    num_examples = dataset_reduce_fn(
        reduce_fn, dataset, lambda: tf.zeros([], dtype=tf.int64)
    )
    model_output = model.report_local_unfinalized_metrics()
    return collections.OrderedDict(
        local_outputs=model_output, num_examples=num_examples
    )

  return client_eval


def _build_functional_local_evaluation(
    model: functional.FunctionalModel,
    model_weights_type: federated_language.StructType,
    batch_type: Union[
        federated_language.StructType, federated_language.TensorType
    ],
) -> federated_language.framework.Computation:
  """Creates client evaluation logic for a functional model.

  This produces an unplaced function that evaluates a
  `tff.learning.models.FunctionalModel` on a `tf.data.Dataset`. This function
  can be mapped to placed data.

  The TFF type notation for the returned computation is:

  ```
  (<M, D*> → <local_outputs=N>)
  ```

  Where `M` is the model weights type structure, `D` is the type structure of a
  single data point, and `N` is the type structure of the local metrics.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    model_weights_type: The `federated_language.Type` of the model parameters
      that will be used in the forward pass.
    batch_type: The type of one entry in the dataset.

  Returns:
    A federated computation (an instance of `federated_language.Computation`)
    that accepts
    model parameters and sequential data, and returns the evaluation metrics.
  """

  @tensorflow_computation.tf_computation(
      model_weights_type, federated_language.SequenceType(batch_type)
  )
  @tf.function
  def local_eval(weights, dataset):
    metrics_state = model.initialize_metrics_state()

    for batch in iter(dataset):
      if isinstance(batch, Mapping):
        x = batch['x']
        y = batch['y']
      else:
        x, y = batch

      batch_output = model.predict_on_batch(weights, x, training=False)
      batch_loss = model.loss(output=batch_output, label=y)
      predictions = tf.nest.flatten(batch_output)[0]
      batch_num_examples = tf.shape(predictions)[0]

      # TODO: b/272099796 - Update `update_metrics_state` of FunctionalModel
      metrics_state = model.update_metrics_state(
          metrics_state,
          batch_output=variable.BatchOutput(
              loss=batch_loss,
              predictions=batch_output,
              num_examples=batch_num_examples,
          ),
          labels=y,
      )

    unfinalized_metrics = metrics_state
    return unfinalized_metrics

  return local_eval


def _build_fed_eval_client_work(
    model_fn: Callable[[], variable.VariableModel],
    metrics_aggregation_process: Optional[_AggregationProcess],
    model_weights_type: federated_language.StructType,
    loop_implementation: loop_builder.LoopImplementation,
) -> client_works.ClientWorkProcess:
  """Builds a `ClientWorkProcess` that performs model evaluation at clients."""

  def _tensor_type_from_tensor_like(x):
    x_as_tensor = tf.convert_to_tensor(x)
    return tensorflow_types.to_type((x_as_tensor.dtype, x_as_tensor.shape))

  with tf.Graph().as_default():
    model = model_fn()
    batch_type = tensorflow_types.to_type(model.input_spec)
    if metrics_aggregation_process is None:
      unfinalized_metrics = model.report_local_unfinalized_metrics()
      unfinalized_metrics_spec = tf.nest.map_structure(
          _tensor_type_from_tensor_like, unfinalized_metrics
      )

  if metrics_aggregation_process is None:
    # TODO: b/319261270 - Avoid the need for inferring types here, if possible.
    metrics_finalizers = model.metric_finalizers()
    unfinalized_metrics_type = federated_language.StructWithPythonType(
        unfinalized_metrics_spec, collections.OrderedDict
    )
    factory = sum_aggregation_factory.SumThenFinalizeFactory(metrics_finalizers)
    metrics_aggregation_process = factory.create(unfinalized_metrics_type)
  else:
    py_typecheck.check_type(
        metrics_aggregation_process,
        _AggregationProcess,
        'metrics_aggregation_process',
    )

  @federated_language.federated_computation
  def init_fn():
    return metrics_aggregation_process.initialize()

  client_update_computation = _build_local_evaluation(
      model_fn,
      model_weights_type,
      batch_type,
      loop_implementation=loop_implementation,
  )

  @federated_language.federated_computation(
      init_fn.type_signature.result,
      federated_language.FederatedType(
          model_weights_type, federated_language.CLIENTS
      ),
      federated_language.FederatedType(
          federated_language.SequenceType(batch_type),
          federated_language.CLIENTS,
      ),
  )
  def next_fn(state, model_weights, client_data):
    model_outputs = federated_language.federated_map(
        client_update_computation, (model_weights, client_data)
    )
    metrics_output = metrics_aggregation_process.next(
        state, model_outputs.local_outputs
    )
    current_round_metrics, total_rounds_metrics = metrics_output.result
    measurements = federated_language.federated_zip(
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=current_round_metrics,
                total_rounds_metrics=total_rounds_metrics,
            )
        )
    )
    # Return empty result as no model update will be performed for evaluation.
    empty_client_result = federated_language.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        federated_language.CLIENTS,
    )
    return measured_process.MeasuredProcessOutput(
        metrics_output.state, empty_client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_functional_fed_eval_client_work(
    model: functional.FunctionalModel,
    metrics_aggregation_process: Optional[_AggregationProcess],
    model_weights_type: federated_language.StructType,
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
  batch_type = tensorflow_types.to_type(model.input_spec)
  local_eval = _build_functional_local_evaluation(
      model,
      tuple_weights_type,  # pytype: disable=wrong-arg-types
      batch_type,
  )

  if metrics_aggregation_process is None:
    unfinalized_metrics_type = local_eval.type_signature.result
    metrics_aggregation_process = (
        sum_aggregation_factory.SumThenFinalizeFactory(
            model.finalize_metrics
        ).create(unfinalized_metrics_type)
    )

  @federated_language.federated_computation
  def init_fn():
    return metrics_aggregation_process.initialize()

  @tensorflow_computation.tf_computation(
      model_weights_type, federated_language.SequenceType(batch_type)
  )
  def client_update_computation(model_weights, client_data):
    # Switch to the tuple expected by FunctionalModel.
    tuple_weights = (model_weights.trainable, model_weights.non_trainable)
    return local_eval(tuple_weights, client_data)

  @federated_language.federated_computation(
      init_fn.type_signature.result,
      federated_language.FederatedType(
          model_weights_type, federated_language.CLIENTS
      ),
      federated_language.FederatedType(
          federated_language.SequenceType(batch_type),
          federated_language.CLIENTS,
      ),
  )
  def next_fn(state, model_weights, client_data):
    unfinalized_metrics = federated_language.federated_map(
        client_update_computation, (model_weights, client_data)
    )
    metrics_output = metrics_aggregation_process.next(
        state, unfinalized_metrics
    )
    current_round_metrics, total_rounds_metrics = metrics_output.result
    measurements = federated_language.federated_zip(
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=current_round_metrics,
                total_rounds_metrics=total_rounds_metrics,
            )
        )
    )
    # Return empty result as no model update will be performed for evaluation.
    empty_client_result = federated_language.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        federated_language.CLIENTS,
    )
    return measured_process.MeasuredProcessOutput(
        metrics_output.state, empty_client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_initial_model_weights_fn(
    model_fns: Sequence[_ModelFnType],
) -> federated_language.framework.Computation:
  """Builds a tff.framework.Computation that returns the initial model weights."""

  @tensorflow_computation.tf_computation()
  def initial_model_weights_fn():
    initial_model_weights: list[model_weights_lib.ModelWeights] = []
    for model_fn in model_fns:
      if not callable(model_fn):
        if not isinstance(model_fn, functional.FunctionalModel):
          raise TypeError(
              'If `model_fn` is not a callable, it must be an instance '
              f'tff.learning.models.FunctionalModel. Got {type(model_fn)}'
          )
        trainable_weights, non_trainable_weights = model_fn.initial_weights
        initial_model_weights.append(
            model_weights_lib.ModelWeights(
                tuple(tf.convert_to_tensor(w) for w in trainable_weights),
                tuple(tf.convert_to_tensor(w) for w in non_trainable_weights),
            )
        )
      else:
        initial_model_weights.append(
            model_weights_lib.ModelWeights.from_model(
                model_fn()  # pytype: disable=not-callable
            )
        )
    return dict(zip(string.ascii_lowercase, initial_model_weights))

  return initial_model_weights_fn


def build_fed_multi_model_eval(
    model_fns: Sequence[_ModelFnType],
    model_distributor: Optional[distributors.DistributionProcess] = None,
    metrics_aggregation_processes: Optional[
        Sequence[aggregation_process.AggregationProcess]
    ] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> learning_process.LearningProcess:
  """Builds a learning process that performs federated multi-model evaluation.

  Note: The models are evaluated independently on the clients using the same
  client data. It is required that all models have the same input spec.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  federated evaluation on clients for multiple models. The learning process has
  the following methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `federated_language.Computation` with type signature `( ->
  S@SERVER)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState`
      representing the initial state of the server.
  *   `next`: A `federated_language.Computation` with type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `LearningAlgorithmState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. The output `L` contains the updated
      server state, as well as aggregated metrics at the server, including
      client evaluation metrics and any other metrics from distribution and
      aggregation processes.
  *   `get_model_weights`: A `federated_language.Computation` with type
  signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next`, and `M` represents the type
      of the model weights used during evaluation.
  *   `set_model_weights`: A `federated_language.Computation` with type
  signature
      `(<S, M> -> S)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `M` represents the type of the model weights
      used during evaluation.

  Each time `next` is called, the server models are broadcast to each client
  using a distributor. Each client evaluates the models and reports local
  unfinalized metrics. The local unfinalized metrics are then aggregated and
  finalized at server using the metrics aggregator. Both current round and
  total rounds metrics will be produced. There are no update of the server model
  during the evaluation process.

  Args:
    model_fns: A sequence of model functions, one for each model. A model
      function should be a no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error. All models must have the same input spec. All model_fns should be
      of the same type. At least one model_fn should be provided.
    model_distributor: An optional `tff.learning.templates.DistributionProcess`
      that broadcasts the model weights on the server to the clients. It must
      support the signature `(input_values@SERVER -> output_values@CLIENTS)` and
      have empty state. If None, the server model is broadcast to the clients
      using the default `federated_language.federated_broadcast`.
    metrics_aggregation_processes: An optional sequence of
      `tff.templates.AggregationProcess` which aggregates the local unfinalized
      metrics at clients to server and finalizes the metrics at server. The
      `tff.templates.AggregationProcess` accumulates unfinalized metrics across
      round in the state, and produces a tuple of current round metrics and
      total rounds metrics in the result. If None, the
      `tff.templates.AggregationProcess` created by the `SumThenFinalizeFactory`
      with metric finalizers defined in the model is used.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tff.learning.templates.LearningProcess` performs federated evaluation on
    clients, and returns updated state and metrics.

  Raises:
    TypeError: If any argument type mismatches.
  """
  if not model_fns:
    raise ValueError('At least one model_fn should be provided.')
  initial_model_weights_fn = _build_initial_model_weights_fn(model_fns)
  multi_model_weights_type = initial_model_weights_fn.type_signature.result
  if not isinstance(multi_model_weights_type, federated_language.StructType):
    raise ValueError(
        'model_weights_type must be a StructType. Found:'
        f' {type(multi_model_weights_type)}.'
    )

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(
        multi_model_weights_type
    )
  else:
    py_typecheck.check_type(
        model_distributor, distributors.DistributionProcess, 'model_distributor'
    )

  if metrics_aggregation_processes is None:
    metrics_aggregation_processes = [None] * len(model_fns)
  else:
    if len(metrics_aggregation_processes) != len(model_fns):
      raise ValueError(
          'If provided, metrics_aggregation_processes should be a sequence of'
          f' length {len(model_fns)} to match the number of model_fns.'
      )

  # Create ClientWorkProcess for each model, identified by letters ('a', 'b',
  # 'c', etc.).
  client_work_processes: dict[str, client_works.ClientWorkProcess] = {}
  for (
      model_id,
      model_fn,
      model_weights_type,
      metrics_aggregation_process,
  ) in zip(
      string.ascii_lowercase,
      model_fns,
      multi_model_weights_type,
      metrics_aggregation_processes,
  ):
    if not callable(model_fn):
      client_work_processes[model_id] = _build_functional_fed_eval_client_work(
          model_fn,
          metrics_aggregation_process,
          model_weights_type,
      )
    else:
      client_work_processes[model_id] = _build_fed_eval_client_work(
          model_fn,
          metrics_aggregation_process,
          model_weights_type,
          loop_implementation,
      )

  @federated_language.federated_computation
  def multi_init():
    return federated_language.federated_zip({
        model_id: client_work.initialize()
        for model_id, client_work in client_work_processes.items()
    })

  # Get the type of the client data from the first ClientWorkProcess. It is
  # required that all models have the same input spec.
  client_data_type = client_work_processes['a'].next.type_signature.parameter[2]

  @federated_language.federated_computation(
      multi_init.type_signature.result,
      federated_language.FederatedType(
          multi_model_weights_type, federated_language.CLIENTS
      ),
      client_data_type,
  )
  def multi_next(state, model_weights, client_data):
    metrics: dict[str, Any] = {}
    new_states: dict[str, Any] = {}
    for model_id, client_work in client_work_processes.items():
      output = client_work.next(
          state[model_id], model_weights[model_id], client_data
      )
      new_states[model_id] = output.state
      metrics[model_id] = output.measurements
    # Evaluation returns no result for model update.
    empty_client_result = federated_language.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        federated_language.CLIENTS,
    )
    return measured_process.MeasuredProcessOutput(
        federated_language.federated_zip(new_states),
        empty_client_result,
        federated_language.federated_zip(metrics),
    )

  client_work = client_works.ClientWorkProcess(multi_init, multi_next)
  client_work_result_type = federated_language.FederatedType(
      client_works.ClientResult(update=(), update_weight=()),
      federated_language.CLIENTS,
  )
  model_update_type = client_work_result_type.member.update  # pytype: disable=attribute-error
  model_update_weight_type = client_work_result_type.member.update_weight  # pytype: disable=attribute-error
  model_aggregator_factory = mean.MeanFactory()
  model_aggregator = model_aggregator_factory.create(
      model_update_type, model_update_weight_type
  )

  # The finalizer performs no update on model weights.
  finalizer = finalizers.build_identity_finalizer(
      multi_model_weights_type,
      model_aggregator.next.type_signature.result.result.member,  # pytype: disable=attribute-error
  )

  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      model_aggregator,
      finalizer,
  )


def _combine_multi_state(
    state_dict: dict[str, Any],
) -> composers.LearningAlgorithmState:
  """Combines the state of multiple models into a single state."""
  global_model_weights = federated_language.federated_zip({
      model_id: state.global_model_weights
      for model_id, state in state_dict.items()
  })
  distributor = federated_language.federated_zip(
      {model_id: state.distributor for model_id, state in state_dict.items()}
  )
  client_work = federated_language.federated_zip(
      {model_id: state.client_work for model_id, state in state_dict.items()}
  )
  aggregator = federated_language.federated_zip(
      {model_id: state.aggregator for model_id, state in state_dict.items()}
  )
  finalizer = federated_language.federated_zip(
      {model_id: state.finalizer for model_id, state in state_dict.items()}
  )
  return federated_language.federated_zip(
      composers.LearningAlgorithmState(
          global_model_weights=global_model_weights,
          distributor=distributor,
          client_work=client_work,
          aggregator=aggregator,
          finalizer=finalizer,
      )
  )


def _combine_multi_metrics(
    metrics_dict: dict[str, Any],
) -> collections.OrderedDict[str, Any]:
  """Combines the metrics of multiple models into a single metrics."""
  return federated_language.federated_zip(
      collections.OrderedDict(
          distributor=federated_language.federated_zip({
              model_id: metrics_dict[model_id].distributor
              for model_id in metrics_dict
          }),
          client_work=federated_language.federated_zip({
              model_id: metrics_dict[model_id].client_work
              for model_id in metrics_dict
          }),
          aggregator=federated_language.federated_zip({
              model_id: metrics_dict[model_id].aggregator
              for model_id in metrics_dict
          }),
          finalizer=federated_language.federated_zip({
              model_id: metrics_dict[model_id].finalizer
              for model_id in metrics_dict
          }),
      )
  )


def _get_single_model_state(
    state: composers.LearningAlgorithmState,
    model_id: str,
) -> composers.LearningAlgorithmState:
  """Returns the state of a single model."""
  return federated_language.federated_zip(
      composers.LearningAlgorithmState(
          global_model_weights=getattr(state.global_model_weights, model_id),
          distributor=getattr(state.distributor, model_id),
          client_work=getattr(state.client_work, model_id),
          aggregator=getattr(state.aggregator, model_id),
          finalizer=getattr(state.finalizer, model_id),
      )
  )


def build_fed_multi_model_eval_from_processes(
    eval_processes: Sequence[learning_process.LearningProcess],
) -> learning_process.LearningProcess:
  """Builds a multi-model eval process from a sequence of eval processes.

  The eval processes are assumed to be single-model eval processes that have
  the same input spec. All eval processes should be
  `tff.learning.templates.LearningProcess` with the state type of
  `tff.learning.templates.LearningAlgorithmState`.

  Args:
    eval_processes: A sequence of single-model eval processes.

  Returns:
    A multi-model eval process.

  Raises:
    ValueError: If the eval_processes is empty or if the eval processes have
      different dataset types.
  """
  if len(eval_processes) < 2:
    raise ValueError(
        'At least two evaluation processes must be provided for multi-model'
        ' evaluation.'
    )

  # Get the type of the client data from the first eval process. It is
  # required that all models have the same input spec.
  # pytype: disable=unsupported-operands
  dataset_type = eval_processes[0].next.type_signature.parameter[1]
  client_data_matches = lambda process: process.next.type_signature.parameter[
      1
  ].is_equivalent_to(dataset_type)
  # pytype: enable=unsupported-operands
  if not all(client_data_matches(p) for p in eval_processes[1:]):
    raise ValueError(
        'All evaluation processes must have the same dataset type.'
    )

  @federated_language.federated_computation()
  def init_fn():
    multi_state = {
        model_id: eval_processes.initialize()
        for model_id, eval_processes in zip(
            string.ascii_letters, eval_processes
        )
    }
    return _combine_multi_state(multi_state)

  state_type = init_fn.type_signature.result

  @federated_language.federated_computation(
      state_type,
      dataset_type,
  )
  def next_fn(state, client_data):
    multi_state = collections.OrderedDict()
    multi_metrics = collections.OrderedDict()

    for model_id, eval_process in zip(string.ascii_letters, eval_processes):
      current_state = _get_single_model_state(state, model_id)
      output = eval_process.next(current_state, client_data)
      multi_state[model_id] = output.state
      multi_metrics[model_id] = output.metrics

    return learning_process.LearningProcessOutput(
        state=_combine_multi_state(multi_state),
        metrics=_combine_multi_metrics(multi_metrics),
    )

  @tensorflow_computation.tf_computation(
      state_type.member,
  )
  def get_model_weights_fn(state):
    return state.global_model_weights

  @tensorflow_computation.tf_computation(
      state_type.member,
      get_model_weights_fn.type_signature.result,
  )
  def set_model_weights_fn(state, model_weights):
    return composers.LearningAlgorithmState(
        global_model_weights=model_weights,
        distributor=state.distributor,
        client_work=state.client_work,
        aggregator=state.aggregator,
        finalizer=state.finalizer,
    )

  multi_eval_process = learning_process.LearningProcess(
      init_fn,
      next_fn,
      get_model_weights_fn,
      set_model_weights_fn,
  )
  return multi_eval_process
