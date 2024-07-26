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
"""An implementation of Mime Lite algorithm.

The algorithm is proposed by the paper:

Breaking the centralized barrier for cross-device federated learning.
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf
"""

import collections
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import tensor_utils
from tensorflow_federated.python.learning.metrics import aggregator as metric_aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process


def _choose_client_weight(weighting, has_non_finite_delta, num_examples):
  if has_non_finite_delta > 0:
    return tf.constant(0.0, tf.float32)
  else:
    if weighting == client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      return tf.cast(num_examples, tf.float32)
    elif weighting == client_weight_lib.ClientWeighting.UNIFORM:
      return tf.constant(1.0, tf.float32)
    else:
      raise ValueError(f'Unexpected weighting argument: {weighting}')


def _build_client_update_fn_for_mime_lite(
    model_fn: Callable[[], variable.VariableModel],
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    loop_implementation: loop_builder.LoopImplementation,
):
  """Builds the `tf_computation` for Mime Lite client training."""

  @tensorflow_computation.tf_computation
  def client_update_fn(global_optimizer_state, initial_weights, data):
    model = model_fn()
    dataset_reduce_fn = loop_builder.build_training_loop(
        loop_implementation=loop_implementation
    )
    weight_tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_weights_lib.weights_type_from_model(model)
    )

    @tf.function
    def client_update(global_optimizer_state, initial_weights, data):
      model_weights = model_weights_lib.ModelWeights.from_model(model)
      tf.nest.map_structure(
          lambda a, b: a.assign(b), model_weights, initial_weights
      )

      def full_gradient_reduce_fn(state, batch):
        """Sums individual gradients, to be later divided by num_examples."""
        gradient_sum, num_examples_sum = state
        with tf.GradientTape() as tape:
          output = model.forward_pass(batch, training=True)
        if output.num_examples is None:
          num_examples = tf.shape(output.predictions, out_type=tf.int64)[0]
        else:
          num_examples = tf.cast(output.num_examples, tf.int64)
        # TODO: b/161529310 - We flatten and convert to tuple, as tf.data
        # iterators would try to stack the tensors in list into a single tensor.
        gradients = tuple(
            tf.nest.flatten(tape.gradient(output.loss, model_weights.trainable))
        )
        gradient_sum = tf.nest.map_structure(
            lambda g_sum, g: g_sum + g * tf.cast(num_examples, g.dtype),
            gradient_sum,
            gradients,
        )
        num_examples_sum += num_examples
        return gradient_sum, num_examples_sum

      def initial_state_for_full_gradient_reduce_fn():
        initial_gradient_sum = tf.nest.map_structure(
            lambda spec: tf.zeros(spec.shape, spec.dtype),
            tuple(tf.nest.flatten(weight_tensor_specs.trainable)),
        )
        initial_num_examples_sum = tf.constant(0, tf.int64)
        return initial_gradient_sum, initial_num_examples_sum

      full_gradient, num_examples = dataset_reduce_fn(
          full_gradient_reduce_fn,
          data,
          initial_state_for_full_gradient_reduce_fn,
      )
      # Compute the average gradient.
      full_gradient = tf.nest.map_structure(
          lambda g: tf.math.divide_no_nan(g, tf.cast(num_examples, g.dtype)),
          full_gradient,
      )

      # Resets the local model variables, including metrics states, as we are
      # not interested in metrics based on the full gradient evaluation, only
      # from the subsequent training.
      model.reset_metrics()

      def train_reduce_fn(state, batch):
        with tf.GradientTape() as tape:
          output = model.forward_pass(batch, training=True)
        gradients = tape.gradient(output.loss, model_weights.trainable)
        # Mime Lite keeps optimizer state unchanged during local training.
        _, updated_weights = optimizer.next(
            global_optimizer_state, model_weights.trainable, gradients
        )
        tf.nest.map_structure(
            lambda a, b: a.assign(b), model_weights.trainable, updated_weights
        )
        return state

      # Performs local training, updating `tf.Variable`s in `model_weights`.
      initial_state_fn = lambda: tf.zeros(shape=[0])
      dataset_reduce_fn(train_reduce_fn, data, initial_state_fn)

      client_weights_delta = tf.nest.map_structure(
          tf.subtract, initial_weights.trainable, model_weights.trainable
      )
      model_output = model.report_local_unfinalized_metrics()

      # TODO: b/122071074 - Consider moving this functionality into aggregation.
      client_weights_delta, has_non_finite_delta = (
          tensor_utils.zero_all_if_any_non_finite(client_weights_delta)
      )
      client_weight = _choose_client_weight(
          client_weighting, has_non_finite_delta, num_examples
      )
      return (
          client_works.ClientResult(
              update=client_weights_delta, update_weight=client_weight
          ),
          model_output,
          full_gradient,
      )

    return client_update(global_optimizer_state, initial_weights, data)

  return client_update_fn


def _build_mime_lite_client_work(
    model_fn: Callable[[], variable.VariableModel],
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory
    ] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for Mime Lite.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    optimizer: A `tff.learning.optimizers.Optimizer` which will be used for both
      creating and updating a global optimizer state, as well as optimization at
      clients given the global state, which is fixed during the optimization.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
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
    A `ClientWorkProcess`.
  """
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.MeanFactory()
  py_typecheck.check_type(
      full_gradient_aggregator, factory.WeightedAggregationFactory
  )
  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    metrics_aggregation_fn = metrics_aggregator(
        model.metric_finalizers(),
    )
  element_type = computation_types.tensorflow_to_type(model.input_spec)
  data_type = computation_types.SequenceType(element_type)
  weights_type = model_weights_lib.weights_type_from_model(model)
  weight_tensor_specs = type_conversions.type_to_tf_tensor_specs(weights_type)

  full_gradient_aggregator = full_gradient_aggregator.create(
      weights_type.trainable, computation_types.TensorType(np.float32)
  )

  @federated_computation.federated_computation
  def init_fn():
    specs = weight_tensor_specs.trainable
    optimizer_state = intrinsics.federated_eval(
        tensorflow_computation.tf_computation(
            lambda: optimizer.initialize(specs)
        ),
        placements.SERVER,
    )
    aggregator_state = full_gradient_aggregator.initialize()
    return intrinsics.federated_zip((optimizer_state, aggregator_state))

  client_update_fn = _build_client_update_fn_for_mime_lite(
      model_fn,
      optimizer,
      client_weighting,
      loop_implementation=loop_implementation,
  )

  @tensorflow_computation.tf_computation(
      init_fn.type_signature.result.member[0], weights_type.trainable
  )
  def update_optimizer_state(state, aggregate_gradient):
    whimsy_weights = tf.nest.map_structure(
        lambda g: tf.zeros(g.shape, g.dtype), aggregate_gradient
    )
    updated_state, _ = optimizer.next(state, whimsy_weights, aggregate_gradient)
    return updated_state

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, weights, client_data):
    optimizer_state, aggregator_state = state
    optimizer_state_at_clients = intrinsics.federated_broadcast(optimizer_state)
    client_result, model_outputs, full_gradient = intrinsics.federated_map(
        client_update_fn, (optimizer_state_at_clients, weights, client_data)
    )
    full_gradient_agg_output = full_gradient_aggregator.next(
        aggregator_state, full_gradient, client_result.update_weight
    )
    updated_optimizer_state = intrinsics.federated_map(
        update_optimizer_state,
        (optimizer_state, full_gradient_agg_output.result),
    )

    new_state = intrinsics.federated_zip(
        (updated_optimizer_state, full_gradient_agg_output.state)
    )
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        new_state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_functional_client_update_fn_for_mime_lite(
    *,
    model: functional.FunctionalModel,
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    loop_implementation: loop_builder.LoopImplementation,
) -> Callable[..., Any]:
  """Builds the `tf_computation` for client training for FunctionalModels."""

  @tensorflow_computation.tf_computation
  def client_update_fn(global_optimizer_state, incoming_weights, data):
    dataset_reduce_fn = loop_builder.build_training_loop(
        loop_implementation=loop_implementation
    )
    weight_tensor_specs = tf.nest.map_structure(
        lambda t: tf.TensorSpec(shape=t.shape, dtype=t.dtype),
        model.initial_weights,
    )

    @tf.function
    def client_update(
        global_optimizer_state: Any,
        incoming_weights: model_weights_lib.ModelWeights,
        data: tf.data.Dataset,
    ) -> Any:
      trainable_weights, _ = incoming_weights  # pytype: disable=attribute-error

      def full_gradient_reduce_fn(state, batch):
        """Sums individual gradients, to be later divided by num_examples."""
        gradient_sum, num_examples_sum = state
        if isinstance(batch, Mapping):
          x = batch['x']
          y = batch['y']
        elif isinstance(batch, Sequence):
          x, y = batch
        else:
          raise TypeError(
              'Examples yielded from the dataset must be either a '
              '`collections.abc.Mapping` or a `collections.abc.Sequence`. Got '
              f'{type(batch)}'
          )

        with tf.GradientTape() as tape:
          tape.watch(trainable_weights)
          batch_output = model.predict_on_batch(
              model_weights=incoming_weights, x=x, training=True
          )
          batch_loss = model.loss(output=batch_output, label=y)

        predictions = tf.nest.flatten(batch_output)[0]
        batch_num_examples = tf.shape(predictions)[0]
        gradients = tape.gradient(batch_loss, trainable_weights)
        gradient_sum = tf.nest.map_structure(
            lambda g_sum, g: g_sum + g * tf.cast(batch_num_examples, g.dtype),
            gradient_sum,
            gradients,
        )
        num_examples_sum += tf.cast(batch_num_examples, tf.int64)
        return gradient_sum, num_examples_sum

      def initial_state_for_full_gradient_reduce_fn():
        trainable_specs, _ = weight_tensor_specs
        initial_gradient_sum = tuple(
            tf.zeros(spec.shape, spec.dtype) for spec in trainable_specs
        )
        initial_num_examples_sum = tf.constant(0, tf.int64)
        return initial_gradient_sum, initial_num_examples_sum

      # Compute the average gradient over all examples without updating the
      # model.
      full_gradient, num_examples = dataset_reduce_fn(
          full_gradient_reduce_fn,
          data,
          initial_state_for_full_gradient_reduce_fn,
      )
      full_gradient = tf.nest.map_structure(
          lambda g: tf.math.divide_no_nan(g, tf.cast(num_examples, g.dtype)),
          full_gradient,
      )

      def train_reduce_fn(state, batch):
        model_weights, metrics_state = state
        trainable_weights, non_trainable_weights = model_weights
        if isinstance(batch, Mapping):
          x = batch['x']
          y = batch['y']
        elif isinstance(batch, Sequence):
          x, y = batch
        else:
          raise TypeError(
              'Examples yielded from the dataset must be either a '
              '`collections.abc.Mapping` or a `collections.abc.Sequence`. Got '
              f'{type(batch)}'
          )

        with tf.GradientTape() as tape:
          tape.watch(trainable_weights)
          batch_output = model.predict_on_batch(
              model_weights=model_weights, x=x, training=True
          )
          batch_loss = model.loss(output=batch_output, label=y)

        gradients = tape.gradient(batch_loss, trainable_weights)
        predictions = tf.nest.flatten(batch_output)[0]
        batch_num_examples = tf.shape(predictions)[0]

        # TODO: b/272099796 - Update `update_metrics_state` of FunctionalModel
        metrics_state = model.update_metrics_state(
            metrics_state,
            batch_output=variable.BatchOutput(
                loss=batch_loss,
                predictions=batch_output,
                num_examples=tf.cast(batch_num_examples, tf.int64),
            ),
            labels=y,
        )
        # Mime Lite keeps optimizer state unchanged during local training.
        _, updated_weights = optimizer.next(
            global_optimizer_state, trainable_weights, gradients
        )
        return (updated_weights, non_trainable_weights), metrics_state

      def initial_training_weights():
        return incoming_weights, model.initialize_metrics_state()

      model_weights, unfinalized_metrics = dataset_reduce_fn(
          train_reduce_fn, data, initial_training_weights
      )

      incoming_training_weights, _ = incoming_weights  # pytype: disable=attribute-error
      trainable_weights, _ = model_weights
      client_weights_delta = tf.nest.map_structure(
          tf.subtract, incoming_training_weights, trainable_weights
      )

      client_weights_delta, has_non_finite_delta = (
          tensor_utils.zero_all_if_any_non_finite(client_weights_delta)
      )
      client_weight = _choose_client_weight(
          client_weighting, has_non_finite_delta, num_examples
      )
      return (
          client_works.ClientResult(
              update=client_weights_delta, update_weight=client_weight
          ),
          unfinalized_metrics,
          full_gradient,
      )

    # Convert `tff.learning.models.ModelWeights` type weights back into the
    # initial shape used by the model.
    incoming_weights = (
        incoming_weights.trainable,
        incoming_weights.non_trainable,
    )
    return client_update(global_optimizer_state, incoming_weights, data)

  return client_update_fn


def _build_mime_lite_functional_client_work(
    model: functional.FunctionalModel,
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory
    ] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for MimeLite with FunctionalModels.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    optimizer: A `tff.learning.optimizers.Optimizer` which will be used for both
      creating and updating a global optimizer state, as well as optimization at
      clients given the global state, which is fixed during the optimization.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
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
    A `ClientWorkProcess`.
  """
  py_typecheck.check_type(model, functional.FunctionalModel)
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.MeanFactory()
  py_typecheck.check_type(
      full_gradient_aggregator, factory.WeightedAggregationFactory
  )
  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize

  element_type = computation_types.tensorflow_to_type(model.input_spec)
  data_type = computation_types.SequenceType(element_type)

  def ndarray_to_tensorspec(ndarray):
    return tf.TensorSpec(
        shape=ndarray.shape, dtype=tf.dtypes.as_dtype(ndarray.dtype)
    )

  # Wrap in a `ModelWeights` structure that is required by the `finalizer.`
  weights_spec = model_weights_lib.ModelWeights(
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[0]),
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[1]),
  )
  weights_type = computation_types.tensorflow_to_type(weights_spec)
  weight_tensor_specs = type_conversions.type_to_tf_tensor_specs(weights_type)

  full_gradient_aggregator = full_gradient_aggregator.create(
      weights_type.trainable,  # pytype: disable=attribute-error
      computation_types.TensorType(np.float32),
  )

  @federated_computation.federated_computation
  def init_fn():
    optimizer_state = intrinsics.federated_eval(
        tensorflow_computation.tf_computation(
            lambda: optimizer.initialize(weight_tensor_specs.trainable)
        ),
        placements.SERVER,
    )
    aggregator_state = full_gradient_aggregator.initialize()
    return intrinsics.federated_zip((optimizer_state, aggregator_state))

  client_update_fn = _build_functional_client_update_fn_for_mime_lite(
      model=model,
      optimizer=optimizer,
      client_weighting=client_weighting,
      loop_implementation=loop_implementation,
  )

  aggregator_state_type, _ = init_fn.type_signature.result.member

  @tensorflow_computation.tf_computation(
      aggregator_state_type,
      weights_type.trainable,  # pytype: disable=attribute-error
  )
  def update_optimizer_state(state, aggregate_gradient):
    whimsy_weights = tf.nest.map_structure(
        lambda g: tf.zeros(g.shape, g.dtype), aggregate_gradient
    )
    updated_state, _ = optimizer.next(state, whimsy_weights, aggregate_gradient)
    return updated_state

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, weights, client_data):
    optimizer_state, aggregator_state = state
    optimizer_state_at_clients = intrinsics.federated_broadcast(optimizer_state)
    client_result, unfinalized_metrics, full_gradient = (
        intrinsics.federated_map(
            client_update_fn, (optimizer_state_at_clients, weights, client_data)
        )
    )
    full_gradient_agg_output = full_gradient_aggregator.next(
        aggregator_state, full_gradient, client_result.update_weight
    )
    updated_optimizer_state = intrinsics.federated_map(
        update_optimizer_state,
        (optimizer_state, full_gradient_agg_output.result),
    )

    new_state = intrinsics.federated_zip(
        (updated_optimizer_state, full_gradient_agg_output.state)
    )

    metrics_aggregation_fn = metrics_aggregator(
        model.finalize_metrics, unfinalized_metrics.type_signature.member
    )
    train_metrics = metrics_aggregation_fn(unfinalized_metrics)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        new_state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_scheduled_mime_lite_client_work(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    learning_rate_fn: Callable[[int], float],
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory
    ] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates `ClientWorkProcess` for Mimelite with learning rate schedule.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    learning_rate_fn: A callable accepting an integer round number and returning
      a float to be used as a learning rate for the optimizer.
    optimizer: A `tff.learning.optimizers.Optimizer` which will be used for both
      creating and updating a global optimizer state, as well as optimization at
      clients given the global state, which is fixed during the optimization.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
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
    A `ClientWorkProcess`.
  """
  if callable(model_fn):
    client_work = _build_mime_lite_client_work(
        model_fn,
        optimizer,
        client_weighting,
        full_gradient_aggregator,
        metrics_aggregator,
        loop_implementation=loop_implementation,
    )
  elif isinstance(model_fn, functional.FunctionalModel):
    client_work = _build_mime_lite_functional_client_work(
        model_fn,
        optimizer,
        client_weighting,
        full_gradient_aggregator,
        metrics_aggregator,
        loop_implementation=loop_implementation,
    )
  else:
    raise TypeError(
        'When `model_fn` is not a callable, it must be an instance'
        ' of tff.learning.models.FunctionalModel. Instead got a: '
        f'{type(model_fn)}'
    )

  federated_mime_state_type, federated_weights_type, federated_data_type = (
      client_work.next.type_signature.parameter
  )  # pytype: disable=attribute-error
  data_type = federated_data_type.member
  weights_type = federated_weights_type.member
  mime_state_type = federated_mime_state_type.member

  @tensorflow_computation.tf_computation(mime_state_type)
  def initialize_learning_rate(mime_state):
    # mime_state is a tuple of the form (optimizer_state, aggregator_state)
    mime_state[0][optimizer_base.LEARNING_RATE_KEY] = learning_rate_fn(0)
    return mime_state

  @federated_computation.federated_computation
  def init_fn():
    initial_state = client_work.initialize()
    updated_state = intrinsics.federated_map(
        initialize_learning_rate, initial_state
    )
    return intrinsics.federated_zip(
        (intrinsics.federated_value(0, placements.SERVER), updated_state)
    )

  state_type = init_fn.type_signature.result.member

  @tensorflow_computation.tf_computation(state_type)
  def update_state(state):
    round_num = state[0]
    updated_round_num = round_num + 1
    updated_learning_rate = learning_rate_fn(updated_round_num)
    mime_state = state[1]
    mime_state[0][optimizer_base.LEARNING_RATE_KEY] = updated_learning_rate
    return (updated_round_num, mime_state)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, weights, client_data):
    round_num, mime_state = state
    output = client_work.next(mime_state, weights, client_data)
    updated_mime_state = output.state
    outer_state = intrinsics.federated_zip((round_num, updated_mime_state))
    updated_state = intrinsics.federated_map(update_state, outer_state)
    return measured_process.MeasuredProcessOutput(
        updated_state, output.result, output.measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def build_weighted_mime_lite(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    base_optimizer: optimizer_base.Optimizer,
    server_optimizer: optimizer_base.Optimizer = sgdm.build_sgdm(1.0),
    client_weighting: Optional[
        client_weight_lib.ClientWeighting
    ] = client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory
    ] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> learning_process.LearningProcess:
  """Builds a learning process that performs Mime Lite.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  Mime Lite algorithm on client models. The iterative process has the following
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
  training is performed using `optimizer`, where its state is communicated by
  the server, and kept intact during local training. The state is updated only
  at the server based on the full gradient evaluated by the clients based on the
  current server model state. The client full gradients are aggregated by
  weighted `full_gradient_aggregator`. Each client computes the difference
  between the client model after training and its initial model. These model
  deltas are then aggregated by weighted `model_aggregator`. Both of the
  aggregations are weighted, according to `client_weighting`. The aggregate
  model delta is added to the existing server model state.

  The Mime Lite algorithm is based on the paper
  "Breaking the centralized barrier for cross-device federated learning."
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf

  Note that Keras optimizers are not supported. This is due to the Mime Lite
  algorithm applying the optimizer without changing it state at clients
  (optimizer's `tf.Variable`s in the case of Keras), which is not possible with
  Keras optimizers without reaching into private implementation details and
  incurring additional computation and memory cost at clients.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    base_optimizer: A `tff.learning.optimizers.Optimizer` which will be used for
      both creating and updating a global optimizer state, as well as
      optimization at clients given the global state, which is fixed during the
      optimization.
    server_optimizer: A `tff.learning.optimizers.Optimizer` which will be used
      for applying the aggregate model update to the global model weights.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method. By default, weighting by number of examples
      is used.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
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
  """
  py_typecheck.check_type(base_optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(server_optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  if not callable(model_fn):
    if not isinstance(model_fn, functional.FunctionalModel):
      raise TypeError(
          'If `model_fn` is not a callable, it must be an instance of '
          f'tff.learning.models.FunctionalModel. Got {type(model_fn)}'
      )

    @tensorflow_computation.tf_computation
    def initial_model_weights_fn():
      trainable_weights, non_trainable_weights = model_fn.initial_weights
      return model_weights_lib.ModelWeights(
          tuple(tf.convert_to_tensor(w) for w in trainable_weights),
          tuple(tf.convert_to_tensor(w) for w in non_trainable_weights),
      )

  else:

    @tensorflow_computation.tf_computation
    def initial_model_weights_fn():
      model = model_fn()  # pytype: disable=not-callable
      if not isinstance(model, variable.VariableModel):
        raise TypeError(
            'When `model_fn` is a callable, it returns instances of'
            ' tff.learning.models.VariableModel. Instead callable returned'
            f' type: {type(model)}'
        )
      return model_weights_lib.ModelWeights.from_model(model)

  model_weights_type = initial_model_weights_fn.type_signature.result
  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)
  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)
  model_update_type = model_weights_type.trainable
  model_aggregator = model_aggregator.create(
      model_update_type, computation_types.TensorType(np.float32)
  )
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.MeanFactory()
  py_typecheck.check_type(
      full_gradient_aggregator, factory.WeightedAggregationFactory
  )

  if callable(model_fn):
    client_work = _build_mime_lite_client_work(
        model_fn=model_fn,
        optimizer=base_optimizer,
        client_weighting=client_weighting,
        full_gradient_aggregator=full_gradient_aggregator,
        metrics_aggregator=metrics_aggregator,
        loop_implementation=loop_implementation,
    )
  elif isinstance(model_fn, functional.FunctionalModel):
    client_work = _build_mime_lite_functional_client_work(
        model=model_fn,
        optimizer=base_optimizer,
        client_weighting=client_weighting,
        full_gradient_aggregator=full_gradient_aggregator,
        metrics_aggregator=metrics_aggregator,
        loop_implementation=loop_implementation,
    )
  else:
    raise TypeError(
        'When `model_fn` is not a callable, it must be an instance'
        ' of tff.learning.models.FunctionalModel. Instead got a: '
        f'{type(model_fn)}'
    )
  finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
      server_optimizer, model_weights_type
  )
  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      model_aggregator,
      finalizer,
  )


def build_unweighted_mime_lite(
    model_fn: Callable[[], variable.VariableModel],
    base_optimizer: optimizer_base.Optimizer,
    server_optimizer: optimizer_base.Optimizer = sgdm.build_sgdm(1.0),
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.UnweightedAggregationFactory] = None,
    full_gradient_aggregator: Optional[
        factory.UnweightedAggregationFactory
    ] = None,
    metrics_aggregator: types.MetricsAggregatorType = metric_aggregator.sum_then_finalize,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> learning_process.LearningProcess:
  """Builds a learning process that performs Mime Lite.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  Mime Lite algorithm on client models. The iterative process has the following
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
  training is performed using `optimizer`, where its state is communicated by
  the server, and kept intact during local training. The state is updated only
  at the server based on the full gradient evaluated by the clients based on the
  current server model state. The client full gradients are aggregated by
  unweighted `full_gradient_aggregator`. Each client computes the difference
  between the client model after training and its initial model. These model
  deltas are then aggregated by unweighted `model_aggregator`. The aggregate
  model delta is added to the existing server model state.

  The Mime Lite algorithm is based on the paper
  "Breaking the centralized barrier for cross-device federated learning."
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf

  Note that Keras optimizers are not supported. This is due to the Mime Lite
  algorithm applying the optimizer without changing it state at clients
  (optimizer's `tf.Variable`s in the case of Keras), which is not possible with
  Keras optimizers without reaching into private implementation details and
  incurring additional computation and memory cost at clients.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    base_optimizer: A `tff.learning.optimizers.Optimizer` which will be used for
      both creating and updating a global optimizer state, as well as
      optimization at clients given the global state, which is fixed during the
      optimization.
    server_optimizer: A `tff.learning.optimizers.Optimizer` which will be used
      for applying the aggregate model update to the global model weights.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.UnweightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.UnweightedMeanFactory`.
    full_gradient_aggregator: An optional
      `tff.aggregators.UnweightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
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
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.UnweightedMeanFactory()
  py_typecheck.check_type(
      full_gradient_aggregator, factory.UnweightedAggregationFactory
  )

  return build_weighted_mime_lite(
      model_fn=model_fn,
      base_optimizer=base_optimizer,
      server_optimizer=server_optimizer,
      client_weighting=client_weight_lib.ClientWeighting.UNIFORM,
      model_distributor=model_distributor,
      model_aggregator=factory_utils.as_weighted_aggregator(model_aggregator),
      full_gradient_aggregator=factory_utils.as_weighted_aggregator(
          full_gradient_aggregator
      ),
      metrics_aggregator=metrics_aggregator,
      loop_implementation=loop_implementation,
  )


def build_mime_lite_with_optimizer_schedule(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    learning_rate_fn: Callable[[int], float],
    base_optimizer: optimizer_base.Optimizer,
    server_optimizer: optimizer_base.Optimizer = sgdm.build_sgdm(1.0),
    client_weighting: Optional[
        client_weight_lib.ClientWeighting
    ] = client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory
    ] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> learning_process.LearningProcess:
  """Builds a learning process for Mime Lite with optimizer scheduling.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  Mime Lite algorithm on client models. The iterative process has the following
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
  training is performed using `optimizer`, where its state is communicated by
  the server, and kept intact during local training. The state is updated only
  at the server based on the full gradient evaluated by the clients based on the
  current server model state. The client full gradients are aggregated by
  weighted `full_gradient_aggregator`. Each client computes the difference
  between the client model after training and its initial model. These model
  deltas are then aggregated by weighted `model_aggregator`. Both of the
  aggregations are weighted, according to `client_weighting`. The aggregate
  model delta is added to the existing server model state.

  The Mime Lite algorithm is based on the paper
  "Breaking the centralized barrier for cross-device federated learning."
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf

  Note that Keras optimizers are not supported. This is due to the Mime Lite
  algorithm applying the optimizer without changing it state at clients
  (optimizer's `tf.Variable`s in the case of Keras), which is not possible with
  Keras optimizers without reaching into private implementation details and
  incurring additional computation and memory cost at clients.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    learning_rate_fn: A callable accepting an integer round number and returning
      a float to be used as a learning rate for the optimizer.
      `learning_rate_fn` must be serializable by Tensorflow (e.g. via
      `tf.function`).
    base_optimizer: A `tff.learning.optimizers.Optimizer` which will be used for
      both creating and updating a global optimizer state, as well as
      optimization at clients given the global state, which is fixed during the
      optimization.
    server_optimizer: A `tff.learning.optimizers.Optimizer` which will be used
      for applying the aggregate model update to the global model weights.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method. By default, weighting by number of examples
      is used.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
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
  """
  py_typecheck.check_type(base_optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(server_optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  if not callable(model_fn):
    if not isinstance(model_fn, functional.FunctionalModel):
      raise TypeError(
          'If `model_fn` is not a callable, it must be an instance of '
          f'tff.learning.models.FunctionalModel. Got {type(model_fn)}'
      )

    @tensorflow_computation.tf_computation
    def initial_model_weights_fn():
      trainable_weights, non_trainable_weights = model_fn.initial_weights
      return model_weights_lib.ModelWeights(
          tuple(tf.convert_to_tensor(w) for w in trainable_weights),
          tuple(tf.convert_to_tensor(w) for w in non_trainable_weights),
      )

  else:

    @tensorflow_computation.tf_computation
    def initial_model_weights_fn():
      model = model_fn()  # pytype: disable=not-callable
      if not isinstance(model, variable.VariableModel):
        raise TypeError(
            'When `model_fn` is a callable, it returns instances of'
            ' tff.learning.models.VariableModel. Instead callable returned'
            f' type: {type(model)}'
        )
      return model_weights_lib.ModelWeights.from_model(model)

  model_weights_type = initial_model_weights_fn.type_signature.result
  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)
  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)
  model_update_type = model_weights_type.trainable
  model_aggregator = model_aggregator.create(
      model_update_type, computation_types.TensorType(np.float32)
  )
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.MeanFactory()
  py_typecheck.check_type(
      full_gradient_aggregator, factory.WeightedAggregationFactory
  )

  client_work = _build_scheduled_mime_lite_client_work(
      model_fn=model_fn,
      learning_rate_fn=learning_rate_fn,
      optimizer=base_optimizer,
      client_weighting=client_weighting,
      full_gradient_aggregator=full_gradient_aggregator,
      metrics_aggregator=metrics_aggregator,
      loop_implementation=loop_implementation,
  )
  finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
      server_optimizer, model_weights_type
  )
  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      model_aggregator,
      finalizer,
  )
