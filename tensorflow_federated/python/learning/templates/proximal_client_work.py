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
"""An implementation of model delta `ClientWork` template.

The term 'model delta' refers to difference between the model weights at the
start and the end of local training. This is for used for example in the
implementation of the generalized FedAvg algorithm implemented in
`tff.learning.algorithms.build_weighted_fed_avg`.
"""

import collections
from collections.abc import Callable
from typing import Any, Optional, Union

from absl import logging
import tensorflow as tf

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
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.tensorflow_libs import tensor_utils


# TODO(b/213433744): Make this method private.
def build_model_delta_update_with_tff_optimizer(
    model_fn: Callable[[], variable.VariableModel],
    *,
    weighting: client_weight_lib.ClientWeighting,
    delta_l2_regularizer: float,
    use_experimental_simulation_loop: bool = False,
):
  """Creates client update logic in FedProx using a TFF optimizer.

  In contrast to using a `tf.keras.optimizers.Optimizer`, we avoid creating
  `tf.Variable`s associated with the optimizer state within the scope of the
  client work, as they are not necessary. This also means that the client's
  model weights are updated by computing `optimizer.next` and then assigning
  the result to the model weights (while a `tf.keras.optimizers.Optimizer` will
  modify the model weight in place using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg callable returning a `tff.learning.Model`.
    weighting: A `tff.learning.ClientWeighting` value.
    delta_l2_regularizer: A nonnegative float, L2 regularization strength of the
      model delta.
    use_experimental_simulation_loop: Controls the reduce loop function for the
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tf.function`.
  """
  model = model_fn()
  dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
      use_experimental_simulation_loop
  )

  @tf.function
  def client_update(optimizer, initial_weights, data):
    model_weights = model_weights_lib.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda a, b: a.assign(b), model_weights, initial_weights
    )

    def reduce_fn(state, batch):
      """Trains a `tff.learning.Model` on a batch of data."""
      num_examples_sum, optimizer_state = state
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      if delta_l2_regularizer > 0.0:
        proximal_term = tf.nest.map_structure(
            lambda x, y: delta_l2_regularizer * (y - x),
            model_weights.trainable,
            initial_weights.trainable,
        )
        gradients = tf.nest.map_structure(tf.add, gradients, proximal_term)
      optimizer_state, updated_weights = optimizer.next(
          optimizer_state,
          tuple(tf.nest.flatten(model_weights.trainable)),
          tuple(tf.nest.flatten(gradients)),
      )
      updated_weights = tf.nest.pack_sequence_as(
          model_weights.trainable, updated_weights
      )
      tf.nest.map_structure(
          lambda a, b: a.assign(b), model_weights.trainable, updated_weights
      )

      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum, optimizer_state

    def initial_state_for_reduce_fn():
      # TODO(b/161529310): We flatten and convert the trainable specs to tuple,
      # as "for batch in data:" pattern would try to stack the tensors in list.
      trainable_tensor_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype),
          tuple(tf.nest.flatten(model_weights.trainable)),
      )
      return (
          tf.zeros(shape=[], dtype=tf.int64),
          optimizer.initialize(trainable_tensor_specs),
      )

    num_examples, _ = dataset_reduce_fn(
        reduce_fn, data, initial_state_for_reduce_fn
    )
    client_update = tf.nest.map_structure(
        tf.subtract, initial_weights.trainable, model_weights.trainable
    )
    model_output = model.report_local_unfinalized_metrics()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    client_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_update)
    )
    client_weight = _choose_client_weight(
        weighting, has_non_finite_delta, num_examples
    )

    return (
        client_works.ClientResult(
            update=client_update, update_weight=client_weight
        ),
        model_output,
    )

  return client_update


# TODO(b/213433744): Make this method private.
def build_model_delta_update_with_keras_optimizer(
    model_fn,
    weighting,
    delta_l2_regularizer,
    use_experimental_simulation_loop: bool = False,
):
  """Creates client update logic in FedProx using a `tf.keras` optimizer.

  In contrast to using a `tff.learning.optimizers.Optimizer`, we have to
  maintain `tf.Variable`s associated with the optimizer state within the scope
  of the client work. Additionally, the client model weights are modified in
  place by using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg callable returning a `tff.learning.Model`.
    weighting: A `tff.learning.ClientWeighting` value.
    delta_l2_regularizer: A nonnegative float, L2 regularization strength of the
      model delta.
    use_experimental_simulation_loop: Controls the reduce loop function for the
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tf.function`.
  """
  model = model_fn()
  dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
      use_experimental_simulation_loop
  )

  @tf.function
  def client_update(optimizer, initial_weights, data):
    model_weights = model_weights_lib.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda a, b: a.assign(b), model_weights, initial_weights
    )

    def reduce_fn(num_examples_sum, batch):
      """Trains a `tff.learning.Model` on a batch of data."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      if delta_l2_regularizer > 0.0:
        proximal_term = tf.nest.map_structure(
            lambda x, y: delta_l2_regularizer * (y - x),
            model_weights.trainable,
            initial_weights.trainable,
        )
        gradients = tf.nest.map_structure(tf.add, gradients, proximal_term)
      grads_and_vars = zip(gradients, model_weights.trainable)
      optimizer.apply_gradients(grads_and_vars)

      # TODO(b/199782787): Add a unit test for a model that does not compute
      # `num_examples` in its forward pass.
      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum

    def initial_state_for_reduce_fn():
      return tf.zeros(shape=[], dtype=tf.int64)

    num_examples = dataset_reduce_fn(
        reduce_fn, data, initial_state_for_reduce_fn
    )
    client_update = tf.nest.map_structure(
        tf.subtract, initial_weights.trainable, model_weights.trainable
    )
    model_output = model.report_local_unfinalized_metrics()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    client_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_update)
    )
    client_weight = _choose_client_weight(
        weighting, has_non_finite_delta, num_examples
    )
    return (
        client_works.ClientResult(
            update=client_update, update_weight=client_weight
        ),
        model_output,
    )

  return client_update


def _build_functional_model_delta_update(
    *,
    model: functional.FunctionalModel,
    weighting: client_weight_lib.ClientWeighting,
    delta_l2_regularizer: float,
) -> Callable[[Any, Any, Any], tuple[client_works.ClientResult, Any]]:
  """Creates client update logic in FedProx for a FunctionalModel.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    weighting: A `tff.learning.ClientWeighting` value.
    delta_l2_regularizer: A nonnegative float, L2 regularization strength of the
      model delta.

  Returns:
    A `tf.function`.
  """
  dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
      simulation_flag=True
  )

  @tf.function
  def client_update(optimizer, initial_weights, data):
    # Switch to the tuple expected by FunctionalModel.
    initial_trainable_weights, initial_non_trainable_weights = initial_weights
    # In the case that `dataset_reduce_fn` is an iterator style loop, autograph
    # requires defining the tensors used across the loop body (such as
    # `trainable_weights`) before the loop, hence we create `trainable_weights`
    # here.
    trainable_weights = initial_trainable_weights

    def reduce_fn(state, batch):
      """Trains a `tff.learning.Model` on a batch of data."""
      num_examples_sum, model_weights, metrics_state, optimizer_state = state
      trainable_weights, non_trainable_weights = model_weights
      with tf.GradientTape() as tape:
        tape.watch(trainable_weights)
        output = model.forward_pass(model_weights, batch, training=True)
      gradients = tape.gradient(output.loss, trainable_weights)
      if delta_l2_regularizer > 0.0:
        proximal_term = tf.nest.map_structure(
            lambda x, y: delta_l2_regularizer * (y - x),
            trainable_weights,
            initial_trainable_weights,
        )
        gradients = tf.nest.map_structure(tf.add, gradients, proximal_term)
      optimizer_state, updated_weights = optimizer.next(
          optimizer_state, tuple(trainable_weights), tuple(gradients)
      )
      trainable_weights = tf.nest.pack_sequence_as(
          trainable_weights, updated_weights
      )
      if isinstance(batch, collections.abc.Mapping):
        labels = batch['y']
      else:
        _, labels = batch
      metrics_state = model.update_metrics_state(
          metrics_state, batch_output=output, labels=labels
      )
      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)
      return (
          num_examples_sum,
          (trainable_weights, non_trainable_weights),
          metrics_state,
          optimizer_state,
      )

    def initial_state_for_reduce_fn():
      trainable_tensor_specs = tuple(
          tf.TensorSpec(v.shape, v.dtype)
          for v in tf.nest.flatten(trainable_weights)
      )
      return (
          tf.zeros(shape=[], dtype=tf.int64),
          (initial_trainable_weights, initial_non_trainable_weights),
          model.initialize_metrics_state(),
          optimizer.initialize(trainable_tensor_specs),
      )

    num_examples, model_weights, metrics_state, _ = dataset_reduce_fn(
        reduce_fn, data, initial_state_for_reduce_fn
    )
    trainable_weights, _ = model_weights
    client_update = tf.nest.map_structure(
        tf.subtract, initial_trainable_weights, trainable_weights
    )
    client_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_update)
    )
    client_weight = _choose_client_weight(
        weighting, has_non_finite_delta, num_examples
    )

    unfinalized_metrics = metrics_state
    return (
        client_works.ClientResult(
            update=client_update, update_weight=client_weight
        ),
        unfinalized_metrics,
    )

  return client_update


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


def build_model_delta_client_work(
    model_fn: Callable[[], variable.VariableModel],
    optimizer: Union[
        optimizer_base.Optimizer, Callable[[], tf.keras.optimizers.Optimizer]
    ],
    client_weighting: client_weight_lib.ClientWeighting,
    delta_l2_regularizer: float,
    metrics_aggregator: Optional[
        Callable[
            [
                types.MetricFinalizersType,
                computation_types.StructWithPythonType,
            ],
            computation_base.Computation,
        ]
    ] = None,
    *,
    use_experimental_simulation_loop: bool = False,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for the FedProx algorithm.

  This client work is constructed in slightly different manners depending on
  whether `optimizer` is a `tff.learning.optimizers.Optimizer`, or a no-arg
  callable returning a `tf.keras.optimizers.Optimizer`.

  If it is a `tff.learning.optimizers.Optimizer`, we avoid creating
  `tf.Variable`s associated with the optimizer state within the scope of the
  client work, as they are not necessary. This also means that the client's
  model weights are updated by computing `optimizer.next` and then assigning
  the result to the model weights (while a `tf.keras.optimizers.Optimizer` will
  modify the model weight in place using `optimizer.apply_gradients`).

  In contrast to FedAvg, this client work employs L2 regularization towards the
  server model, the strength of which is governed by the `delta_l2_regularizer`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    optimizer: A `tff.learning.optimizers.Optimizer`, or a no-arg callable that
      returns a `tf.keras.Optimizer`.
    client_weighting:  A `tff.learning.ClientWeighting` value.
    delta_l2_regularizer: A nonnegative float representing the parameter of the
      L2-regularization term applied to the delta from initial model weights
      during training. Values larger than 0.0 prevent clients from moving too
      far from the server model during local training.
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
    A `ClientWorkProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  py_typecheck.check_type(delta_l2_regularizer, float)
  if delta_l2_regularizer < 0.0:
    raise ValueError(
        'Provided delta_l2_regularizer must be nonnegative,'
        f'but found: {delta_l2_regularizer}'
    )
  elif delta_l2_regularizer == 0.0:
    logging.warning(
        'delta_l2_regularizer is set to 0.0, which means no '
        'regularization will occur.'
    )
  if not (
      isinstance(optimizer, optimizer_base.Optimizer) or callable(optimizer)
  ):
    raise TypeError(
        'Provided optimizer must a either a tff.learning.optimizers.Optimizer '
        'or a no-arg callable returning an tf.keras.optimizers.Optimizer.'
    )

  if metrics_aggregator is None:
    metrics_aggregator = aggregator.sum_then_finalize

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        model.report_local_unfinalized_metrics()
    )
    metrics_aggregation_fn = metrics_aggregator(
        model.metric_finalizers(), unfinalized_metrics_type
    )
  data_type = computation_types.SequenceType(model.input_spec)
  weights_type = model_weights_lib.weights_type_from_model(model)

  if isinstance(optimizer, optimizer_base.Optimizer):

    @tensorflow_computation.tf_computation(weights_type, data_type)
    def client_update_computation(initial_model_weights, dataset):
      client_update = build_model_delta_update_with_tff_optimizer(
          model_fn=model_fn,
          weighting=client_weighting,
          delta_l2_regularizer=delta_l2_regularizer,
          use_experimental_simulation_loop=use_experimental_simulation_loop,
      )
      return client_update(optimizer, initial_model_weights, dataset)

  else:

    @tensorflow_computation.tf_computation(weights_type, data_type)
    def client_update_computation(initial_model_weights, dataset):
      keras_optimizer = optimizer()
      client_update = build_model_delta_update_with_keras_optimizer(
          model_fn=model_fn,
          weighting=client_weighting,
          delta_l2_regularizer=delta_l2_regularizer,
          use_experimental_simulation_loop=use_experimental_simulation_loop,
      )
      return client_update(keras_optimizer, initial_model_weights, dataset)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type),
  )
  def next_fn(state, weights, client_data):
    client_result, model_outputs = intrinsics.federated_map(
        client_update_computation, (weights, client_data)
    )
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def build_functional_model_delta_client_work(
    *,
    model: functional.FunctionalModel,
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    delta_l2_regularizer: float,
    metrics_aggregator: Optional[
        Callable[
            [
                types.MetricFinalizersType,
                computation_types.StructWithPythonType,
            ],
            computation_base.Computation,
        ]
    ] = None,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for the FedProx algorithm.

  Args:
    model: A `tff.learning.models.FunctionalModel` to train.
    optimizer: A `tff.learning.optimizers.Optimizer`.
    client_weighting:  A `tff.learning.ClientWeighting` value.
    delta_l2_regularizer: A nonnegative float representing the parameter of the
      L2-regularization term applied to the delta from initial model weights
      during training. Values larger than 0.0 prevent clients from moving too
      far from the server model during local training.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics. If
      `None`, this is set to `tff.learning.metrics.sum_then_finalize`.

  Returns:
    A `ClientWorkProcess`.
  """
  py_typecheck.check_type(model, functional.FunctionalModel)
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  py_typecheck.check_type(delta_l2_regularizer, float)
  if delta_l2_regularizer < 0.0:
    raise ValueError(
        'Provided delta_l2_regularizer must be nonnegative,'
        f'but found: {delta_l2_regularizer}'
    )
  elif delta_l2_regularizer == 0.0:
    logging.warning(
        'delta_l2_regularizer is set to 0.0, which means no '
        'regularization will occur.'
    )
  if not (
      isinstance(optimizer, optimizer_base.Optimizer) or callable(optimizer)
  ):
    raise TypeError(
        'Provided optimizer must a either a tff.learning.optimizers.Optimizer '
        'or a no-arg callable returning an tf.keras.optimizers.Optimizer.'
    )

  if metrics_aggregator is None:
    metrics_aggregator = aggregator.sum_then_finalize
  data_type = computation_types.SequenceType(model.input_spec)

  def ndarray_to_tensorspec(ndarray):
    return tf.TensorSpec(shape=ndarray.shape, dtype=ndarray.dtype)

  # Wrap in a `ModelWeights` structure that is required by the `finalizer.`
  weights_type = model_weights_lib.ModelWeights(
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[0]),
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[1]),
  )

  @tensorflow_computation.tf_computation(weights_type, data_type)
  def client_update_computation(initial_model_weights, dataset):
    # Tuple the model weights in the format matching
    # FunctionalModel.initial_weights.
    initial_model_weights = (
        initial_model_weights.trainable,
        initial_model_weights.non_trainable,
    )
    client_update = _build_functional_model_delta_update(
        model=model,
        weighting=client_weighting,
        delta_l2_regularizer=delta_l2_regularizer,
    )
    return client_update(optimizer, initial_model_weights, dataset)

  @federated_computation.federated_computation
  def init_fn():
    empty_state = ()
    return intrinsics.federated_value(empty_state, placements.SERVER)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type),
  )
  def next_fn(state, weights, client_data):
    client_result, unfinalized_metrics = intrinsics.federated_map(
        client_update_computation, (weights, client_data)
    )
    metrics_aggregation_fn = metrics_aggregator(
        model.finalize_metrics, unfinalized_metrics.type_signature.member
    )
    train_metrics = metrics_aggregation_fn(unfinalized_metrics)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)
