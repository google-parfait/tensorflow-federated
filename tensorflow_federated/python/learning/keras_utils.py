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
"""Utility methods for working with Keras in TensorFlow Federated."""

import collections
from typing import Callable, List, Optional, OrderedDict, Sequence, Union
import warnings

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning.metrics import finalizer

Loss = Union[tf.keras.losses.Loss, List[tf.keras.losses.Loss]]


# TODO(b/197746608): Remove the code path that takes in constructed Keras
# metrics, because reconstructing metrics via `from_config` can cause problems.
def from_keras_model(
    keras_model: tf.keras.Model,
    loss: Loss,
    input_spec,
    loss_weights: Optional[List[float]] = None,
    metrics: Optional[Union[List[tf.keras.metrics.Metric],
                            List[Callable[[], tf.keras.metrics.Metric]]]] = None
) -> model_lib.Model:
  """Builds a `tff.learning.Model` from a `tf.keras.Model`.

  The `tff.learning.Model` returned by this function uses `keras_model` for
  its forward pass and autodifferentiation steps.

  Notice that since TFF couples the `tf.keras.Model` and `loss`,
  TFF needs a slightly different notion of "fully specified type" than
  pure Keras does. That is, the model `M` takes inputs of type `x` and
  produces predictions of type `p`; the loss function `L` takes inputs of type
  `<p, y>` (where `y` is the ground truth label type) and produces a scalar.
  Therefore in order to fully specify the type signatures for computations in
  which the generated `tff.learning.Model` will appear, TFF needs the type `y`
  in addition to the type `x`.

  Note: This function does not currently accept subclassed `tf.keras.Models`,
  as it makes assumptions about presence of certain attributes which are
  guaranteed to exist through the functional or Sequential API but are
  not necessarily present for subclassed models.

  Note: This function raises a UserWarning if the `tf.keras.Model` contains a
  BatchNormalization layer, as the batch mean and variance will be treated as
  non-trainable variables and won't be updated during the training (see
  b/186845846 for more information). Consider using Group Normalization instead.

  Args:
    keras_model: A `tf.keras.Model` object that is not compiled.
    loss: A single `tf.keras.losses.Loss` or a list of losses-per-output. If a
      single loss is provided, then all model output (as well as all prediction
      information) is passed to the loss; this includes situations of multiple
      model outputs and/or predictions. If multiple losses are provided as a
      list, then each loss is expected to correspond to a model output; the
      model will attempt to minimize the sum of all individual losses
      (optionally weighted using the `loss_weights` argument).
    input_spec: A structure of `tf.TensorSpec`s or `tff.Type` specifying the
      type of arguments the model expects. If `input_spec` is a `tff.Type`, its
      leaf nodes must be `TensorType`s. Note that `input_spec` must be a
      compound structure of two elements, specifying both the data fed into the
      model (x) to generate predictions as well as the expected type of the
      ground truth (y). If provided as a list, it must be in the order [x, y].
      If provided as a dictionary, the keys must explicitly be named `'{}'` and
      `'{}'`.
    loss_weights: (Optional) A list of Python floats used to weight the loss
      contribution of each model output (when providing a list of losses for the
      `loss` argument).
    metrics: (Optional) a list of `tf.keras.metrics.Metric` objects or a list of
      no-arg callables that each constructs a `tf.keras.metrics.Metric`.

  Returns:
    A `tff.learning.Model` object.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`, if
      `loss` is not an instance of `tf.keras.losses.Loss` nor a list of
      instances of `tf.keras.losses.Loss`, if `input_spec` is a `tff.Type` but
      the leaf nodes are not `tff.TensorType`s, if `loss_weight` is provided but
      is not a list of floats, or if `metrics` is provided but is not a list of
      instances of `tf.keras.metrics.Metric`.
    ValueError: If `keras_model` was compiled, if `loss` is a list of unequal
      length to the number of outputs of `keras_model`, if `loss_weights` is
      specified but `loss` is not a list, if `input_spec` does not contain
      exactly two elements, or if `input_spec` is a dictionary and does not
      contain keys `'x'` and `'y'`.
  """.format(model_lib.MODEL_ARG_NAME, model_lib.MODEL_LABEL_NAME)
  # Validate `keras_model`
  py_typecheck.check_type(keras_model, tf.keras.Model)
  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled')

  # Validate and normalize `loss` and `loss_weights`
  if not isinstance(loss, list):
    py_typecheck.check_type(loss, tf.keras.losses.Loss)
    if loss_weights is not None:
      raise ValueError('`loss_weights` cannot be used if `loss` is not a list.')
    loss = [loss]
    loss_weights = [1.0]
  else:
    if len(loss) != len(keras_model.outputs):
      raise ValueError('If a loss list is provided, `keras_model` must have '
                       'equal number of outputs to the losses.\nloss: {}\nof '
                       'length: {}.\noutputs: {}\nof length: {}.'.format(
                           loss, len(loss), keras_model.outputs,
                           len(keras_model.outputs)))
    for loss_fn in loss:
      py_typecheck.check_type(loss_fn, tf.keras.losses.Loss)

    if loss_weights is None:
      loss_weights = [1.0] * len(loss)
    else:
      if len(loss) != len(loss_weights):
        raise ValueError(
            '`keras_model` must have equal number of losses and loss_weights.'
            '\nloss: {}\nof length: {}.'
            '\nloss_weights: {}\nof length: {}.'.format(loss, len(loss),
                                                        loss_weights,
                                                        len(loss_weights)))
      for loss_weight in loss_weights:
        py_typecheck.check_type(loss_weight, float)

  if len(input_spec) != 2:
    raise ValueError('The top-level structure in `input_spec` must contain '
                     'exactly two top-level elements, as it must specify type '
                     'information for both inputs to and predictions from the '
                     'model. You passed input spec {}.'.format(input_spec))
  if isinstance(input_spec, computation_types.Type):
    if not type_analysis.is_structure_of_tensors(input_spec):
      raise TypeError(
          'Expected a `tff.Type` with all the leaf nodes being '
          '`tff.TensorType`s, found an input spec {}.'.format(input_spec))
    input_spec = type_conversions.structure_from_tensor_type_tree(
        lambda tensor_type: tf.TensorSpec(tensor_type.shape, tensor_type.dtype),
        input_spec)
  else:
    tensor_spec = (tf.TensorSpec, tf.RaggedTensorSpec)
    tf.nest.map_structure(
        lambda s: py_typecheck.check_type(s, tensor_spec, 'input spec member'),
        input_spec)
  if isinstance(input_spec, collections.abc.Mapping):
    if model_lib.MODEL_ARG_NAME not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'{}\'`, representing the input(s) '
          'to the Keras model.'.format(model_lib.MODEL_ARG_NAME))
    if model_lib.MODEL_LABEL_NAME not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'{}\'`, representing the label(s) '
          'to be used in the Keras loss(es).'.format(
              model_lib.MODEL_LABEL_NAME))

  if metrics is None:
    metrics = []
  else:
    py_typecheck.check_type(metrics, list)

  for layer in keras_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      warnings.warn(
          'Batch Normalization contains non-trainable variables that won\'t be '
          'updated during the training. Consider using Group Normalization '
          'instead.', UserWarning)
      break

  return _KerasModel(
      keras_model,
      input_spec=input_spec,
      loss_fns=loss,
      loss_weights=loss_weights,
      metrics=metrics)


def federated_aggregate_keras_metric(
    metrics: Union[tf.keras.metrics.Metric, Sequence[tf.keras.metrics.Metric],
                   Callable[[], tf.keras.metrics.Metric],
                   Sequence[Callable[[], tf.keras.metrics.Metric]]],
    federated_values):
  """Aggregates variables a keras metric placed at CLIENTS to SERVER.

  Args:
    metrics: A single or a `Sequence` of `tf.keras.metrics.Metric` objects, or
      a single or a `Sequence` of no-arg callables that each constructs a
      `tf.keras.metrics.Metric`. The order must match the order of variables in
      `federated_values`.
    federated_values: A single federated value, or a `Sequence` of federated
      values. The values must all have `tff.CLIENTS` placement. If value is a
      `Sequence` type, it must match the order of the sequence in `metrics.

  Returns:
    The result of performing a federated sum on federated_values, then assigning
    the aggregated values into the variables of the corresponding
    `tf.keras.metrics.Metric` and calling `tf.keras.metrics.Metric.result`. The
    resulting structure has `tff.SERVER` placement.
  """
  member_types = tf.nest.map_structure(lambda t: t.type_signature.member,
                                       federated_values)

  @computations.tf_computation
  def zeros_fn():
    # `member_type` is a (potentially nested) `tff.StructType`, which is an
    # `structure.Struct`.
    return structure.map_structure(lambda v: tf.zeros(v.shape, dtype=v.dtype),
                                   member_types)

  zeros = zeros_fn()

  @computations.tf_computation(member_types, member_types)
  def accumulate(accumulators, variables):
    return tf.nest.map_structure(tf.add, accumulators, variables)

  @computations.tf_computation(member_types, member_types)
  def merge(a, b):
    return tf.nest.map_structure(tf.add, a, b)

  @computations.tf_computation(member_types)
  def report(accumulators):
    """Insert `accumulators` back into the keras metric to obtain result."""

    def finalize_metric(metric: Union[tf.keras.metrics.Metric,
                                      Callable[[], tf.keras.metrics.Metric]],
                        values):
      # Note: if the input metric is an instance of `tf.keras.metrics.Metric`,
      # the following call requires that `type(metric)` have a no argument
      # __init__ method, which will restrict the types of metrics that can be
      # used. This is somewhat limiting, but the pattern to use default
      # arguments and export the values in `get_config()` (see
      # `tf.keras.metrics.TopKCategoricalAccuracy`) works well.
      #
      # If type(metric) is subclass of another tf.keras.metric arguments passed
      # to __init__ must include arguments expected by the superclass and
      # specified in superclass get_config().
      keras_metric = finalizer.create_keras_metric(metric)

      assignments = []
      for v, a in zip(keras_metric.variables, values):
        assignments.append(v.assign(a))
      with tf.control_dependencies(assignments):
        return keras_metric.result()

    if isinstance(metrics, tf.keras.metrics.Metric):
      # Only a single metric to aggregate.
      return finalize_metric(metrics, accumulators)
    else:
      # Otherwise map over all the metrics.
      return collections.OrderedDict([
          (name, finalize_metric(metric, values))
          for metric, (name, values) in zip(metrics, accumulators.items())
      ])

  return intrinsics.federated_aggregate(federated_values, zeros, accumulate,
                                        merge, report)


class _KerasModel(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self, keras_model: tf.keras.Model, input_spec,
               loss_fns: List[tf.keras.losses.Loss], loss_weights: List[float],
               metrics: Union[List[tf.keras.metrics.Metric],
                              List[Callable[[], tf.keras.metrics.Metric]]]):
    self._keras_model = keras_model
    self._input_spec = input_spec
    self._loss_fns = loss_fns
    self._loss_weights = loss_weights

    self._metrics = []
    self._metric_constructors = []
    if metrics:
      has_keras_metric = False
      has_keras_metric_constructor = False

      for metric in metrics:
        if isinstance(metric, tf.keras.metrics.Metric):
          self._metrics.append(metric)
          has_keras_metric = True
        elif callable(metric):
          constructed_metric = metric()
          if not isinstance(constructed_metric, tf.keras.metrics.Metric):
            raise TypeError(
                f'Metric constructor {metric} is not a no-arg callable that '
                'creates a `tf.keras.metrics.Metric`.')
          self._metric_constructors.append(metric)
          self._metrics.append(constructed_metric)
          has_keras_metric_constructor = True
        else:
          raise TypeError(
              'Expected the input metric to be either a '
              '`tf.keras.metrics.Metric` or a no-arg callable that constructs '
              'a `tf.keras.metrics.Metric`, found a non-callable '
              f'{py_typecheck.type_string(type(metric))}.')

      if has_keras_metric and has_keras_metric_constructor:
        raise TypeError(
            'Expected the input `metrics` to be either a list of '
            '`tf.keras.metrics.Metric` objects or a list of no-arg callables '
            'that each constructs a `tf.keras.metrics.Metric`, '
            f'found both types in the `metrics`: {metrics}.')

    # This is defined here so that it closes over the `loss_fn`.
    class _WeightedMeanLossMetric(tf.keras.metrics.Mean):
      """A `tf.keras.metrics.Metric` wrapper for the loss function."""

      def __init__(self, name='loss', dtype=tf.float32):
        super().__init__(name, dtype)
        self._loss_fns = loss_fns
        self._loss_weights = loss_weights

      def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_pred, list):
          batch_size = tf.shape(y_pred[0])[0]
        else:
          batch_size = tf.shape(y_pred)[0]

        if len(self._loss_fns) == 1:
          batch_loss = self._loss_fns[0](y_true, y_pred)
        else:
          batch_loss = tf.zeros(())
          for i in range(len(self._loss_fns)):
            batch_loss += self._loss_weights[i] * self._loss_fns[i](y_true[i],
                                                                    y_pred[i])

        return super().update_state(batch_loss, batch_size)

    self._metrics.append(_WeightedMeanLossMetric())
    if not metrics or self._metric_constructors:
      self._metric_constructors.append(_WeightedMeanLossMetric)

    metric_variable_type_dict = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, self.report_local_outputs())
    federated_local_outputs_type = computation_types.FederatedType(
        metric_variable_type_dict, placements.CLIENTS)

    def federated_output(local_outputs):
      if self._metric_constructors:
        return federated_aggregate_keras_metric(self._metric_constructors,
                                                local_outputs)
      return federated_aggregate_keras_metric(self.get_metrics(), local_outputs)

    self._federated_output_computation = computations.federated_computation(
        federated_output, federated_local_outputs_type)

  @property
  def trainable_variables(self):
    return self._keras_model.trainable_variables

  @property
  def non_trainable_variables(self):
    return self._keras_model.non_trainable_variables

  @property
  def local_variables(self):
    local_variables = []
    for metric in self.get_metrics():
      local_variables.extend(metric.variables)
    return local_variables

  def get_metrics(self):
    return self._metrics

  @property
  def input_spec(self):
    return self._input_spec

  @tf.function
  def predict_on_batch(self, x, training=True):
    return self._keras_model(x, training=training)

  def _forward_pass(self, batch_input, training=True):
    if isinstance(batch_input, collections.abc.Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     f'Instead have keys {list(batch_input.keys())}')
    predictions = self.predict_on_batch(inputs, training)

    if isinstance(batch_input, collections.abc.Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]
    if y_true is not None:
      if len(self._loss_fns) == 1:
        loss_fn = self._loss_fns[0]
        # Note: we add each of the per-layer regularization losses to the loss
        # that we use to update trainable parameters, in addition to the
        # user-provided loss function. Keras does the same in the
        # `tf.keras.Model` training step. This is expected to have no effect if
        # no per-layer losses are added to the model.
        batch_loss = tf.add_n([loss_fn(y_true=y_true, y_pred=predictions)] +
                              self._keras_model.losses)

      else:
        # Note: we add each of the per-layer regularization losses to the losses
        # that we use to update trainable parameters, in addition to the
        # user-provided loss functions. Keras does the same in the
        # `tf.keras.Model` training step. This is expected to have no effect if
        # no per-layer losses are added to the model.
        batch_loss = tf.add_n([tf.zeros(())] + self._keras_model.losses)
        for i in range(len(self._loss_fns)):
          loss_fn = self._loss_fns[i]
          loss_wt = self._loss_weights[i]
          batch_loss += loss_wt * loss_fn(
              y_true=y_true[i], y_pred=predictions[i])
    else:
      batch_loss = None

    # TODO(b/145308951): Follow up here to pass through sample_weight in the
    # case that we have a model supporting masking.
    for metric in self.get_metrics():
      metric.update_state(y_true=y_true, y_pred=predictions)

    def nrows(t):
      return t.nrows() if isinstance(t, tf.RaggedTensor) else tf.shape(t)[0]

    return model_lib.BatchOutput(
        loss=batch_loss,
        predictions=predictions,
        num_examples=nrows(tf.nest.flatten(inputs)[0]))

  @tf.function
  def forward_pass(self, batch_input, training=True):
    return self._forward_pass(batch_input, training=training)

  @tf.function
  def report_local_outputs(self):
    """Reports the variables of the metrics tracked during local training.

    Returns:
      A `collections.OrderedDict` of metric name keys to lists of metric
      variables.
    """
    outputs = collections.OrderedDict()
    for metric in self.get_metrics():
      outputs[metric.name] = [v.read_value() for v in metric.variables]
    return outputs

  @property
  def federated_output_computation(self):
    return self._federated_output_computation

  @tf.function
  def report_local_unfinalized_metrics(
      self) -> OrderedDict[str, List[tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to unfinalized values.

    Returns:
      An `OrderedDict` of metric names to lists of unfinalized metric values.
      For a Keras metric, its unfinalized values are the tensor values of its
      variables tracked during local training. The returned `OrderedDict` has
      the same keys (metric names) as the `OrderedDict` returned by the method
      `metric_finalizers()`, and can be used as input to the finalizers to get
      the finalized metric values. This method and the `metric_finalizers()`
      method can be used to construct a cross-client metrics aggregator when
      defining the federated training processes or evaluation computations.
    """
    outputs = collections.OrderedDict()
    for metric in self.get_metrics():
      outputs[metric.name] = [v.read_value() for v in metric.variables]
    return outputs

  def metric_finalizers(
      self) -> OrderedDict[str, finalizer.KerasMetricFinalizer]:
    """Creates an `OrderedDict` of metric names to finalizers.

    Returns:
      An `OrderedDict` of metric names to finalizers. A finalizer of a Keras
      metric is a `tf.function` decorated callable that takes in this metric's
      unfinalized values (created by `report_local_unfinalized_metrics`), and
      returns the metric value computed by `tf.keras.metrics.Metric.result()`.
      This method and the `report_local_unfinalized_metrics` method can be used
      to construct a cross-client metrics aggregator when defining the federated
      training processes or evaluation computations.
    """
    finalizers = collections.OrderedDict()
    for metric in self.get_metrics():
      finalizers[metric.name] = finalizer.create_keras_metric_finalizer(metric)
    return finalizers
