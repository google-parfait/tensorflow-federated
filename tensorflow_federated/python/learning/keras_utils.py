# Lint as: python3
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
"""Utility methods for working with Keras in TensorFlow Federated."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import graph_keys


def assign_weights_to_keras_model(keras_model, tff_weights):
  """Assigns a nested structure of TFF weights to a Keras model.

  This function may be used to retrieve the model parameters trained by the
  federated averaging process for use in an existing `tf.keras.models.Model`,
  e.g.:

  ```
  keras_model = tf.keras.models.Model(inputs=..., outputs=...)

  def model_fn():
    return tff.learning.from_keras_model(keras_model)

  fed_avg = tff.learning.build_federated_averaging_process(model_fn, ...)
  state = fed_avg.initialize()
  state = fed_avg.next(state, ...)
  ...
  tff.learning.assign_weights_to_keras_model(keras_model, state.model)
  ```

  Args:
    keras_model: A `tf.keras.models.Model` instance to assign weights to.
    tff_weights: A TFF value representing the weights of a model.

  Raises:
    TypeError: if `tff_weights` is not a TFF value, or `keras_model` is not a
      `tf.keras.models.Model` instance.
  """
  # TODO(b/123092620): Simplify this.
  py_typecheck.check_type(
      tff_weights, (anonymous_tuple.AnonymousTuple, model_utils.ModelWeights))
  py_typecheck.check_type(keras_model, tf.keras.models.Model)
  if isinstance(tff_weights, anonymous_tuple.AnonymousTuple):
    weights_to_assign = model_utils.ModelWeights.from_tff_value(tff_weights)
  else:
    weights_to_assign = tff_weights
  weights_to_assign.assign_weights_to(keras_model)


def _preprocess_dummy_batch(dummy_batch):
  """Converts a batch (a nested structure of Python objects) to tensors."""
  dummy_tensors = tf.nest.map_structure(tf.convert_to_tensor, dummy_batch)
  if isinstance(dummy_tensors, (list, tuple, collections.OrderedDict)):
    return dummy_tensors
  elif py_typecheck.is_named_tuple(dummy_tensors):
    return dummy_tensors._asdict()
  elif isinstance(dummy_tensors, dict):
    raise TypeError('Called with argument of type `dict`, '
                    'change to supported `collections.OrderedDict` type.')
  else:
    raise NotImplementedError(
        'No implementation for dummy batch of type {!s}'.format(
            type(dummy_batch)))


def from_keras_model(keras_model,
                     dummy_batch,
                     loss,
                     loss_weights=None,
                     metrics=None,
                     optimizer=None):
  """Builds a `tff.learning.Model` for an example mini batch.

  Args:
    keras_model: A `tf.keras.Model` object that is not compiled.
    dummy_batch: A nested structure of values that are convertible to *batched*
      tensors with the same shapes and types as would be input to `keras_model`.
      The values of the tensors are not important and can be filled with any
      reasonable input value.
    loss: A callable that takes two batched tensor parameters, `y_true` and
      `y_pred`, and returns the loss. If the model has multiple outputs, you can
      use a different loss on each output by passing a dictionary or a list of
      losses. The loss value that will be minimized by the model will then be
      the sum of all individual losses, each weighted by `loss_weights`.
    loss_weights: (Optional) a list or dictionary specifying scalar coefficients
      (Python floats) to weight the loss contributions of different model
      outputs. The loss value that will be minimized by the model will then be
      the *weighted sum* of all individual losses, weighted by the
      `loss_weights` coefficients. If a list, it is expected to have a 1:1
        mapping to the model's outputs. If a tensor, it is expected to map
        output names (strings) to scalar coefficients.
    metrics: (Optional) a list of `tf.keras.metrics.Metric` objects.
    optimizer: (Optional) a `tf.keras.optimizer.Optimizer`. If None, returned
      model cannot be used for training.

  Returns:
    A `tff.learning.Model` object.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: If `keras_model` was compiled.
    KeyError: If `loss` is a `dict` and does not have the same keys as
      `keras_model.outputs`.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  py_typecheck.check_type(
      loss, (tf.keras.losses.Loss, collections.Sequence, collections.Mapping))

  if loss_weights is not None:
    py_typecheck.check_type(loss, (collections.Sequence, collections.Mapping))

  if isinstance(loss, (collections.Mapping, collections.Sequence)):
    if len(loss) != len(keras_model.outputs):
      raise ValueError('`keras_model` must have equal number of '
                       'outputs and losses.\nloss: {}\noutputs: {}'.format(
                           loss, keras_model.outputs))
    if loss_weights is not None and len(loss) != len(loss_weights):
      raise ValueError(
          '`keras_model` must have equal number of '
          'losses and loss_weights.\nloss: {} \nloss_weights:{}'.format(
              loss, loss_weights))

  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled. Use '
                     'from_compiled_keras_model() instead.')

  dummy_tensors = _preprocess_dummy_batch(dummy_batch)
  if optimizer is None:
    if isinstance(loss, collections.Mapping):
      loss_functions = []
      for name in keras_model.output_names:
        if name not in loss:
          raise KeyError('Output missing from loss dictionary'
                         '\nlosses: {}\noutputs: {}'.format(
                             list(loss.keys()), keras_model.output_names))
        loss_functions.append(loss[name])
    elif isinstance(loss, collections.Sequence):
      loss_functions = loss
    else:
      loss_functions = [loss]

    return model_utils.enhance(
        _KerasModel(keras_model, dummy_tensors, loss_functions, loss_weights,
                    metrics))

  keras_model.compile(
      loss=loss,
      optimizer=optimizer,
      loss_weights=loss_weights,
      metrics=metrics)
  # NOTE: A sub-classed tf.keras.Model does not produce the compiled metrics
  # until the model has been called on input. The work-around is to call
  # Model.test_on_batch() once before asking for metrics.
  if isinstance(dummy_tensors, collections.Mapping):
    keras_model.test_on_batch(**dummy_tensors)
  else:
    keras_model.test_on_batch(*dummy_tensors)
  return model_utils.enhance(_TrainableKerasModel(keras_model, dummy_tensors))


def from_compiled_keras_model(keras_model, dummy_batch):
  """Builds a `tff.learning.Model` for an example mini batch.

  Args:
    keras_model: A `tf.keras.Model` object that was compiled.
    dummy_batch: A nested structure of values that are convertible to *batched*
      tensors with the same shapes and types as expected by `forward_pass()`.
      The values of the tensors are not important and can be filled with any
      reasonable input value.

  Returns:
    A `tff.learning.Model`.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: If `keras_model` was *not* compiled.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  # Optimizer attribute is only set after calling tf.keras.Model.compile().
  if not keras_model.optimizer:
    raise ValueError('`keras_model` must be compiled. Use from_keras_model() '
                     'instead.')
  dummy_tensors = _preprocess_dummy_batch(dummy_batch)
  # NOTE: A sub-classed tf.keras.Model does not produce the compiled metrics
  # until the model has been called on input. The work-around is to call
  # Model.test_on_batch() once before asking for metrics.
  if isinstance(dummy_tensors, collections.Mapping):
    keras_model.test_on_batch(**dummy_tensors)
  else:
    keras_model.test_on_batch(*dummy_tensors)
  return model_utils.enhance(_TrainableKerasModel(keras_model, dummy_tensors))


def federated_aggregate_keras_metric(metric_type, metric_config,
                                     federated_variables):
  """Aggregates variables a keras metric placed at CLIENTS to SERVER.

  Args:
    metric_type: a type object (type must inherit from
      `tf.keras.metrics.Metric`).
    metric_config: the result of calling `get_config()` on a metric object, used
      with `metric_type.from_config()` to locally construct a new metric object.
    federated_variables: a federated value place on clients that is the value
      returned by `tf.keras.metrics.Metric.variables`.

  Returns:
    The result of calling `result()` on a `tf.keras.metrics.Metric` of type
  `metric_type`, after aggregation all CLIENTS places `variables`.
  """
  member_type = federated_variables.type_signature.member

  @tff.tf_computation
  def zeros_fn():
    # `member_type` is a (potentially nested) `tff.NamedTupleType`, which is an
    # `anonymous_tuple.AnonymousTuple`.
    return anonymous_tuple.map_structure(
        lambda v: tf.zeros(v.shape, dtype=v.dtype), member_type)

  zeros = zeros_fn()

  # TODO(b/123995628): as of 2019-02-01 all variables created in a
  # `tf.keras.metrics.Metric` use the argument
  # `aggregation=tf.VariableAggregation.SUM`, hence below only uses `tf.add`.
  # This may change in the future (and the `tf.Variable.aggregation` property
  # will be exposed in a future TF version). Need to handle non-SUM variables.

  @tff.tf_computation(member_type, member_type)
  def accumulate(accumulators, variables):
    return tf.nest.map_structure(tf.add, accumulators, variables)

  @tff.tf_computation(member_type, member_type)
  def merge(a, b):
    return tf.nest.map_structure(tf.add, a, b)

  @tff.tf_computation(member_type)
  def report(accumulators):
    """Insert `accumulators` back into the keras metric to obtain result."""
    # NOTE: the following call requires that `metric_type` have a no argument
    # __init__ method, which will restrict the types of metrics that can be
    # used. This is somewhat limiting, but the pattern to use default arguments
    # and export the values in `get_config()` (see
    # `tf.keras.metrics.TopKCategoricalAccuracy`) works well.
    keras_metric = None
    try:
      keras_metric = metric_type.from_config(metric_config)
    except TypeError as e:
      # Re-raise the error with a more helpful message, but the previous stack
      # trace.
      raise TypeError(
          'Caught exception trying to call `{t}.from_config()` with '
          'config {c}. Confirm that {t}.__init__() has an argument for '
          'each member of the config.\nException: {e}'.format(
              t=metric_type, c=metric_config, e=e))

    assignments = []
    for v, a in zip(keras_metric.variables, accumulators):
      assignments.append(v.assign(a))
    with tf.control_dependencies(assignments):
      return keras_metric.result()

  return tff.federated_aggregate(federated_variables, zeros, accumulate, merge,
                                 report)


class _KerasModel(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self,
               inner_model,
               dummy_batch,
               loss_fns,
               loss_weights=None,
               metrics=None):

    # NOTE: sub-classed `tf.keras.Model`s do not have fully initialized
    # variables until they are called on input. We forced that here.
    if isinstance(dummy_batch, collections.Mapping):
      inner_model(dummy_batch['x'])
    else:
      inner_model(dummy_batch[0])

    def _tensor_spec_with_undefined_batch_dim(tensor):
      # Remove the batch dimension and leave it unspecified.
      spec = tf.TensorSpec(
          shape=[None] + tensor.shape.dims[1:], dtype=tensor.dtype)
      return spec

    self._input_spec = tf.nest.map_structure(
        _tensor_spec_with_undefined_batch_dim, dummy_batch)

    self._keras_model = inner_model
    self._loss_fns = loss_fns

    if isinstance(loss_weights, collections.Mapping):
      self._loss_weights = []
      for name in inner_model.output_names:
        if name not in loss_weights:
          raise KeyError('Output missing from loss_weights dictionary'
                         '\nloss_weights: {}\noutputs: {}'.format(
                             list(loss_weights.keys()),
                             inner_model.output_names))
        else:
          self._loss_weights.append(loss_weights[name])
    else:
      if loss_weights is None:
        self._loss_weights = [1.0 for _ in range(len(loss_fns))]
      else:
        self._loss_weights = loss_weights

    loss_weights = self._loss_weights
    self._metrics = metrics if metrics is not None else []

    # This is defined here so that it closes over the `loss_fn`.
    class _WeightedMeanLossMetric(tf.keras.metrics.Mean):
      """A `tf.keras.metrics.Metric` wrapper for the loss function."""

      def __init__(self, name='loss', dtype=tf.float32):
        super(_WeightedMeanLossMetric, self).__init__(name, dtype)
        self._loss_fns = loss_fns
        self._loss_weights = loss_weights

      def update_state(self, y_true, y_pred, sample_weight=None):
        if len(self._loss_fns) == 1:
          batch_size = tf.cast(tf.shape(y_pred)[0], self._dtype)
          y_true = tf.cast(y_true, self._dtype)
          y_pred = tf.cast(y_pred, self._dtype)
          batch_loss = self._loss_fns[0](y_true, y_pred)

        else:
          batch_loss = tf.zeros(())
          for i in range(len(self._loss_fns)):
            y_t = tf.cast(y_true[i], self._dtype)
            y_p = tf.cast(y_pred[i], self._dtype)
            batch_loss += self._loss_weights[i] * self._loss_fns[i](y_t, y_p)

          batch_size = tf.cast(tf.shape(y_pred[0])[0], self._dtype)

        return super(_WeightedMeanLossMetric,
                     self).update_state(batch_loss, batch_size)

    class _TrainingTimeHistory(tf.keras.metrics.Sum):

      def update_state(self, y_true, y_pred, sample_weight=None):
        pass

      def log_time(self, time_value):
        return super(_TrainingTimeHistory, self).update_state(values=time_value)

    self._loss_metric = _WeightedMeanLossMetric()
    self._training_timing = _TrainingTimeHistory(name='training_time_sec')

    metric_variable_type_dict = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, self.report_local_outputs())
    federated_local_outputs_type = tff.FederatedType(metric_variable_type_dict,
                                                     tff.CLIENTS)

    def federated_output(local_outputs):
      results = collections.OrderedDict()
      for metric, variables in zip(self.get_metrics(), local_outputs):
        results[metric.name] = federated_aggregate_keras_metric(
            type(metric), metric.get_config(), variables)
      return results

    self._federated_output_computation = tff.federated_computation(
        federated_output, federated_local_outputs_type)

    # Keras creates variables that are not added to any collection, making it
    # impossible for TFF to extract them and create the appropriate initializer
    # before call a tff.Computation. Here we store them in a TFF specific
    # collection so that they can be retrieved later.
    # TODO(b/122081673): this likely goes away in TF2.0
    for variable in itertools.chain(self.trainable_variables,
                                    self.non_trainable_variables,
                                    self.local_variables):
      tf.compat.v1.add_to_collection(
          graph_keys.GraphKeys.VARS_FOR_TFF_TO_INITIALIZE, variable)

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
    if not self._keras_model._is_compiled:  # pylint: disable=protected-access
      return self._metrics + [self._loss_metric, self._training_timing]
    else:
      return self._keras_model.metrics + [
          self._loss_metric, self._training_timing
      ]

  @property
  def input_spec(self):
    return self._input_spec

  def _forward_pass(self, batch_input, training=True):
    if hasattr(batch_input, '_asdict'):
      batch_input = batch_input._asdict()
    if isinstance(batch_input, collections.Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     'Instead have keys {}'.format(list(batch_input.keys())))

    predictions = self._keras_model(inputs=inputs, training=training)

    if isinstance(batch_input, collections.Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]
    if y_true is not None:
      if len(self._loss_fns) == 1:
        loss_fn = self._loss_fns[0]
        batch_loss = loss_fn(y_true=y_true, y_pred=predictions)

      else:
        batch_loss = tf.zeros(())
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
    return model_lib.BatchOutput(
        loss=batch_loss,
        predictions=predictions,
        num_examples=tf.shape(tf.nest.flatten(inputs)[0])[0])

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

  @classmethod
  def make_batch(cls, x, y):
    return cls.Batch(x=x, y=y)


class _TrainableKerasModel(_KerasModel, model_lib.TrainableModel):
  """Wrapper class for `tf.keras.Model`s that can be trained."""

  def __init__(self, inner_model, dummy_batch):
    super(_TrainableKerasModel,
          self).__init__(inner_model, dummy_batch, inner_model.loss_functions,
                         inner_model.loss_weights, inner_model.metrics)

  @property
  def local_variables(self):
    return (super(_TrainableKerasModel, self).local_variables +
            self._keras_model.optimizer.variables())

  @tf.function
  def train_on_batch(self, batch_input):
    train_start = tf.timestamp()
    batch_output = self._forward_pass(batch_input)
    _ = self._keras_model.optimizer.get_updates(
        loss=batch_output.loss, params=self.trainable_variables)
    train_end = tf.timestamp()
    self._training_timing.log_time(train_end - train_start)
    return batch_output
