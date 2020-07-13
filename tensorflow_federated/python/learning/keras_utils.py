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

import collections
from typing import Sequence, Union

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils


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
    weights_to_assign = model_utils.ModelWeights.from_tff_result(tff_weights)
  else:
    weights_to_assign = tff_weights
  weights_to_assign.assign_weights_to(keras_model)


def from_keras_model(keras_model,
                     loss,
                     input_spec,
                     loss_weights=None,
                     metrics=None):
  """Builds a `tff.learning.Model` for a given input type.

  `from_keras_model` validates its arguments, normalizes them as appropriate and
  instantiates a `tff.learning.Model` backed by `keras_model` for the forward
  pass and autodifferentiation steps. This function needs three pieces of
  information in order to accomplish this goal: a `tf.keras.Model` to use for
  its forward pass; a loss function (or group of loss functions) `loss`; and a
  way to infer the TFF type signatures for the `tff.Computation` in which this
  model will appear, the `input_spec`.

  Notice that since TFF couples the `tf.keras.Model` and
  `loss`, TFF needs a slightly different notion of "fully specified type" than
  pure Keras does. That is, the model `M` takes inputs of type `x` and
  produces predictions of type `p`; the loss function `L` takes inputs of type
  `<p, y>` and produces a scalar. Therefore in order to fully specify the type
  signatures for computations in which the generated `tff.learning.Model` will
  appear, TFF needs the type `y` in addition to the type `x`.

  Args:
    keras_model: A `tf.keras.Model` object that is not compiled.
    loss: A `tf.keras.losses.Loss` that takes two batched tensor parameters,
      `y_true` and `y_pred`, and returns the loss. If the model has multiple
      outputs, you can  use a different loss on each output by passing a
      dictionary or a list of losses. The loss value that will be minimized
      by the model will then be the sum of all individual losses, each weighted
      by `loss_weights`.
    input_spec: A value convertible to `tff.Type` specifying the type
      of arguments the model expects. Notice this must be a compound structure
      of two elements, specifying both the data fed into the model to generate
      predictions, as its first element, as well as the expected type of the
      ground truth as its second.
    loss_weights: (Optional) a list or dictionary specifying scalar coefficients
      (Python floats) to weight the loss contributions of different model
      outputs. The loss value that will be minimized by the model will then be
      the *weighted sum* of all individual losses, weighted by the
      `loss_weights` coefficients. If a list, it is expected to have a 1:1
        mapping to the model's outputs. If a tensor, it is expected to map
        output names (strings) to scalar coefficients.
    metrics: (Optional) a list of `tf.keras.metrics.Metric` objects.

  Returns:
    A `tff.learning.Model` object.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: If `keras_model` was compiled, or , or `input_spec` does not
      contain two elements.
    KeyError: If `loss` is a `dict` and does not have the same keys as
      `keras_model.outputs`.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  py_typecheck.check_type(loss, (tf.keras.losses.Loss, collections.Sequence))
  if len(input_spec) != 2:
    raise ValueError('The top-level structure in `input_spec` must contain '
                     'exactly two elements, as it must specify type '
                     'information for both inputs to and predictions from the '
                     'model. You passed input spec {}.'.format(input_spec))
  if loss_weights is not None:
    py_typecheck.check_type(loss, collections.Sequence)
  if isinstance(loss, collections.Sequence):
    if len(loss) != len(keras_model.outputs):
      raise ValueError('`keras_model` must have equal number of '
                       'outputs and losses.\nloss: {}\nof length: {}.'
                       '\noutputs: {}\nof length: {}.'.format(
                           loss, len(loss), keras_model.outputs,
                           len(keras_model.outputs)))
    if loss_weights is not None and len(loss) != len(loss_weights):
      raise ValueError('`keras_model` must have equal number of '
                       'losses and loss_weights.\nloss: {}\nof length: {}.'
                       '\nloss_weights: {}\nof length: {}.'.format(
                           loss, len(loss), loss_weights, len(loss_weights)))
    for loss_fn in loss:
      py_typecheck.check_type(loss_fn, tf.keras.losses.Loss)

  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled')

  if isinstance(loss, collections.Sequence):
    loss_functions = loss
  else:
    loss_functions = [loss]

  return model_utils.enhance(
      _KerasModel(
          keras_model,
          input_spec=input_spec,
          loss_fns=loss_functions,
          loss_weights=loss_weights,
          metrics=metrics))


def federated_aggregate_keras_metric(
    metrics: Union[tf.keras.metrics.Metric,
                   Sequence[tf.keras.metrics.Metric]], federated_values):
  """Aggregates variables a keras metric placed at CLIENTS to SERVER.

  Args:
    metrics: a single `tf.keras.metrics.Metric` or a `Sequence` of metrics . The
      order must match the order of variables in `federated_values`.
    federated_values: a single federated value, or a `Sequence` of federated
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

  @tff.tf_computation
  def zeros_fn():
    # `member_type` is a (potentially nested) `tff.NamedTupleType`, which is an
    # `anonymous_tuple.AnonymousTuple`.
    return anonymous_tuple.map_structure(
        lambda v: tf.zeros(v.shape, dtype=v.dtype), member_types)

  zeros = zeros_fn()

  @tff.tf_computation(member_types, member_types)
  def accumulate(accumulators, variables):
    return tf.nest.map_structure(tf.add, accumulators, variables)

  @tff.tf_computation(member_types, member_types)
  def merge(a, b):
    return tf.nest.map_structure(tf.add, a, b)

  @tff.tf_computation(member_types)
  def report(accumulators):
    """Insert `accumulators` back into the keras metric to obtain result."""

    def finalize_metric(metric: tf.keras.metrics.Metric, values):
      # Note: the following call requires that `type(metric)` have a no argument
      # __init__ method, which will restrict the types of metrics that can be
      # used. This is somewhat limiting, but the pattern to use default
      # arguments and export the values in `get_config()` (see
      # `tf.keras.metrics.TopKCategoricalAccuracy`) works well.
      keras_metric = None
      try:
        # This is some trickery to reconstruct a metric object in the current
        # scope, so that the `tf.Variable`s get created when we desire.
        keras_metric = type(metric).from_config(metric.get_config())
      except TypeError as e:
        # Re-raise the error with a more helpful message, but the previous stack
        # trace.
        raise TypeError(
            'Caught exception trying to call `{t}.from_config()` with '
            'config {c}. Confirm that {t}.__init__() has an argument for '
            'each member of the config.\nException: {e}'.format(
                t=type(metric), c=metric.config(), e=e))

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

  return tff.federated_aggregate(federated_values, zeros, accumulate, merge,
                                 report)


class _KerasModel(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self,
               inner_model,
               input_spec,
               loss_fns,
               loss_weights=None,
               metrics=None):
    self._input_spec = input_spec

    if not loss_fns:
      raise ValueError(
          'Must specify at least one loss_fns, got: {l}'.format(l=loss_fns))
    if (len(tf.nest.flatten(loss_fns)) != len(
        tf.nest.flatten(inner_model.output))):
      raise ValueError('Must specify the same number of loss_fns as model '
                       'outputs.\nloss_fns: {l}\nmodel outputs: {o}'.format(
                           l=loss_fns, o=inner_model.output))
    self._loss_fns = loss_fns

    if loss_weights is None:
      loss_weights = [1.0] * len(loss_fns)
    else:
      py_typecheck.check_type(loss_weights, collections.Sequence)
      if len(loss_weights) != len(loss_fns):
        raise ValueError('Must specify the same number of '
                         'loss_weights (got {llw}) as loss_fns (got {llf}).\n'
                         'loss_weights: {lw}\nloss_fns: {lf}'.format(
                             lw=loss_weights,
                             llw=len(loss_weights),
                             lf=loss_fns,
                             llf=len(loss_fns)))
    self._loss_weights = loss_weights
    self._keras_model = inner_model
    self._metrics = metrics if metrics is not None else []

    # This is defined here so that it closes over the `loss_fn`.
    class _WeightedMeanLossMetric(tf.keras.metrics.Mean):
      """A `tf.keras.metrics.Metric` wrapper for the loss function."""

      def __init__(self, name='loss', dtype=tf.float32):
        super().__init__(name, dtype)
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

        return super().update_state(batch_loss, batch_size)

    self._loss_metric = _WeightedMeanLossMetric()

    metric_variable_type_dict = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, self.report_local_outputs())
    federated_local_outputs_type = tff.FederatedType(metric_variable_type_dict,
                                                     tff.CLIENTS)

    def federated_output(local_outputs):
      return federated_aggregate_keras_metric(self.get_metrics(), local_outputs)

    self._federated_output_computation = tff.federated_computation(
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
    if not self._keras_model._is_compiled:  # pylint: disable=protected-access
      return self._metrics + [self._loss_metric]
    else:
      return self._keras_model.metrics + [self._loss_metric]

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
    predictions = self._keras_model(inputs, training=training)

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
