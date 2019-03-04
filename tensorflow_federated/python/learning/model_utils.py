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
"""Utility methods for working with TensorFlow Federated Model objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import six
from six.moves import zip
import tensorflow as tf

# TODO(b/123578208): Remove deep keras imports after updating TF version.
from tensorflow.python.keras import metrics as keras_metrics
from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.tensorflow_libs import graph_keys
from tensorflow_federated.python.tensorflow_libs import tensor_utils

nest = tf.contrib.framework.nest


def model_initializer(model, name=None):
  """Creates an initializer op for all of the model's variables."""
  py_typecheck.check_type(model, model_lib.Model)
  return tf.variables_initializer(
      model.trainable_variables + model.non_trainable_variables +
      model.local_variables,
      name=(name or 'model_initializer'))


class ModelWeights(
    collections.namedtuple(
        'ModelWeightsBase',
        [
            # An OrderedDict of `Model.trainable_variables` keyed by name.
            'trainable',
            # An OrderedDict of `Model.non_trainable_variables` keyed by name.
            'non_trainable'
        ])):
  """A container for the trainable and non-trainable variables of a `Model`.

  Note this does not include the model's local variables.

  It may also be used to hold other values that are parallel to these variables,
  e.g., tensors corresponding to variable values, or updates to model variables.
  """

  # Necessary to work around for problematic _asdict() returning empty
  # dictionary between Python 3.4.2 and 3.4.5.
  #
  # Addtionally prevents __dict__ from being created, which can improve memory
  # usage of ModelWeights object.
  __slots__ = ()

  def __new__(cls, trainable, non_trainable):
    return super(ModelWeights, cls).__new__(
        cls, tensor_utils.to_odict(trainable),
        tensor_utils.to_odict(non_trainable))

  @classmethod
  def from_model(cls, model):
    py_typecheck.check_type(model, (model_lib.Model, tf.keras.Model))
    # N.B. to_var_dict preserves the order of the variables, which
    # is critical so we can re-use the list of values e.g. when doing
    # keras_model.set_weights
    return cls(
        tensor_utils.to_var_dict(model.trainable_variables),
        tensor_utils.to_var_dict(model.non_trainable_variables))

  @classmethod
  def from_tff_value(cls, anon_tuple):
    py_typecheck.check_type(anon_tuple, anonymous_tuple.AnonymousTuple)
    return cls(
        anonymous_tuple.to_odict(anon_tuple.trainable),
        anonymous_tuple.to_odict(anon_tuple.non_trainable))

  @property
  def keras_weights(self):
    """Returns a list of weights in the same order as `tf.keras.Model.weights`.

    (Assuming that this ModelWeights object corresponds to the weights of
    a keras model).
    """
    return list(self.trainable.values()) + list(self.non_trainable.values())


def keras_weights_from_tff_weights(tff_weights):
  """Converts TFF's nested weights structure to flat weights.

  This function may be used, for example, to retrieve the model parameters
  trained by the federated averaging process for use in an existing
  `keras` model, e.g.:

  ```
  fed_avg = tff.learning.build_federated_averaging_process(...)
  state = fed_avg.initialize()
  state = fed_avg.next(state, ...)
  ...
  keras_model.set_weights(
      tff.learning.keras_weights_from_tff_weights(state.model))
  ```

  Args:
    tff_weights: A TFF value representing the weights of a model.

  Returns:
    A list of tensors suitable for passing to `tf.keras.Model.set_weights`.
  """
  # TODO(b/123092620): Simplify this.
  py_typecheck.check_type(tff_weights,
                          (anonymous_tuple.AnonymousTuple, ModelWeights))
  if isinstance(tff_weights, anonymous_tuple.AnonymousTuple):
    return list(tff_weights.trainable) + list(tff_weights.non_trainable)
  else:
    return tff_weights.keras_weights


def from_keras_model(keras_model,
                     dummy_batch,
                     loss,
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
      `y_pred`, and returns the loss.
    metrics: (Optional) a list of `tf.keras.metrics.Metric` objects.
    optimizer: (Optional) a `tf.keras.optimizer.Optimizer`. If None, returned
      model cannot be used for training.

  Returns:
    A `tff.learning.Model` object.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: If `keras_model` was compiled.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled. Use '
                     'from_compiled_keras_model() instead.')
  if optimizer is None:
    return enhance(_KerasModel(keras_model, dummy_batch, loss, metrics))
  keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  return enhance(_TrainableKerasModel(keras_model, dummy_batch))


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
  if not hasattr(keras_model, 'optimizer'):
    raise ValueError('`keras_model` must be compiled. Use from_keras_model() '
                     'instead.')
  return enhance(_TrainableKerasModel(keras_model, dummy_batch))


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
    return anonymous_tuple.map_structure(tf.add, accumulators, variables)

  @tff.tf_computation(member_type, member_type)
  def merge(a, b):
    return anonymous_tuple.map_structure(tf.add, a, b)

  @tff.tf_computation(member_type)
  def report(accumulators):
    """Insert `accumulators` back into the kera metric to obtain result."""
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
          'Caught expection trying to call `{t}.from_config()` with '
          'config {c}. Confirm that {t}.__init__() has an argument for '
          'each member of the config.\nException: {e}'.format(
              t=metric_type, c=metric_config, e=e))

    assignments = []
    for v, a in zip(keras_metric.variables, accumulators):
      assignments.append(tf.assign(v, a))
    with tf.control_dependencies(assignments):
      return keras_metric.result()

  return tff.federated_aggregate(federated_variables, zeros, accumulate, merge,
                                 report)


class _KerasModel(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self, inner_model, dummy_batch, loss_func, metrics):
    # TODO(b/124477598): the following set_session() should be removed in the
    # future. This is a workaround for Keras' caching sessions in a way that
    # isn't compatible with TFF. This is already fixed in TF master, but not as
    # of v1.13.1.
    #
    # We do not use .clear_session() because it blows away the graph stack by
    # resetting the default graph.
    tf.keras.backend.set_session(None)

    if hasattr(dummy_batch, '_asdict'):
      dummy_batch = dummy_batch._asdict()
    # Convert input to tensors, possibly from nested lists that need to be
    # converted to a single top-level tensor.
    dummy_tensors = collections.OrderedDict(
        [(k, tf.convert_to_tensor_or_sparse_tensor(v))
         for k, v in six.iteritems(dummy_batch)])
    # NOTE: sub-classed `tf.keras.Model`s do not have fully initialized
    # variables until they are called on input. We forced that here.
    inner_model(dummy_tensors['x'])

    def _tensor_spec_with_undefined_batch_dim(tensor):
      # Remove the batch dimension and leave it unspecified.
      spec = tf.TensorSpec(
          shape=[None] + tensor.shape.dims[1:], dtype=tensor.dtype)
      return spec

    self._input_spec = nest.map_structure(_tensor_spec_with_undefined_batch_dim,
                                          dummy_tensors)

    self._keras_model = inner_model
    self._loss_fn = loss_func
    self._metrics = metrics if metrics is not None else []

    # This is defined here so that it closes over the `loss_func`.
    class _WeightedMeanLossMetric(keras_metrics.Metric):
      """A `tf.keras.metrics.Metric` wrapper for the loss function."""

      def __init__(self, name='loss', dtype=tf.float32):
        super(_WeightedMeanLossMetric, self).__init__(name, dtype)
        self._total_loss = self.add_weight('total_loss', initializer='zeros')
        self._total_weight = self.add_weight(
            'total_weight', initializer='zeros')
        self._loss_fn = loss_func

      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # _loss_fn is expected to return the scalar mean loss, so we multiply by
        # the batch_size to get back to total loss.
        batch_size = tf.cast(tf.shape(y_pred)[0], self._dtype)
        batch_total_loss = self._loss_fn(y_true, y_pred) * batch_size

        op = self._total_loss.assign_add(batch_total_loss)
        with tf.control_dependencies([op]):
          return self._total_weight.assign_add(batch_size)

      def result(self):
        return tf.div_no_nan(self._total_loss, self._total_weight)

    self._loss_metric = _WeightedMeanLossMetric()

    # Keras creates variables that are not added to any collection, making it
    # impossible for TFF to extract them and create the appropriate initializer
    # before call a tff.Computation. Here we store them in a TFF specific
    # collection so that they can be retrieved later.
    # TODO(b/122081673): this likely goes away in TF2.0
    for variable in itertools.chain(self.trainable_variables,
                                    self.non_trainable_variables,
                                    self.local_variables):
      tf.add_to_collection(graph_keys.GraphKeys.VARS_FOR_TFF_TO_INITIALIZE,
                           variable)

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
    # forward_pass requires batch_input be a dictionary that can be passed to
    # tf.keras.Model.__call__, namely it has keys `x`, and optionally `y`.
    if hasattr(batch_input, '_asdict'):
      batch_input = batch_input._asdict()

    inputs = batch_input.get('x')
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     'Instead have keys {}'.format(batch_input.keys()))
    predictions = self._keras_model(inputs=inputs, training=training)

    y_true = batch_input.get('y')
    if y_true is not None:
      batch_loss = self._loss_fn(y_true=y_true, y_pred=predictions)
      for metric in self.get_metrics():
        metric.update_state(y_true=y_true, y_pred=predictions)
    else:
      batch_loss = None

    return model_lib.BatchOutput(loss=batch_loss, predictions=predictions)

  @tf.contrib.eager.function(autograph=False)
  def forward_pass(self, batch_input, training=True):
    return self._forward_pass(batch_input, training=training)

  @tf.contrib.eager.function(autograph=False)
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
    metric_variable_type_dict = nest.map_structure(tf.TensorSpec.from_tensor,
                                                   self.report_local_outputs())
    federated_local_outputs_type = tff.FederatedType(
        metric_variable_type_dict, tff.CLIENTS, all_equal=False)

    @tff.federated_computation(federated_local_outputs_type)
    def federated_output(local_outputs):
      results = collections.OrderedDict()
      for metric, variables in zip(self.get_metrics(), local_outputs):
        results[metric.name] = federated_aggregate_keras_metric(
            type(metric), metric.get_config(), variables)
      return results

    return federated_output

  @classmethod
  def make_batch(cls, x, y):
    return cls.Batch(x=x, y=y)


class _TrainableKerasModel(_KerasModel, model_lib.TrainableModel):
  """Wrapper class for `tf.keras.Model`s that can be trained."""

  def __init__(self, inner_model, dummy_batch):
    if hasattr(dummy_batch, '_asdict'):
      dummy_batch = dummy_batch._asdict()
    # NOTE: A sub-classed tf.keras.Model does not produce the compiled metrics
    # until the model has been called on input. The work-around is to call
    # Model.test_on_batch() once before asking for metrics.
    inner_model.test_on_batch(**dummy_batch)
    # This must occur after test_on_batch()
    if len(inner_model.loss_functions) != 1:
      raise NotImplementedError('only a single loss functions is supported')
    super(_TrainableKerasModel,
          self).__init__(inner_model, dummy_batch,
                         inner_model.loss_functions[0], inner_model.metrics)

  @property
  def non_trainable_variables(self):
    return (super(_TrainableKerasModel, self).non_trainable_variables +
            self._keras_model.optimizer.variables())

  @tf.contrib.eager.function(autograph=False)
  def train_on_batch(self, batch_input):
    batch_output = self._forward_pass(batch_input)
    _ = self._keras_model.optimizer.get_updates(
        loss=batch_output.loss, params=self.trainable_variables)
    return batch_output


def enhance(model):
  """Wraps a `tff.learning.Model` as an `EnhancedModel`.

  Args:
    model: A `tff.learning.Model`.

  Returns:
    An `EnhancedModel` or `TrainableEnhancedModel`, depending on the type of the
    input model. If `model` has already been wrapped as such, this is a no-op.
  """
  py_typecheck.check_type(model, model_lib.Model)
  if isinstance(model, EnhancedModel):
    return model

  if isinstance(model, model_lib.TrainableModel):
    return EnhancedTrainableModel(model)
  else:
    return EnhancedModel(model)


def _check_iterable_of_variables(variables):
  py_typecheck.check_type(variables, collections.Iterable)
  for v in variables:
    py_typecheck.check_type(v, tf.Variable)
  return variables


class EnhancedModel(model_lib.Model):
  """A wrapper around a Model that adds sanity checking and metadata helpers."""

  def __init__(self, model):
    super(EnhancedModel, self).__init__()
    py_typecheck.check_type(model, model_lib.Model)
    if isinstance(model, EnhancedModel):
      raise ValueError(
          'Attempting to wrap an EnhancedModel in another EnhancedModel')
    self._model = model

  #
  # Methods offering additional functionality and metadata:
  #

  @property
  def weights(self):
    """Returns a `tff.learning.ModelWeights`."""
    return ModelWeights.from_model(self)

  #
  # The following delegate to the Model interface:
  #

  @property
  def trainable_variables(self):
    return _check_iterable_of_variables(self._model.trainable_variables)

  @property
  def non_trainable_variables(self):
    return _check_iterable_of_variables(self._model.non_trainable_variables)

  @property
  def local_variables(self):
    return _check_iterable_of_variables(self._model.local_variables)

  @property
  def input_spec(self):
    return self._model.input_spec

  def forward_pass(self, batch_input, training=True):
    return py_typecheck.check_type(
        self._model.forward_pass(batch_input, training), model_lib.BatchOutput)

  def report_local_outputs(self):
    return self._model.report_local_outputs()

  @property
  def federated_output_computation(self):
    return self._model.federated_output_computation


class EnhancedTrainableModel(EnhancedModel, model_lib.TrainableModel):

  def __init__(self, model):
    py_typecheck.check_type(model, model_lib.TrainableModel)
    super(EnhancedTrainableModel, self).__init__(model)

  def train_on_batch(self, batch_input):
    return py_typecheck.check_type(
        self._model.train_on_batch(batch_input), model_lib.BatchOutput)
