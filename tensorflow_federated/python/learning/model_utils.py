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

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
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
    py_typecheck.check_type(model, model_lib.Model)
    return cls(
        tensor_utils.to_var_dict(model.trainable_variables),
        tensor_utils.to_var_dict(model.non_trainable_variables))


def from_keras_model(keras_model,
                     dummy_batch,
                     loss,
                     metrics=None,
                     optimizer=None):
  """Builds a `tff.learning.Model` for an example mini batch.

  Args:
    keras_model: a `tf.keras.Model` object that is not compiled.
    dummy_batch: a nested structure of values that are convertible to *batched*
      tensors with the same shapes and types as would be input to `keras_model`.
      The values of the tensors are not important and can be filled with any
      reasonable input value.
    loss: a callable that takes two batched tensor parameters, `y_true` and
      `y_pred`, and returns the loss.
    metrics: (optional) a list of `tf.keras.metrics.Metric` objects. The value
      of `Metric.result` for each metric is included in the list of tensors
      returned in `aggregated_outputs`.
    optimizer: (optional) a `tf.keras.optimizer.Optimizer`. If None, returned
      model cannot be used for training.

  Returns:
    A `tff.learning.Model` object.

  Raises:
    TypeError: if `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: if `keras_model` was compiled.
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
    keras_model: a `tf.keras.Model` object that was compiled.
    dummy_batch: a nested structure of values that are convertible to *batched*
      tensors with the same shapes and types as expected by `forward_pass()`.
      The values of the tensors are not important and can be filled with any
      reasonable input value.

  Returns:
    A `tff.learning.Model`.

  Raises:
    TypeError: if `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: if `keras_model` was *not* compiled.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  # Optimizer attribute is only set after calling tf.keras.Model.compile().
  if not hasattr(keras_model, 'optimizer'):
    raise ValueError('`keras_model` must be compiled. Use from_keras_model() '
                     'instead.')
  return enhance(_TrainableKerasModel(keras_model, dummy_batch))


class _KerasModel(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self, inner_model, dummy_batch, loss_func, metrics):
    self._keras_model = inner_model
    self._loss_fn = loss_func
    self._metrics = metrics

    def _tensor_spec_with_undefined_batch_dim(tensor):
      tensor = tf.convert_to_tensor_or_sparse_tensor(tensor)
      # Remove the batch dimension and leave it unspecified.
      spec = tf.TensorSpec(
          shape=[None] + tensor.shape.dims[1:],
          dtype=tensor.dtype)
      return spec

    self._input_spec = nest.map_structure(_tensor_spec_with_undefined_batch_dim,
                                          dummy_batch)

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
      return self._metrics
    else:
      return self._keras_model.metrics

  @property
  def input_spec(self):
    return self._input_spec

  @tf.contrib.eager.function(autograph=False)
  def forward_pass(self, batch_input, training=True):
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
  def report_local_outputs(self):
    return collections.OrderedDict(
        [(metric.name, metric.result()) for metric in self.get_metrics()])

  def federated_output_computation(self):
    # TODO(b/122116149): Automatically generate federated_output_computation
    # for Keras models by appropriately aggregating the keras.Metrics using
    # the metric's state variables' VariableAggregation properties.
    return None

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
    super(_TrainableKerasModel, self).__init__(
        inner_model, dummy_batch, inner_model.loss, inner_model.metrics)

  @property
  def non_trainable_variables(self):
    return (super(_TrainableKerasModel, self).non_trainable_variables +
            self._keras_model.optimizer.variables())

  @tf.contrib.eager.function(autograph=False)
  def train_on_batch(self, batch_input):
    batch_output = self.forward_pass(batch_input)
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
