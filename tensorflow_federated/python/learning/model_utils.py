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


def from_keras_model(keras_model, loss, metrics=None, optimizer=None):
  """Builds a `tff.learning.Model`.

  Args:
    keras_model: a `tf.keras.Model` object that is not compiled.
    loss: a callable that takes two batched tensor parameters, `y_true` and
      `y_pred`, and returns the loss.
    metrics: a list of `tf.keras.metrics.Metric` objects. The value of
      `Metric.result` for each metric is included in the list of tensors
      returned in `aggregated_outputs`.
    optimizer: a `tf.keras.optimizer.Optimizer`.

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
    return enhance(_KerasModel(keras_model, loss, metrics))
  keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  return enhance(_TrainableKerasModel(keras_model))


def from_compiled_keras_model(keras_model):
  """Builds a `tff.learning.Model`.

  Args:
    keras_model: a `tf.keras.Model` object that was compiled.

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
  return enhance(_TrainableKerasModel(keras_model))


class _KerasModel(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

  def __init__(self, inner_model, loss_func, metrics):
    self._keras_model = inner_model
    self._loss_fn = loss_func
    self._metrics = metrics if metrics is not None else []

  @property
  def trainable_variables(self):
    return self._keras_model.trainable_variables

  @property
  def non_trainable_variables(self):
    return self._keras_model.non_trainable_variables

  @property
  def local_variables(self):
    local_variables = []
    for metric in self._metrics:
      local_variables.extend(metric.variables)
    return local_variables

  @tf.contrib.eager.function
  def forward_pass(self, batch_input, training=True):
    predictions = self._keras_model(batch_input.x, training=training)
    batch_loss = self._loss_fn(y_true=batch_input.y, y_pred=predictions)

    for metric in self._metrics:
      metric.update_state(y_true=batch_input.y, y_pred=predictions)

    return model_lib.BatchOutput(loss=batch_loss, predictions=predictions)

  @tf.contrib.eager.function(autograph=False)
  def report_local_outputs(self):
    keras_metrics = getattr(self._keras_model, 'metrics')
    if keras_metrics:
      return collections.OrderedDict(
          [(metric.name, metric.result()) for metric in keras_metrics])
    else:
      return collections.OrderedDict(
          [(metric.name, metric.result()) for metric in self._metrics])

  def federated_output_computation(self):
    # TODO(b/122116149): Automatically generate federated_output_computation
    # for Keras models by appropriately aggregating the keras.Metrics using
    # the metric's state variables' VariableAggregation properties.
    return None

  @classmethod
  def make_batch(cls, x, y):
    return cls.Batch(x=x, y=y)


class _TrainableKerasModel(_KerasModel, model_lib.TrainableModel):
  """Wrapper class for tf.keras.Models that can be trained."""

  def __init__(self, inner_model):
    super(_TrainableKerasModel, self).__init__(inner_model, inner_model.loss,
                                               inner_model.metrics)
    # NOTE: the Keras optimizer lazily creates variables under-the-hood. Ideally
    # we'd expose them in the `local_variables` property, but there isn't a way
    # to get them currently. Users must use tf.global_variables_initializer().

  @tf.contrib.eager.function
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

  def forward_pass(self, batch, training=True):
    return py_typecheck.check_type(
        self._model.forward_pass(batch, training), model_lib.BatchOutput)

  def report_local_outputs(self):
    return self._model.report_local_outputs()

  @property
  def federated_output_computation(self):
    return self._model.federated_output_computation


class EnhancedTrainableModel(EnhancedModel, model_lib.TrainableModel):

  def __init__(self, model):
    py_typecheck.check_type(model, model_lib.TrainableModel)
    super(EnhancedTrainableModel, self).__init__(model)

  def train_on_batch(self, batch):
    return py_typecheck.check_type(
        self._model.train_on_batch(batch), model_lib.BatchOutput)
