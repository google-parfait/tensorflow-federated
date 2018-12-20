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

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model


def from_keras_model(keras_model, loss, metrics=None, optimizer=None):
  """Builds a `tensorflow_federated.learning.Model`.

  Args:
    keras_model: a `tf.keras.Model` object that is not compiled.
    loss: a callable that takes two batched tensor parameters, `y_true` and
      `y_pred`, and returns the loss.
    metrics: a list of `tf.keras.metrics.Metric` objects. The value of
      `Metric.result()` for each metric is included in the list of tensors
      returned in `aggregated_outputs()`.
    optimizer: a `tf.keras.optimizer.Optimizer`.

  Returns:
    A `tensorflow_federated.learning.TrainableModel` object iff optimizer is not
    `None`, otherwise a `tensorflow_federated.learning.Model` object.

  Raises:
    TypeError: if keras_model is not an instace of `tf.keras.Model`.
    ValueError: if keras_model was compiled.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled. Use '
                     'from_compiled_keras_model() instead.')
  if optimizer is None:
    return _KerasModel(keras_model, loss, metrics)
  keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  return _TrainableKerasModel(keras_model)


def from_compiled_keras_model(keras_model):
  """Builds a `tensorflow_federated.learning.TrainableModel`.

  Args:
    keras_model: a `tf.keras.Model` object that was compiled.

  Returns:
    A `tensorflow_federated.learning.TrainableModel`.

  Raises:
    TypeError: if keras_model is not an instace of `tf.keras.Model`.
    ValueError: if keras_model was not compiled.
  """
  py_typecheck.check_type(keras_model, tf.keras.Model)
  # Optimizer attribute is only set after calling tf.keras.Model.compile().
  if not hasattr(keras_model, 'optimizer'):
    raise ValueError('`keras_model` must be compiled. Use from_keras_model() '
                     'instead.')
  return _TrainableKerasModel(keras_model)


class _KerasModel(model.Model):
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

    return model.BatchOutput(loss=batch_loss, predictions=predictions)

  @tf.contrib.eager.function(autograph=False)
  def aggregated_outputs(self):
    keras_metrics = getattr(self._keras_model, 'metrics')
    if keras_metrics:
      return collections.OrderedDict(
          [(metric.name, metric.result()) for metric in keras_metrics])
    else:
      return collections.OrderedDict(
          [(metric.name, metric.result()) for metric in self._metrics])

  def make_batch(self, x, y):
    return self.Batch(x=x, y=y)


class _TrainableKerasModel(_KerasModel, model.TrainableModel):
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
