# Lint as: python3
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

import collections

import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib


def model_initializer(model, name=None):
  """Creates an initializer op for all of the model's variables."""
  py_typecheck.check_type(model, model_lib.Model)
  return tf.compat.v1.initializers.variables(
      model.trainable_variables + model.non_trainable_variables +
      model.local_variables,
      name=(name or 'model_initializer'))


@attr.s(eq=False, frozen=True, slots=True)
class ModelWeights(object):
  """A container for the trainable and non-trainable variables of a `Model`.

  Note this does not include the model's local variables.

  It may also be used to hold other values that are parallel to these variables,
  e.g., tensors corresponding to variable values, or updates to model variables.
  """
  trainable = attr.ib()
  non_trainable = attr.ib()

  @classmethod
  def from_model(cls, model):
    py_typecheck.check_type(model, (model_lib.Model, tf.keras.Model))
    return cls(model.trainable_variables, model.non_trainable_variables)

  @classmethod
  def from_tff_result(cls, anon_tuple):
    py_typecheck.check_type(anon_tuple, anonymous_tuple.AnonymousTuple)
    return cls([
        value
        for _, value in anonymous_tuple.iter_elements(anon_tuple.trainable)
    ], [
        value
        for _, value in anonymous_tuple.iter_elements(anon_tuple.non_trainable)
    ])

  def assign_weights_to(self, keras_model):
    """Assign these TFF model weights to the weights of a `tf.keras.Model`.

    Args:
      keras_model: the `tf.keras.Model` object to assign weights to.
    """
    tf.nest.map_structure(lambda var, t: var.assign(t),
                          keras_model.trainable_weights, self.trainable)
    tf.nest.map_structure(lambda var, t: var.assign(t),
                          keras_model.non_trainable_weights, self.non_trainable)


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
    super().__init__()
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
    super().__init__(model)

  def train_on_batch(self, batch_input):
    return py_typecheck.check_type(
        self._model.train_on_batch(batch_input), model_lib.BatchOutput)
