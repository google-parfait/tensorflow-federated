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

from collections.abc import Callable
from typing import Any, NamedTuple, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.models import variable


class ModelWeights(NamedTuple):
  """A container for the trainable and non-trainable variables of a `Model`.

  Note this does not include the model's local variables.

  It may also be used to hold other values that are parallel to these variables,
  e.g., tensors corresponding to variable values, or updates to model variables.
  """

  trainable: Any
  non_trainable: Any

  @classmethod
  def from_model(cls, model):
    py_typecheck.check_type(model, (variable.VariableModel, tf.keras.Model))
    return cls(model.trainable_variables, model.non_trainable_variables)

  @classmethod
  def from_tff_result(cls, struct):
    py_typecheck.check_type(struct, structure.Struct)
    return cls(
        [value for _, value in structure.iter_elements(struct.trainable)],
        [value for _, value in structure.iter_elements(struct.non_trainable)],
    )

  def assign_weights_to(
      self, model: Union[variable.VariableModel, tf.keras.Model]
  ) -> None:
    """Assign these TFF model weights to the weights of a model.

    Args:
      model: A `tf.keras.Model` or `tff.learning.models.VariableModel` instance
        to assign the weights to.
    """
    py_typecheck.check_type(model, (variable.VariableModel, tf.keras.Model))
    if isinstance(model, tf.keras.Model):
      # We do not use `tf.nest.map_structure` here because
      # tf.keras.Model.*weights are always flat lists and we want to be flexible
      # between sequence types (e.g. tuple/list). `tf.nest` wille error if
      # the types differ.
      for var, t in zip(model.trainable_weights, self.trainable):
        var.assign(t)
      for var, t in zip(model.non_trainable_weights, self.non_trainable):
        var.assign(t)
    else:
      tf.nest.map_structure(
          lambda var, t: var.assign(t),
          model.trainable_variables,
          self.trainable,
      )
      tf.nest.map_structure(
          lambda var, t: var.assign(t),
          model.non_trainable_variables,
          self.non_trainable,
      )

  def convert_variables_to_arrays(self) -> 'ModelWeights':
    """Converts any internal `tf.Variable`s to numpy arrays."""

    if not tf.compat.v1.executing_eagerly():
      raise ValueError(
          'Can only convert to numpy array in eager mode outside '
          'a @tf.function.'
      )

    if isinstance(self.trainable, structure.Struct):
      new_trainable = structure.map_structure(np.array, self.trainable)
    else:
      new_trainable = tf.nest.map_structure(np.array, self.trainable)

    if isinstance(self.non_trainable, structure.Struct):
      new_non_trainable = structure.map_structure(np.array, self.non_trainable)
    else:
      new_non_trainable = tf.nest.map_structure(np.array, self.non_trainable)

    return ModelWeights(new_trainable, new_non_trainable)


def weights_type_from_model(
    model: Union[variable.VariableModel, Callable[[], variable.VariableModel]]
) -> computation_types.StructType:
  """Creates a `tff.Type` from a `tff.learning.models.VariableModel` or callable that constructs a model.

  Args:
    model: A `tff.learning.models.VariableModel` instance, or a no-arg callable
      that returns a model.

  Returns:
    A `tff.StructType` representing the TFF type of the `ModelWeights`
    structure for `model`.
  """
  if callable(model):
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    with tf.Graph().as_default():
      model = model()
  py_typecheck.check_type(model, variable.VariableModel)
  model_weights = ModelWeights.from_model(model)

  def _variable_to_type(x: tf.Variable) -> computation_types.Type:
    return computation_types.tensorflow_to_type((x.dtype, x.shape))

  model_weights_type = tf.nest.map_structure(_variable_to_type, model_weights)
  # StructWithPythonType operates recursively, and will preserve the python type
  # information of model.trainable_variables and model.non_trainable_variables.
  return computation_types.StructWithPythonType(
      model_weights_type, ModelWeights
  )
