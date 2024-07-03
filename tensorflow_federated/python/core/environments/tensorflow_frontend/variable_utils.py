# Copyright 2020, The TensorFlow Federated Authors.
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
"""Library of helper functions for working with TensorFlow `tf.Variable`."""

import contextlib
import operator

import numpy as np
import tensorflow as tf


@contextlib.contextmanager
def record_variable_creation_scope():
  """Creates a single use contextmanager for capture variable creation calls."""
  variable_list = []

  def logging_variable_creator(next_creator, **kwargs):
    variable = next_creator(**kwargs)
    variable_list.append(variable)
    return variable

  with contextlib.ExitStack() as stack:
    stack.enter_context(tf.variable_creator_scope(logging_variable_creator))
    yield variable_list


class TensorVariable:
  """A class that is duck-typed to `tf.Variable` but only uses `tf.Tensor`.

  This class implements the interface contract of `tf.Variable`, for
  documentation see http://www.tensorflow.org/api_docs/python/tf/Variable. To
  be true to the API sometimes arguments are ignored (e.g. `use_locking`).

  This is intended for creating `tff.learning.models.FunctionalModel` and is
  *not* compatible with `tf.distribute` strategies.

  IMPORTANT: this class behaves as if
  `tf.autograph.experimental.Feature.AUTO_CONTROL_DEPS` (ACD) was applied, which
  is the same behavior as inside a `tf.function`. This may have surprising
  side-effects if code authors were not expecting it, but also is more similar
  to standard Python code where the line ordering implies execution ordering.

  IMPORTANT: the `name` attribute does not behave the same as `tf.Variable`.
  Notably, it does not do name deduplication in graph contexts (no `_#` suffix
  is applied), and the returned name string does not refer to the fetchable
  resource from a session.
  """

  def __init__(
      self, initial_value, dtype=None, validate_shape=True, shape=None, **kwargs
  ):
    """For details see https://www.tensorflow.org/api_docs/python/tf/Variable#args_1."""
    if callable(initial_value):
      if dtype is None:
        raise ValueError(
            'When `initial_value` is a callable, `dtype` must be specified.'
        )
      initial_value = initial_value()
    if tf.is_tensor(initial_value):
      self._initial_value = initial_value
    else:
      if dtype is not None:
        self._initial_value = tf.convert_to_tensor(initial_value, dtype)
      else:
        self._initial_value = tf.convert_to_tensor(initial_value)
    self._tensor = self._initial_value
    self._validate_shape = validate_shape
    if shape is None:
      self._shape = self._initial_value.shape
    else:
      if not isinstance(shape, tf.TensorShape):
        shape = tf.TensorShape(shape)
      self._shape = shape
      self._check_shape(self._tensor)
    self._name = kwargs.get('name', 'Variable')
    self._save_slice_info = None

  @property
  def shape(self) -> tf.TensorShape:
    return self._tensor.shape

  @property
  def dtype(self) -> tf.dtypes.DType:
    return self._tensor.dtype

  @property
  def name(self) -> str:
    return self._name

  def assign(self, value, use_locking=False, name=None, read_value=True):
    del use_locking  # Unused.
    value = tf.convert_to_tensor(value)
    self._check_shape(value)
    self._tensor = value
    if tf.executing_eagerly() and not read_value:
      return None
    else:
      return tf.identity(self._tensor, name)

  def _check_shape(self, value):
    if not self._validate_shape:
      return
    if not self._shape.is_compatible_with(tf.TensorShape(value.shape)):
      raise ValueError(
          f'Cannot assign value to variable {self!r}. The TensorVariable shape '
          f'{self._shape}, and the value shape {value.shape} are incompatible.'
      )

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    del use_locking  # Unused.
    value = tf.convert_to_tensor(value)
    self._check_shape(value)
    self._tensor = tf.math.add(self._tensor, value, name=name)
    if tf.executing_eagerly() and not read_value:
      return None
    else:
      return self._tensor

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    del use_locking  # Unused.
    value = tf.convert_to_tensor(value)
    self._check_shape(value)
    self._tensor = tf.math.subtract(self._tensor, value, name=name)
    if tf.executing_eagerly() and not read_value:
      return None
    else:
      return self._tensor

  def get_shape(self):
    return self._tensor.shape

  def read_value(self):
    return self._tensor

  def value(self):
    return self._tensor

  def ref(self):
    return self._tensor.ref()

  def __abs__(self):
    return operator.__abs__(self._tensor)

  def __add__(self, value):
    return operator.__add__(self._tensor, value)

  def __sub__(self, value):
    return operator.__sub__(self._tensor, value)

  def __eq__(self, value):
    return operator.__eq__(self._tensor, value)

  def __ne__(self, value):
    return operator.__ne__(self._tensor, value)

  def __ge__(self, value):
    return operator.__ge__(self._tensor, value)

  def __gt__(self, value):
    return operator.__gt__(self._tensor, value)

  def __le__(self, value):
    return operator.__le__(self._tensor, value)

  def __lt__(self, value):
    return operator.__lt__(self._tensor, value)

  def __getitem__(self, slice_spec):
    return self._tensor[slice_spec]

  def __invert__(self):
    return operator.__invert__(self._tensor)

  def __mul__(self, value):
    return operator.__mul__(self._tensor, value)

  def __neg__(self):
    return operator.__neg__(self._tensor)

  def __truediv__(self, value):
    return operator.__truediv__(self._tensor, value)

  def __floordiv__(self, value):
    return operator.__floordiv__(self._tensor, value)

  def __pow__(self, value):
    return operator.__pow__(self._tensor, value)

  def __hash__(self):
    if not tf.executing_eagerly():
      return hash(self._tensor)
    else:
      raise TypeError(
          f'TensorVariable {self!r} is unhashable. Instead, use '
          'tensorvariable.ref() as the key.'
      )

  def __repr__(self) -> str:
    return f'<TensorVariable: {self._tensor}>'

  def __array__(self):
    return np.array(self._tensor)

  # _save_slice_info is an internal implementation detail of old style
  # tf.compat.v1.get_variable() creation and should not generally be used.
  def _set_save_slice_info(self, save_slice_info):
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info


def create_tensor_variable(next_creator_fn, **kwargs):
  del next_creator_fn  # Unused.
  initial_value = kwargs.pop('initial_value')
  return TensorVariable(initial_value, **kwargs)


def _convert_tensor_variable_to_tensor(value, *args, **kwargs):
  del args  # unused
  del kwargs  # unused
  return value.read_value()


tf.register_tensor_conversion_function(
    TensorVariable, _convert_tensor_variable_to_tensor
)
