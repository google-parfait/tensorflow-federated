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
"""General utilities specific to the manipulation of tensors and operators."""

import collections
import functools
import operator

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


def check_nested_equal(nested_x, nested_y, eq_fn=operator.eq):
  """Raises error if two nested structures are not equal.

  Nested structures are equal iff they have the same structure and the values at
  each position are equal.

  Args:
    nested_x: an arbitrarily nested structure.
    nested_y: an arbitrarily nested structure.
    eq_fn: a callable of two parameters that returns True iff the two parameters
      are equal.

  Raises:
    ValueError: If the two structures differ in value at any position in the
      nested structure.
  """
  tf.nest.assert_same_structure(nested_x, nested_y)
  flat_x = tf.nest.flatten(nested_x)
  flat_y = tf.nest.flatten(nested_y)
  for x, y in zip(flat_x, flat_y):
    if not eq_fn(x, y):
      raise ValueError('{x} != {y}'.format(x=x, y=y))


# TODO(b/124544593): Rename to_var_odict for consistency.
def to_var_dict(variables):
  """Returns an `OrderedDict` of `vars`, keyed by names.

  Checks that all variables have unique names. The order of the variables
  is preserved, since this may be important if the values are used as a list
  later, as in keras_model.set_weights().

  Args:
    variables: An iterable of variables.

  Returns:
    A `collections.OrderedDict` keyed by variable name with the ":0" removed.

  """
  tuples = []
  seen_names = set()
  for v in variables:
    py_typecheck.check_type(v, tf.Variable, 'v')
    name = v.name
    if name[-2:] != ':0':
      raise ValueError('Variable has unexpected name {}'.format(v.name))
    name = v.name[:-2]

    if name in seen_names:
      raise ValueError('Found multiple variables with the name', name)
    tuples.append((name, v))
    seen_names.add(name)
  return collections.OrderedDict(tuples)


def to_odict(d):
  """Converts d to an OrderedDict with lexically sorted string keys."""
  if isinstance(d, collections.OrderedDict):
    return d
  py_typecheck.check_type(d, dict)
  items = []
  for k, v in d.items():
    py_typecheck.check_type(k, str)
    items.append((k, v))
  return collections.OrderedDict(sorted(items))


@tf.function
def zero_all_if_any_non_finite(structure):
  """Zeroes out all entries in input if any are not finite.

  Args:
    structure: A structure supported by tf.nest.

  Returns:
     A tuple (input, 0) if all entries are finite or the structure is empty, or
     a tuple (zeros, 1) if any non-finite entries were found.
  """
  flat = tf.nest.flatten(structure)
  if not flat:
    return (structure, tf.constant(0))
  flat_bools = [tf.reduce_all(tf.math.is_finite(t)) for t in flat]
  all_finite = functools.reduce(tf.logical_and, flat_bools)
  if all_finite:
    return (structure, tf.constant(0))
  else:
    return (tf.nest.map_structure(tf.zeros_like, structure), tf.constant(1))


def is_scalar(tensor):
  """Returns True iff the given tensor is a scalar.

  Args:
    tensor: The tensor to test for being a scalar.

  Returns:
    True if 'tensor' is a scalar, i.e. all dims are 1, False otherwise.

  Raises:
    TypeError: when the argument is not a tensor.
  """
  if not tf.is_tensor(tensor):
    raise TypeError('Expected a tensor, found "{}".'.format(
        py_typecheck.type_string(type(tensor))))
  return (hasattr(tensor, 'get_shape') and
          all(dim == 1 for dim in tensor.get_shape()))


def _same_dimension(x, y):
  """Determines if two `tf.Dimension`s are the same.

  Args:
    x: a `tf.Dimension` object.
    y: a `tf.Dimension` object.

  Returns:
    True iff `x` and `y` are either both _unknown_ (i.e. `None`), or both have
    the same value.
  """
  if x is None:
    return y is None
  else:
    return y is not None and x.value == y.value


def same_shape(x, y):
  """Determines if two `tf.TensorShape`s are the same.

  Args:
    x: a `tf.TensorShape` object.
    y: a `tf.TensorShape` object.

  Returns:
    True iff `x` and `y` are either both _unknown_ shapes (e.g.
    `tf.TensorShape(None)`) or have each dimension the same.
  """
  if x.ndims != y.ndims:
    return False
  if x.dims is None:
    return y.dims is None
  else:
    return y.dims is not None and all(
        _same_dimension(a, b) for a, b in zip(x.dims, y.dims))
