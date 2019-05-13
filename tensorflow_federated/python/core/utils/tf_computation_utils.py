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
"""Defines utility functions/classes for constructing TF computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types


def get_variables(name, type_spec, **kwargs):
  """Creates a set of variables that matches the given `type_spec`.

  Args:
    name: The common name to use for the scope in which all of the variables are
      to be created.
    type_spec: An instance of `tff.Type` or something convertible to it. The
      type signature may only be composed of tensor types and named tuples,
      possibly nested.
    **kwargs: Additional keyword args to pass to `tf.get_variable` calls.

  Returns:
    Either a single variable when invoked with a tensor TFF type, or a nested
    structure of variables created in the appropriately-named variable scopes
    made up of anonymous tuples if invoked with a named tuple TFF type.

  Raises:
    TypeError: if `type_spec` is not a type signature composed of tensor and
      named tuple TFF types.
  """
  py_typecheck.check_type(name, six.string_types)
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return tf.get_variable(
        name, dtype=type_spec.dtype, shape=type_spec.shape, **kwargs)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    with tf.variable_scope(name):
      return anonymous_tuple.AnonymousTuple([
          (k, get_variables(k if k is not None else str(i), v, **kwargs))
          for i, (k, v) in enumerate(anonymous_tuple.to_elements(type_spec))
      ])
  else:
    raise TypeError(
        'Expected a TFF type signature composed of tensors and named tuples, '
        'found {}.'.format(str(type_spec)))


def assign(target, source):
  """Creates an op that assigns `target` from `source`.

  This utility function provides the exact same behavior as `tf.assign`, but it
  generalizes to a wider class of objects, including ordinary variables as well
  as various types of nested structures.

  Args:
    target: A nested structure composed of variables embedded in containers that
      are compatible with `tf.nest`, or instances of
      `anonymous_tuple.AnonymousTuple`.
    source: A nsested structure composed of tensors, matching that of `target`.

  Returns:
    A single op that represents the assignment.

  Raises:
    TypeError: If types mismatch.
  """
  # TODO(b/113112108): Extend this to containers of mixed types.
  if isinstance(target, anonymous_tuple.AnonymousTuple):
    return tf.group(*anonymous_tuple.flatten(
        anonymous_tuple.map_structure(tf.assign, target, source)))
  else:
    return tf.group(
        *tf.nest.flatten(tf.nest.map_structure(tf.assign, target, source)))


def identity(source):
  """Applies `tf.identity` pointwise to `source`.

  This utility function provides the exact same behavior as `tf.identity`, but
  it generalizes to a wider class of objects, including ordinary tensors,
  variables, as well as various types of nested structures. It would typically
  be used together with `tf.control_dependencies` in non-eager TensorFlow.

  Args:
    source: A nested structure composed of tensors or variables embedded in
      containers that are compatible with `tf.nest`, or
      instances of `anonymous_tuple.AnonymousTuple`. Elements that represent
      variables have their content extracted prior to identity mapping by first
      invoking `tf.Variable.read_value`.

  Returns:
    The result of applying `tf.identity` to read all elements of the `source`
    pointwise, with the same structure as `source`.

  Raises:
    TypeError: If types mismatch.
  """

  def _mapping_fn(x):
    if not tf.is_tensor(x):
      raise TypeError('Expected a tensor, found {}.'.format(
          str(py_typecheck.type_string(type(x)))))
    if hasattr(x, 'read_value'):
      x = x.read_value()
    return tf.identity(x)

  # TODO(b/113112108): Extend this to containers of mixed types.
  if isinstance(source, anonymous_tuple.AnonymousTuple):
    return anonymous_tuple.map_structure(_mapping_fn, source)
  else:
    return tf.nest.map_structure(_mapping_fn, source)
