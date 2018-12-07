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

# Dependency imports
import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core import api as fc


def get_variables(name, type_spec, **kwargs):
  """Creates a set of variables that matches the given `type_spec`.

  Args:
    name: The common name to use for the scope in which all of the variables are
      to be created.
    type_spec: An instance of `fc.Type` or something convertible to it. The type
      signature may only be composed of tensor types and named tuples, possibly
      nested.
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
  type_spec = fc.to_type(type_spec)
  py_typecheck.check_type(type_spec, fc.Type)
  if isinstance(type_spec, fc.TensorType):
    return tf.get_variable(
        name, dtype=type_spec.dtype, shape=type_spec.shape, **kwargs)
  elif isinstance(type_spec, fc.NamedTupleType):
    with tf.variable_scope(name):
      return anonymous_tuple.AnonymousTuple(
          [(k, get_variables(k if k is not None else str(i), v, **kwargs))
           for i, (k, v) in enumerate(type_spec.elements)])
  else:
    raise TypeError(
        'Expected a TFF type signature composed of tensors and named tuples, '
        'found {}.'.format(str(type_spec)))
