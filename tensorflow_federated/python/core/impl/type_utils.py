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
# limitations under the License.
"""Utilities for type conversion, type checking, type inference, etc."""

import collections


from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl.types import type_analysis


def to_canonical_value(value):
  """Converts a Python object to a canonical TFF value for a given type.

  Args:
    value: The object to convert.

  Returns:
    The canonical TFF representation of `value` for a given type.
  """
  if value is None:
    return None
  elif isinstance(value, dict):
    if isinstance(value, collections.OrderedDict):
      items = value.items()
    else:
      items = sorted(value.items())
    return anonymous_tuple.AnonymousTuple(
        (k, to_canonical_value(v)) for k, v in items)
  elif isinstance(value, (tuple, list)):
    return [to_canonical_value(e) for e in value]
  return value


def get_named_tuple_element_type(type_spec, name):
  """Returns the type of a named tuple member.

  Args:
    type_spec: Type specification, either an instance of computation_types.Type
      or something convertible to it by computation_types.to_type().
    name: The string name of the named tuple member.

  Returns:
    The TFF type of the element.

  Raises:
    TypeError: if arguments are of the wrong computation_types.
    ValueError: if the tuple does not have an element with the given name.
  """
  py_typecheck.check_type(name, str)
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
  elements = anonymous_tuple.iter_elements(type_spec)
  for elem_name, elem_type in elements:
    if name == elem_name:
      return elem_type
  raise ValueError('The name \'{}\' of the element does not correspond to any '
                   'of the names {} in the named tuple type.'.format(
                       name, [e[0] for e in elements if e[0]]))


def reconcile_value_with_type_spec(value, type_spec):
  """Reconciles the type of `value` with the given `type_spec`.

  The currently implemented logic only performs reconciliation of `value` and
  `type` for values that implement `tff.TypedObject`. Future extensions may
  perform reconciliation for a greater range of values; the caller should not
  depend on the limited implementation. This method may fail in case of any
  incompatibility between `value` and `type_spec`. In any case, the method is
  going to fail if the type cannot be determined.

  Args:
    value: An object that represents a value.
    type_spec: An instance of `tff.Type` or something convertible to it.

  Returns:
    An instance of `tff.Type`. If `value` is not a `tff.TypedObject`, this is
    the same as `type_spec`, which in this case must not be `None`. If `value`
    is a `tff.TypedObject`, and `type_spec` is `None`, this is simply the type
    signature of `value.` If the `value` is a `tff.TypedObject` and `type_spec`
    is not `None`, this is `type_spec` to the extent that it is eqiuvalent to
    the type signature of `value`, otherwise an exception is raised.

  Raises:
    TypeError: If the `value` type and `type_spec` are incompatible, or if the
      type cannot be determined..
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(value, typed_object.TypedObject):
    return reconcile_value_type_with_type_spec(value.type_signature, type_spec)
  elif type_spec is not None:
    return type_spec
  else:
    raise TypeError(
        'Cannot derive an eager representation for a value of an unknown type.')


def reconcile_value_type_with_type_spec(value_type, type_spec):
  """Reconciles a pair of types.

  Args:
    value_type: An instance of `tff.Type` or something convertible to it. Must
      not be `None`.
    type_spec: An instance of `tff.Type`, something convertible to it, or
      `None`.

  Returns:
    Either `value_type` if `type_spec` is `None`, or `type_spec` if `type_spec`
    is not `None` and rquivalent with `value_type`.

  Raises:
    TypeError: If arguments are of incompatible types.
  """
  value_type = computation_types.to_type(value_type)
  py_typecheck.check_type(value_type, computation_types.Type)
  if type_spec is None:
    return value_type
  else:
    type_spec = computation_types.to_type(type_spec)
    if type_analysis.are_equivalent_types(value_type, type_spec):
      return type_spec
    else:
      raise TypeError('Expected a value of type {}, found {}.'.format(
          type_spec, value_type))
