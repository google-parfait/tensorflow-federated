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
from typing import Any, Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object


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
    return structure.Struct((k, to_canonical_value(v)) for k, v in items)
  elif isinstance(value, (tuple, list)):
    return [to_canonical_value(e) for e in value]
  return value


def reconcile_value_with_type_spec(
    value: Any, type_spec: computation_types.Type) -> computation_types.Type:
  """Reconciles the type of `value` with the given `type_spec`.

  The currently implemented logic only performs reconciliation of `value` and
  `type` for values that implement `tff.TypedObject`. Future extensions may
  perform reconciliation for a greater range of values; the caller should not
  depend on the limited implementation. This method may fail in case of any
  incompatibility between `value` and `type_spec`. In any case, the method is
  going to fail if the type cannot be determined.

  Args:
    value: An object that represents a value.
    type_spec: An instance of `tff.Type`.

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
  if isinstance(value, typed_object.TypedObject):
    return reconcile_value_type_with_type_spec(value.type_signature, type_spec)
  elif type_spec is not None:
    return type_spec
  else:
    raise TypeError(
        'Cannot derive an eager representation for a value of an unknown type.')


def reconcile_value_type_with_type_spec(
    value_type: computation_types.Type,
    type_spec: Optional[computation_types.Type]) -> computation_types.Type:
  """Reconciles a pair of types.

  Args:
    value_type: An instance of `tff.Type`.
    type_spec: An instance of `tff.Type`, or `None`.

  Returns:
    Either `value_type` if `type_spec` is `None`, or `type_spec` if `type_spec`
    is not `None` and rquivalent with `value_type`.

  Raises:
    TypeError: If arguments are of incompatible types.
  """
  py_typecheck.check_type(value_type, computation_types.Type)
  if type_spec is not None:
    py_typecheck.check_type(value_type, computation_types.Type)
    if not value_type.is_equivalent_to(type_spec):
      raise TypeError('Expected a value of type {}, found {}.'.format(
          type_spec, value_type))
  return type_spec if type_spec is not None else value_type
