# Copyright 2019, The TensorFlow Federated Authors.
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
"""Utility functions for writing executors."""

from typing import Optional

import federated_language


def reconcile_value_with_type_spec(
    value: object, type_spec: federated_language.Type
) -> federated_language.Type:
  """Reconciles the type of `value` with the given `type_spec`.

  The currently implemented logic only performs reconciliation of `value` and
  `type` for values that implement `federated_language.TypedObject`. Future
  extensions may
  perform reconciliation for a greater range of values; the caller should not
  depend on the limited implementation. This method may fail in case of any
  incompatibility between `value` and `type_spec`. In any case, the method is
  going to fail if the type cannot be determined.

  Args:
    value: An object that represents a value.
    type_spec: An instance of `federated_language.Type`.

  Returns:
    An instance of `federated_language.Type`. If `value` is not a
    `federated_language.TypedObject`, this is
    the same as `type_spec`, which in this case must not be `None`. If `value`
    is a `federated_language.TypedObject`, and `type_spec` is `None`, this is
    simply the type
    signature of `value.` If the `value` is a `federated_language.TypedObject`
    and `type_spec`
    is not `None`, this is `type_spec` to the extent that it is eqiuvalent to
    the type signature of `value`, otherwise an exception is raised.
  """
  if isinstance(value, federated_language.TypedObject):
    return reconcile_value_type_with_type_spec(value.type_signature, type_spec)
  elif type_spec is not None:
    return type_spec
  else:
    raise TypeError(
        'Cannot derive an eager representation for a value of an unknown type.'
    )


def reconcile_value_type_with_type_spec(
    value_type: federated_language.Type,
    type_spec: Optional[federated_language.Type],
) -> federated_language.Type:
  """Reconciles a pair of types.

  Args:
    value_type: An instance of `federated_language.Type`.
    type_spec: An instance of `federated_language.Type`, or `None`.

  Returns:
    Either `value_type` if `type_spec` is `None`, or `type_spec` if `type_spec`
    is not `None` and rquivalent with `value_type`.

  Raises:
    TypeError: If arguments are of incompatible types.
  """
  if type_spec is not None:
    if not value_type.is_equivalent_to(type_spec):
      raise TypeError(
          'Expected a value of type {}, found {}.'.format(type_spec, value_type)
      )
    return type_spec
  else:
    return value_type
