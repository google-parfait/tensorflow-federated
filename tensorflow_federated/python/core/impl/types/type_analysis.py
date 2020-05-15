# Lint as: python3
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
"""A library of static analysis functions for computation types."""

from typing import Callable, Optional, Tuple, Type, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_transformations

_TypeOrTupleOfTypes = Union[Type[computation_types.Type],
                            Tuple[Type[computation_types.Type], ...]]


def _visit_postorder(type_signature: computation_types.Type,
                     function: Callable[[computation_types.Type], None]):
  py_typecheck.check_type(type_signature, computation_types.Type)

  def _visit(inner_type):
    function(inner_type)
    return inner_type, False

  type_transformations.transform_type_postorder(type_signature, _visit)


def count(
    type_signature: computation_types.Type,
    predicate: Optional[Callable[[computation_types.Type],
                                 bool]] = None) -> int:
  """Returns the number of types in `type_signature` matching `predicate`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    predicate: An optional Python function that takes a type as a parameter and
      returns a boolean value. If `None`, all types are counted.
  """
  counter = 0

  def _function(inner_type):
    nonlocal counter
    counter += 1 if predicate(inner_type) else 0

  _visit_postorder(type_signature, _function)
  return counter


def count_types(type_signature: computation_types.Type,
                types: _TypeOrTupleOfTypes) -> int:
  """Returns the number of instances of `types` in `type_signature`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    types: A `computation_types.Type` type or a tuple of
      `computation_types.Type` types; the same as what is accepted by
      `isinstance`.
  """
  return count(type_signature, lambda x: isinstance(x, types))


def contains_types(type_signature: computation_types.Type,
                   types: _TypeOrTupleOfTypes) -> bool:
  """Checks if `type_signature` contains any instance of `types`.

  Args:
    type_signature: A tree of `computation_type.Type` to test.
    types: A `computation_types.Type` type or a tuple of
      `computation_types.Type` types; the same as what is accepted by
      `isinstance`.

  Returns:
    `True` if `type_signature` contains any instance of `types`, otherwise
    `False`.
  """
  return count_types(type_signature, types) > 0


def contains_federated_types(type_signature):
  """Returns whether or not `type_signature` contains a federated type."""
  return contains_types(type_signature, computation_types.FederatedType)


def contains_tensor_types(type_signature):
  """Returns whether or not `type_signature` contains a tensor type."""
  return contains_types(type_signature, computation_types.TensorType)


def contains_only_types(type_signature: computation_types.Type,
                        types: _TypeOrTupleOfTypes) -> bool:
  """Checks if `type_signature` contains only instances of `types`.

  Args:
    type_signature: A tree of `computation_type.Type` to test.
    types: A `computation_types.Type` type or a tuple of
      `computation_types.Type` types; the same as what is accepted by
      `isinstance`.

  Returns:
    `True` if `type_signature` contains only instance of `types`, otherwise
    `False`.
  """
  return count(type_signature, lambda x: not isinstance(x, types)) == 0


def check_well_formed(type_spec):
  """Checks that `type_spec` represents a well-formed type.

  Performs the following checks of well-formedness for `type_spec`:
    1. If `type_spec` contains a  `computation_types.FederatedType`, checks
    that its `member` contains nowhere in its structure intances
    of `computation_types.FunctionType` or `computation_types.FederatedType`.
    2. If `type_spec` contains a `computation_types.SequenceType`, checks that
    its `element` contains nowhere in its structure instances of
    `computation_types.SequenceType`,  `computation_types.FederatedType`
    or `computation_types.FunctionType`.

  Args:
    type_spec: The type specification to check, either an instance of
      `computation_types.Type` or something convertible to it by
      `computation_types.to_type()`.

  Raises:
    TypeError: if `type_spec` is not a well-formed TFF type.
  """
  # TODO(b/113112885): Reinstate a call to `check_all_abstract_types_are_bound`
  # after revising the definition of well-formedness.
  type_signature = computation_types.to_type(type_spec)

  def _check_for_disallowed_type(type_to_check, disallowed_types):
    """Checks subtree of `type_to_check` for `disallowed_types`."""
    for disallowed_type, disallowed_context in disallowed_types.items():
      if isinstance(type_to_check, disallowed_type):
        raise TypeError('{} has been encountered in the type signature {}. '
                        '{} is disallowed inside of {}.'.format(
                            type_to_check,
                            type_signature,
                            disallowed_type,
                            disallowed_context,
                        ))
    if isinstance(type_to_check, computation_types.FederatedType):
      context = 'federated types (types placed @CLIENT or @SERVER)'
      disallowed_types = {
          **disallowed_types,
          computation_types.FederatedType: context,
          computation_types.FunctionType: context,
      }
    if isinstance(type_to_check, computation_types.SequenceType):
      context = 'sequence types'
      disallowed_types = {
          **disallowed_types,
          computation_types.FederatedType: context,
          computation_types.SequenceType: context,
      }
    return disallowed_types

  type_transformations.visit_preorder(type_signature,
                                      _check_for_disallowed_type, dict())
