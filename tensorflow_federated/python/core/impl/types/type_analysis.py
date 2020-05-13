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
                            Tuple[Type[computation_types.Type]]]


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
