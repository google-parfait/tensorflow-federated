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
"""A library of contruction functions for computation type structures."""

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements


def reduction_op(result_type_spec, element_type_spec):
  """Returns the type of a reduction operator of the form `(<U,T> -> U)`.

  Args:
    result_type_spec: The type of the result of reduction (`U`). An instance of
      computation_types.Type, or something convertible to it.
    element_type_spec: The type of elements to be reduced (`T`). An instance of
      computation_types.Type, or something convertible to it.

  Returns:
    The type of the corresponding reduction operator (`(<U,T> -> U)`).
  """
  result_type_spec = computation_types.to_type(result_type_spec)
  element_type_spec = computation_types.to_type(element_type_spec)
  return computation_types.FunctionType([result_type_spec, element_type_spec],
                                        result_type_spec)


def unary_op(type_spec):
  """Returns the type of an unary operator that operates on `type_spec`.

  Args:
    type_spec: An instance of computation_types.Type, or something convertible
      to it.

  Returns:
    The type of the corresponding unary operator.
  """
  type_spec = computation_types.to_type(type_spec)
  return computation_types.FunctionType(type_spec, type_spec)


def binary_op(type_spec):
  """Returns the type of a binary operator that operates on `type_spec`.

  Args:
    type_spec: An instance of computation_types.Type, or something convertible
      to it.

  Returns:
    The type of the corresponding binary operator.
  """
  type_spec = computation_types.to_type(type_spec)
  return reduction_op(type_spec, type_spec)


def at_server(type_spec):
  """Constructs a federated type of the form `T@SERVER`.

  Args:
    type_spec: An instance of computation_types.Type, or something convertible
      to it.

  Returns:
    The type of the form `T@SERVER` where `T` is the `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  return computation_types.FederatedType(
      type_spec, placements.SERVER, all_equal=True)


def at_clients(type_spec, all_equal=False):
  """Constructs a federated type of the form `{T}@CLIENTS`.

  Args:
    type_spec: An instance of computation_types.Type, or something convertible
      to it.
    all_equal: The optional `all_equal` bit, `False` by default.

  Returns:
    The type of the form `{T}@CLIENTS` (by default) or `T@CLIENTS` (if specified
    by setting the `all_equal` bit), where `T` is the `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  return computation_types.FederatedType(
      type_spec, placements.CLIENTS, all_equal=all_equal)
