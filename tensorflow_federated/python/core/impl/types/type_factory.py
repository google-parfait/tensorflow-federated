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

from tensorflow_federated.python.core.impl.types import computation_types


def reduction_op(
    result_type_spec: computation_types.Type,
    element_type_spec: computation_types.Type,
) -> computation_types.Type:
  """Returns the type of a reduction operator of the form `(<U,T> -> U)`.

  Args:
    result_type_spec: A `computation_types.Type`, the result of reduction (`U`).
    element_type_spec: A `computation_types.Type`, the type of elements to be
      reduced (`T`).

  Returns:
    The type of the corresponding reduction operator (`(<U,T> -> U)`).
  """
  return computation_types.FunctionType(
      computation_types.StructType([result_type_spec, element_type_spec]),
      result_type_spec,
  )


def unary_op(type_spec: computation_types.Type) -> computation_types.Type:
  """Returns the type of an unary operator that operates on `type_spec`.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    The type of the corresponding unary operator.
  """
  return computation_types.FunctionType(type_spec, type_spec)


def binary_op(type_spec: computation_types.Type) -> computation_types.Type:
  """Returns the type of a binary operator that operates on `type_spec`.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    The type of the corresponding binary operator.
  """
  return reduction_op(type_spec, type_spec)
