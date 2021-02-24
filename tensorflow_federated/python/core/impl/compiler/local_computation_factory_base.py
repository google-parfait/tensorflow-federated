# Copyright 2021, The TensorFlow Federated Authors.
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
"""Defines the interface for factories of framework-specific computations."""

import abc
from typing import Tuple

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computation_types


ComputationProtoAndType = Tuple[pb.Computation, computation_types.Type]


class LocalComputationFactory(metaclass=abc.ABCMeta):
  """Interface for factories of backend framework-specific local computations.

  Implementations of this interface encapsulate the logic for constructing local
  computations that are executable on a particular type of backends (such as
  TensorFlow or XLA).
  """

  @abc.abstractmethod
  def create_constant_from_scalar(
      self, value,
      type_spec: computation_types.Type) -> ComputationProtoAndType:
    """Creates a TFF computation returning a constant based on a scalar value.

    The returned computation has the type signature `( -> T)`, where `T` may be
    either a scalar, or a nested structure made up of scalars.

    Args:
      value: A numpy scalar representing the value to return from the
        constructed computation (or to broadcast to all parts of a nested
        structure if `type_spec` is a structured type).
      type_spec: A `computation_types.Type` of the constructed constant. Must be
        either a tensor, or a nested structure of tensors.

    Returns:
      A tuple `(pb.Computation, computation_types.Type)` with the first element
      being a TFF computation with semantics as described above, and the second
      element representing the formal type of that computation.
    """
    raise NotImplementedError

  def create_plus_operator(
      self, type_spec: computation_types.Type) -> ComputationProtoAndType:
    """Creates a TFF computation computing a binary plus operation.

    The returned computation has the type signature `(<T,T> -> T)`, where `T` is
    the `type_spec`.

    Note: If `type_spec` is a `computation_types.StructType`, then
    `operator` will be applied pointwise.

    Args:
      type_spec: A `computation_types.Type` to use as the argument to the
        constructed binary operator; must contain only named tuples and tensor
        types.

    Returns:
      A tuple `(pb.Computation, computation_types.Type)` with the first element
      being a TFF computation with semantics as described above, and the second
      element representing the formal type of that computation.
    """
    raise NotImplementedError

  def create_multiply_operator(
      self, type_spec: computation_types.Type) -> ComputationProtoAndType:
    """Creates a TFF computation computing a binary point-wise multiplication.

    The returned computation has the type signature `(<T,T> -> T)`, where `T` is
    the `type_spec`.

    Note: If `type_spec` is a `computation_types.StructType`, then
    `operator` will be applied pointwise.

    Args:
      type_spec: A `computation_types.Type` to use as the argument to the
        constructed binary operator; must contain only named tuples and tensor
        types.

    Returns:
      A tuple `(pb.Computation, computation_types.Type)` with the first element
      being a TFF computation with semantics as described above, and the second
      element representing the formal type of that computation.
    """
    raise NotImplementedError

  def create_scalar_multiply_operator(
      self, operand_type: computation_types.Type,
      scalar_type: computation_types.TensorType) -> ComputationProtoAndType:
    """Creates a TFF computation multiplying an argument by a scalar.

    Args:
      operand_type: The type of the value to multiply by a scalar.
      scalar_type: The type of the scalar to multiply by.

    Returns:
      A tuple `(pb.Computation, computation_types.Type)` with the first element
      being a TFF computation with semantics as described above, and the second
      element representing the formal type of that computation.
    """
    raise NotImplementedError
