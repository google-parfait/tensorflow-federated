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

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.types import computation_types

ComputationProtoAndType = tuple[pb.Computation, computation_types.Type]


class LocalComputationFactory(metaclass=abc.ABCMeta):
  """Interface for factories of backend framework-specific local computations.

  Implementations of this interface encapsulate the logic for constructing local
  computations that are executable on a particular type of backend.
  """

  @abc.abstractmethod
  def create_constant_from_scalar(
      self, value, type_spec: computation_types.Type
  ) -> ComputationProtoAndType:
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
