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
"""Defines the implementation of the base Computation interface."""

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import function_utils


class ComputationImpl(function_utils.ConcreteFunction):
  """An implementation of the base interface cb.Computation."""

  @classmethod
  def get_proto(cls, value):
    py_typecheck.check_type(value, cls)
    return value._computation_proto  # pylint: disable=protected-access

  def to_building_block(self):
    # TODO(b/161560999): currently destroys annotated type.
    # This should perhaps be fixed by adding `type_parameter` to `from_proto`.
    return building_blocks.ComputationBuildingBlock.from_proto(
        self._computation_proto)

  def to_compiled_building_block(self):
    return building_blocks.CompiledComputation(
        self._computation_proto, type_signature=self.type_signature)

  def __init__(self, computation_proto, context_stack, annotated_type=None):
    """Constructs a new instance of ComputationImpl from the computation_proto.

    Args:
      computation_proto: The protocol buffer that represents the computation, an
        instance of pb.Computation.
      context_stack: The context stack to use.
      annotated_type: Optional, type information with additional annotations
        that replaces the information in `computation_proto.type`.

    Raises:
      TypeError: if `annotated_type` is not `None` and is not compatible with
      `computation_proto.type`.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    type_spec = type_serialization.deserialize_type(computation_proto.type)
    py_typecheck.check_type(type_spec, computation_types.Type)
    if annotated_type is not None:
      if not type_spec.is_assignable_from(annotated_type):
        raise TypeError(
            'annotated_type not compatible with computation_proto.type\n'
            'computation_proto.type: {!s}\n'
            'annotated_type: {!s}'.format(type_spec, annotated_type))
      type_spec = annotated_type

    if not type_spec.is_function():
      raise TypeError('{} is not a functional type, from proto: {}'.format(
          str(type_spec), str(computation_proto)))

    super().__init__(type_spec, context_stack)
    self._computation_proto = computation_proto

  def __hash__(self) -> int:
    return hash(self._computation_proto.SerializeToString(deterministic=True))
